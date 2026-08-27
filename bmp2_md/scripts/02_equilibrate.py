#!/usr/bin/env python3
"""
Minimise, heat under NVT, then relax the density under NPT.

Protocol, and why each stage exists:

  1. MINIMISE                Remove the clashes that solvation always leaves.
                             Skipping this is the usual cause of a blown-up
                             first step.

  2. NVT, restrained         Heat 100 K -> 310 K with solute heavy atoms held
     heating                 by harmonic restraints.  Water reorganises around
                             a stationary solute instead of dragging it, so
                             the peptides do not diffuse before they are
                             solvated properly.

  3. NPT, restraints         Let the box find its equilibrium density while
     released in stages      the solute is progressively freed.  Releasing all
                             at once shocks the solute; this schedule does not.

The last NPT window is the density check: it reports the equilibrated box
volume against the DESIGN volume from 00_box_size.py, and therefore the
concentration you are actually simulating.  That comparison is the whole
point of the density pre-compensation in the build step, so it is checked
here rather than assumed.

Outputs:  equilibrated.xml (full State), equilibrated.pdb, equil_log.csv,
          equil_summary.json, provenance_equilibrate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdlib  # noqa: E402
from importlib import import_module  # noqa: E402

boxmod = import_module("00_box_size")

SOLVENT_RESIDUES = {"HOH", "WAT", "NA", "CL", "K", "MG", "CA"}


def solute_heavy_atoms(topology: app.Topology) -> list[int]:
    return [a.index for a in topology.atoms()
            if a.residue.name not in SOLVENT_RESIDUES
            and a.element is not None and a.element.symbol != "H"]


def add_position_restraints(system: openmm.System, positions, indices: list[int]
                            ) -> openmm.CustomExternalForce:
    """Flat harmonic restraints whose strength is a tunable global parameter.

    Using a global parameter (rather than per-particle k) means the schedule
    can be changed with context.setParameter, with no System rebuild and no
    loss of velocities between stages.  periodicdistance() keeps the restraint
    correct for an atom that wraps across the boundary.
    """
    force = openmm.CustomExternalForce(
        "k_restr*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k_restr",
                             0.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    for p in ("x0", "y0", "z0"):
        force.addPerParticleParameter(p)
    pos = positions.value_in_unit(unit.nanometer)
    for i in indices:
        force.addParticle(i, [pos[i][0], pos[i][1], pos[i][2]])
    system.addForce(force)
    return force


def make_integrator(cfg: dict, temperature_K: float, timestep_fs: float,
                    seed: int) -> openmm.LangevinMiddleIntegrator:
    """LangevinMiddle = BAOAB splitting: the best configurational sampling
    accuracy per step of the Langevin integrators OpenMM ships."""
    integ = openmm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        cfg["conditions"]["friction_per_ps"] / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    integ.setRandomNumberSeed(seed)
    return integ


def pick_platform(requested: str | None) -> tuple[openmm.Platform, dict]:
    """Prefer CUDA with mixed precision: single precision for forces, double
    for accumulation.  Fast, and accurate enough that energy conservation is
    not the limiting error."""
    order = [requested] if requested else ["CUDA", "OpenCL", "CPU"]
    for name in order:
        try:
            plat = openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
        props = {"Precision": "mixed"} if name in ("CUDA", "OpenCL") else {}
        if name == "CUDA":
            props["DeterministicForces"] = "true"
        return plat, props
    raise SystemExit(f"No usable OpenMM platform among {order}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="runs",
                   help="run outputs go here; paths are relative to the "
                        "directory you launch from (intended: bmp2_md/)")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--platform", default=None)
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale all stage lengths (0.001 for a smoke test)")
    a = p.parse_args(argv)

    cfg = mdlib.load_config(a.config)
    out = mdlib.run_dir(cfg, a.output_root, a.replicate)
    seeds = mdlib.derive_seeds(cfg["reproducibility"]["master_seed"], a.replicate)

    mdlib.banner(f"EQUILIBRATE  {cfg['metadata']['name']}  replicate {a.replicate}")

    pdb = app.PDBFile(str(out / "system.pdb"))
    system = openmm.XmlSerializer.deserialize((out / "system.xml").read_text())
    build = json.loads((out / "build_summary.json").read_text())

    plat, props = pick_platform(a.platform)
    print(f"  platform      {plat.getName()} {props}")
    print(f"  atoms         {system.getNumParticles():,}")

    cond, eq = cfg["conditions"], cfg["equilibration"]
    T = cond["temperature_K"]

    restrained = solute_heavy_atoms(pdb.topology)
    print(f"  restrained    {len(restrained):,} solute heavy atoms")
    restraint = add_position_restraints(system, pdb.positions, restrained)

    # ---------------------------------------------------------------- minimise
    integ = make_integrator(cfg, eq["nvt"]["start_temperature_K"],
                            eq["nvt"]["timestep_fs"], seeds["integrator"])
    sim = app.Simulation(pdb.topology, system, integ, plat, props)
    sim.context.setPositions(pdb.positions)
    sim.context.setParameter("k_restr", eq["nvt"]["restraint_force_constant"]
                             * unit.kilojoule_per_mole / unit.nanometer**2)

    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"\n  [1/3] minimising ... initial PE {e0.value_in_unit(unit.kilojoule_per_mole):,.0f} kJ/mol",
          flush=True)
    sim.minimizeEnergy(
        tolerance=cfg["minimization"]["tolerance_kj_per_mol_nm"]
        * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=cfg["minimization"]["max_iterations"],
    )
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"        final PE   {e1.value_in_unit(unit.kilojoule_per_mole):,.0f} kJ/mol")

    # ------------------------------------------------------------- NVT heating
    nvt_steps = max(1, int(eq["nvt"]["length_ps"] * 1000 /
                           eq["nvt"]["timestep_fs"] * a.scale))
    print(f"\n  [2/3] NVT heating {eq['nvt']['start_temperature_K']:.0f} -> {T:.0f} K"
          f"  ({nvt_steps:,} steps)", flush=True)
    sim.context.setVelocitiesToTemperature(
        eq["nvt"]["start_temperature_K"] * unit.kelvin, seeds["velocities"])
    sim.reporters.append(app.StateDataReporter(
        str(out / "equil_nvt.csv"), max(1, nvt_steps // 100), step=True,
        time=True, potentialEnergy=True, temperature=True, speed=True))

    n_chunks = 20
    for i in range(n_chunks):
        target = eq["nvt"]["start_temperature_K"] + \
            (T - eq["nvt"]["start_temperature_K"]) * (i + 1) / n_chunks
        integ.setTemperature(target * unit.kelvin)
        sim.step(max(1, nvt_steps // n_chunks))
    print(f"        reached {integ.getTemperature().value_in_unit(unit.kelvin):.1f} K")

    # ------------------------------------------- NPT, restraints released
    state = sim.context.getState(getPositions=True, getVelocities=True)
    barostat = openmm.MonteCarloBarostat(
        cond["pressure_bar"] * unit.bar, T * unit.kelvin,
        cond["barostat_interval_steps"])
    barostat.setRandomNumberSeed(seeds["barostat"])
    system.addForce(barostat)

    integ2 = make_integrator(cfg, T, eq["npt"]["timestep_fs"], seeds["integrator"])
    sim = app.Simulation(pdb.topology, system, integ2, plat, props)
    sim.context.setState(state)

    schedule = eq["npt"]["restraint_schedule"]
    total_ps = sum(d for _, d in schedule)
    print(f"\n  [3/3] NPT, {total_ps:.0f} ps in {len(schedule)} restraint stages",
          flush=True)
    sim.reporters.append(app.StateDataReporter(
        str(out / "equil_npt.csv"), 500, step=True, time=True,
        potentialEnergy=True, temperature=True, density=True, volume=True,
        speed=True))

    volumes_final_stage: list[float] = []
    for k, dur_ps in schedule:
        steps = max(1, int(dur_ps * 1000 / eq["npt"]["timestep_fs"] * a.scale))
        sim.context.setParameter(
            "k_restr", k * unit.kilojoule_per_mole / unit.nanometer**2)
        # Sample the volume only in the unrestrained window: a restrained
        # solute biases the density, so earlier stages must not enter the mean.
        if k == 0.0:
            n_samples = 50
            per = max(1, steps // n_samples)
            for _ in range(n_samples):
                sim.step(per)
                st = sim.context.getState()
                bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                volumes_final_stage.append(float(np.linalg.det(bv)))
        else:
            sim.step(steps)
        st = sim.context.getState(getEnergy=True)
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        print(f"        k={k:>7.1f} kJ/mol/nm^2  {dur_ps:>6.0f} ps  "
              f"V={np.linalg.det(bv):>9.1f} nm^3", flush=True)

    # ------------------------------------------------- density / volume check
    n_copies = cfg["system"]["n_copies"]
    design_V = build["box"]["design_volume_nm3"]
    eq_V = float(np.mean(volumes_final_stage)) if volumes_final_stage else float("nan")
    eq_V_sd = float(np.std(volumes_final_stage)) if volumes_final_stage else float("nan")
    eq_C = boxmod.concentration_from_volume(n_copies, eq_V)
    dev = 100.0 * (eq_V - design_V) / design_V
    tol = cfg["box"].get("volume_tolerance_percent", 2.0)

    print(f"\n  VOLUME CHECK")
    print(f"    design volume       {design_V:,.1f} nm^3"
          f"   ({build['box']['design_concentration_mM']:.3f} mM)")
    print(f"    equilibrated volume {eq_V:,.1f} +/- {eq_V_sd:.1f} nm^3"
          f"   ({eq_C:.3f} mM)")
    print(f"    deviation           {dev:+.2f} %   (tolerance +/-{tol:.1f} %)")
    if abs(dev) > tol:
        factor = (design_V / eq_V) ** (1.0 / 3.0)
        print(f"    !! OUT OF TOLERANCE. Rebuild with the box edge multiplied by "
              f"{factor:.5f}, or report {eq_C:.3f} mM as the actual concentration.")
    else:
        print(f"    OK -- report {eq_C:.3f} mM as the simulated concentration.")

    # ------------------------------------------------------------------ write
    final = sim.context.getState(getPositions=True, getVelocities=True,
                                 getParameters=True, enforcePeriodicBox=True)
    (out / "equilibrated.xml").write_text(openmm.XmlSerializer.serialize(final))
    with open(out / "equilibrated.pdb", "w") as fh:
        app.PDBFile.writeFile(pdb.topology, final.getPositions(), fh, keepIds=True)

    summary = {
        "replicate": a.replicate,
        "design_volume_nm3": design_V,
        "equilibrated_volume_nm3": round(eq_V, 3),
        "equilibrated_volume_sd_nm3": round(eq_V_sd, 3),
        "equilibrated_concentration_mM": round(eq_C, 5),
        "deviation_percent": round(dev, 4),
        "within_tolerance": bool(abs(dev) <= tol),
        "final_temperature_K": T,
        "seeds": seeds,
        "scale": a.scale,
    }
    (out / "equil_summary.json").write_text(json.dumps(summary, indent=2))
    mdlib.write_provenance(cfg, "equilibrate", out, {"equil_summary": summary})
    print(f"\n  wrote {out}/equilibrated.xml, equilibrated.pdb, equil_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
