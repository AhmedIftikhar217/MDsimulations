#!/usr/bin/env python3
"""
Production NPT molecular dynamics.

Restartable by design.  A 10 ns run is short, but on a shared queue it will
still be interrupted, and a run that cannot resume is a run you have to throw
away.  A checkpoint is written periodically; relaunching the same command
picks up from it automatically and runs only the remaining steps.

Reproducibility notes:
  * Every random stream (integrator, barostat, velocities) is seeded from
    (master_seed, replicate).  Two launches of the same command produce the
    same trajectory on the same hardware and platform.
  * Bitwise reproducibility does NOT survive a change of GPU model, CUDA
    version, or precision setting -- that is a property of floating-point
    hardware, not of this script.  What IS reproducible across machines is the
    ensemble: the distributions, not the individual frames.  Say that in the
    paper rather than claiming exact reproducibility.
  * The trajectory is written with the barostat applied, so box vectors change
    frame to frame; analysis must read them per frame, not assume a fixed box.

Outputs:  traj.dcd, prod_log.csv, checkpoint.chk, final_state.xml,
          prod_summary.json, provenance_production.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdlib  # noqa: E402
from importlib import import_module  # noqa: E402

boxmod = import_module("00_box_size")
equil = import_module("02_equilibrate")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="../runs")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--platform", default=None)
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale the run length (0.001 for a smoke test)")
    a = p.parse_args(argv)

    cfg = mdlib.load_config(a.config)
    out = mdlib.run_dir(cfg, a.output_root, a.replicate)
    seeds = mdlib.derive_seeds(cfg["reproducibility"]["master_seed"], a.replicate)
    cond, prod = cfg["conditions"], cfg["production"]
    T, dt_fs = cond["temperature_K"], prod["timestep_fs"]

    mdlib.banner(f"PRODUCTION  {cfg['metadata']['name']}  replicate {a.replicate}")

    # --- guard: 4 fs is only safe with hydrogen mass repartitioning ---------
    hmass = cfg["integration"]["hydrogen_mass_amu"]
    if dt_fs > 2.5 and hmass < 2.0:
        raise SystemExit(
            f"timestep_fs={dt_fs} with hydrogen_mass_amu={hmass} is unstable. "
            f"Either set integration.hydrogen_mass_amu to 3.0 (HMR) or drop "
            f"production.timestep_fs to 2.0."
        )

    pdb = app.PDBFile(str(out / "system.pdb"))
    system = openmm.XmlSerializer.deserialize((out / "system.xml").read_text())

    # The equilibrated System carried position restraints; production must not.
    # Rebuilding from system.xml and adding only the barostat guarantees that.
    barostat = openmm.MonteCarloBarostat(cond["pressure_bar"] * unit.bar,
                                         T * unit.kelvin,
                                         cond["barostat_interval_steps"])
    barostat.setRandomNumberSeed(seeds["barostat"])
    system.addForce(barostat)

    plat, props = equil.pick_platform(a.platform)
    integ = equil.make_integrator(cfg, T, dt_fs, seeds["integrator"])
    sim = app.Simulation(pdb.topology, system, integ, plat, props)

    total_steps = max(1, int(prod["length_ns"] * 1e6 / dt_fs * a.scale))
    traj_every = max(1, int(prod["trajectory_interval_ps"] * 1000 / dt_fs))
    log_every = max(1, int(prod["log_interval_ps"] * 1000 / dt_fs))
    chk_every = max(1, int(prod["checkpoint_interval_ps"] * 1000 / dt_fs))

    chk = out / "checkpoint.chk"
    if chk.exists():
        sim.loadCheckpoint(str(chk))
        done = sim.currentStep
        print(f"  RESUMING from checkpoint at step {done:,} "
              f"({done * dt_fs / 1e6:.3f} ns)")
        append = True
    else:
        eq_state = openmm.XmlSerializer.deserialize(
            (out / "equilibrated.xml").read_text())
        # NOT setState(): the equilibrated State carries the k_restr global
        # parameter from the equilibration restraints, and this System has no
        # restraint force to receive it. Transfer only the physical state --
        # box, positions, velocities -- which is all that carries over anyway.
        sim.context.setPeriodicBoxVectors(*eq_state.getPeriodicBoxVectors())
        sim.context.setPositions(eq_state.getPositions())
        # Continue with the equilibrated velocities rather than re-drawing:
        # they are already at temperature and already replicate-specific, so
        # redrawing would only inject a thermal discontinuity.
        sim.context.setVelocities(eq_state.getVelocities())
        sim.currentStep = 0
        done = 0
        append = False
        print(f"  starting from equilibrated state")

    remaining = total_steps - done
    if remaining <= 0:
        print(f"  already complete ({done:,} / {total_steps:,} steps)")
        return 0

    print(f"  platform      {plat.getName()} {props}")
    print(f"  atoms         {system.getNumParticles():,}")
    print(f"  timestep      {dt_fs} fs   (H mass {hmass} amu)")
    print(f"  temperature   {T} K,  pressure {cond['pressure_bar']} bar")
    print(f"  total         {total_steps:,} steps = "
          f"{total_steps * dt_fs / 1e6:.3f} ns")
    print(f"  remaining     {remaining:,} steps")
    print(f"  frames        every {traj_every:,} steps "
          f"({prod['trajectory_interval_ps']} ps) -> "
          f"{total_steps // traj_every:,} frames")

    sim.reporters.append(app.DCDReporter(str(out / "traj.dcd"), traj_every,
                                         append=append))
    sim.reporters.append(app.StateDataReporter(
        str(out / "prod_log.csv"), log_every, step=True, time=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        remainingTime=True, totalSteps=total_steps, append=append))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, max(1, remaining // 20), step=True, time=True,
        temperature=True, density=True, speed=True, remainingTime=True,
        totalSteps=total_steps))
    sim.reporters.append(app.CheckpointReporter(str(chk), chk_every))

    print(flush=True)
    t0 = time.time()
    sim.step(remaining)
    wall_s = time.time() - t0

    final = sim.context.getState(getPositions=True, getVelocities=True,
                                 enforcePeriodicBox=True)
    if prod.get("save_final_state", True):
        (out / "final_state.xml").write_text(openmm.XmlSerializer.serialize(final))
    with open(out / "final.pdb", "w") as fh:
        app.PDBFile.writeFile(pdb.topology, final.getPositions(), fh, keepIds=True)

    bv = final.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    v_final = float(np.linalg.det(bv))
    ns_done = remaining * dt_fs / 1e6
    summary = {
        "replicate": a.replicate,
        "steps_total": total_steps,
        "ns_total": total_steps * dt_fs / 1e6,
        "timestep_fs": dt_fs,
        "temperature_K": T,
        "final_volume_nm3": round(v_final, 3),
        "final_concentration_mM": round(
            boxmod.concentration_from_volume(cfg["system"]["n_copies"], v_final), 5),
        "wall_seconds_this_launch": round(wall_s, 1),
        "ns_per_day": round(ns_done / (wall_s / 86400.0), 2) if wall_s > 0 else None,
        "platform": plat.getName(),
        "seeds": seeds,
        "scale": a.scale,
    }
    (out / "prod_summary.json").write_text(json.dumps(summary, indent=2))
    mdlib.write_provenance(cfg, "production", out, {"prod_summary": summary})

    print(f"\n  done: {ns_done:.3f} ns in {wall_s/3600:.2f} h "
          f"= {summary['ns_per_day']} ns/day")
    print(f"  wrote {out}/traj.dcd, prod_log.csv, final_state.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
