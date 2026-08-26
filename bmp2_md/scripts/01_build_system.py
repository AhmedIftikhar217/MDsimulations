#!/usr/bin/env python3
"""
Build the solvated, ionised, parameterised system.

  sequence -> extended peptide -> ACE/NME caps -> 10 dispersed copies
           -> exact cubic box -> OPC water -> 150 mM NaCl -> ff19SB System

Two design decisions that matter scientifically:

1. The box edge comes from `00_box_size.py`, i.e. from the concentration you
   are claiming, checked against the minimum-image requirement.  It is not a
   padding value picked by eye.

2. The 10 copies start DISPERSED, with an enforced minimum centre-of-mass
   separation and no inter-peptide heavy-atom contacts.  If peptides start
   touching, any aggregation you observe is the initial condition, not physics.

Outputs (all in the replicate directory):
  system.pdb           solvated, ionised coordinates
  system.xml           serialised OpenMM System -- the complete force field
                       assignment, so nobody has to rebuild it to reproduce you
  topology.pdb         solute-only topology, for analysis
  build_summary.json   composition, box, achieved concentration
  provenance_build.json
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


# ----------------------------------------------------------------------------
# Step 1: one peptide, extended, with neutral caps
# ----------------------------------------------------------------------------

# ff19SB cap atom names, verified against
# openmm/app/data/amber19/protein.ff19SB.xml:
#   ACE: H1 H2 H3 CH3 C O      external bond through C
#   NME: N H C H1 H2 H3        external bond through N
# A glycine has exactly the heavy-atom skeleton of both caps, so we build
# G-<sequence>-G with ideal geometry and then relabel the flanking glycines.
# This keeps the cap geometry ideal instead of hand-placed.
ACE_FROM_GLY = {"CA": "CH3", "C": "C", "O": "O"}   # drop N
NME_FROM_GLY = {"N": "N", "CA": "C"}               # drop C, O

# Extended / beta-region backbone dihedrals.  Starting every copy from an
# extended chain avoids biasing the run toward any particular fold, which is
# what you want when the question is what the peptides do on their own.
PHI_EXTENDED = -135.0
PSI_EXTENDED = 135.0
OMEGA_TRANS = 180.0


def build_capped_peptide(sequence: str, termini: str) -> app.PDBFile:
    """Build one peptide in an extended conformation, heavy atoms only."""
    import io

    import PeptideBuilder
    from Bio.PDB import PDBIO
    from PeptideBuilder import Geometry

    build_seq = f"G{sequence}G" if termini == "capped" else sequence

    geo = Geometry.geometry(build_seq[0])
    geo.phi, geo.psi_im1, geo.omega = PHI_EXTENDED, PSI_EXTENDED, OMEGA_TRANS
    structure = PeptideBuilder.initialize_res(geo)
    for aa in build_seq[1:]:
        geo = Geometry.geometry(aa)
        geo.phi, geo.psi_im1, geo.omega = PHI_EXTENDED, PSI_EXTENDED, OMEGA_TRANS
        PeptideBuilder.add_residue(structure, geo)

    buf = io.StringIO()
    io_ = PDBIO()
    io_.set_structure(structure)
    io_.save(buf)
    buf.seek(0)

    if termini == "capped":
        buf = io.StringIO(_relabel_caps(buf.read(), n_residues=len(build_seq)))

    pdb = app.PDBFile(buf)
    return pdb


def _relabel_caps(pdb_text: str, n_residues: int) -> str:
    """Turn the flanking glycines into ACE and NME records."""
    out: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith("TER") or line.startswith("END"):
                out.append(line)
            continue

        name = line[12:16].strip()
        resseq = int(line[22:26])
        element = line[76:78].strip() or name[0]

        if element == "H":          # strip all H; OpenMM rebuilds them for ff19SB
            continue

        if resseq == 1:             # N-terminal glycine -> ACE
            if name not in ACE_FROM_GLY:
                continue
            new_name, new_res = ACE_FROM_GLY[name], "ACE"
        elif resseq == n_residues:  # C-terminal glycine -> NME
            if name not in NME_FROM_GLY:
                continue
            new_name, new_res = NME_FROM_GLY[name], "NME"
        else:
            out.append(line)
            continue

        # PDB fixed columns: name is 13-16, resName 18-20.
        padded = f" {new_name:<3}" if len(new_name) < 4 else new_name
        out.append(line[:12] + padded + line[16] + f"{new_res:>3}" + line[20:])

    out.append("END")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Step 2: place N dispersed copies in the box
# ----------------------------------------------------------------------------

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed 3-D rotation (Shoemake's quaternion method).

    Uniform on SO(3) -- sampling Euler angles independently is NOT uniform and
    would bias the starting orientations.
    """
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def min_image(delta: np.ndarray, edge: float) -> np.ndarray:
    """Minimum-image displacement in a cubic box."""
    return delta - edge * np.round(delta / edge)


def place_copies(coords_nm: np.ndarray, n_copies: int, edge_nm: float,
                 min_com_nm: float, min_atom_nm: float,
                 rng: np.random.Generator, max_tries: int = 20000
                 ) -> list[np.ndarray]:
    """Rejection-sample N non-overlapping, randomly oriented copies.

    Rejects on BOTH centre-of-mass separation (keeps copies genuinely
    dispersed) and closest heavy-atom approach under the minimum image
    convention (guarantees no contact at t=0, including across the periodic
    boundary -- a copy near one face is a neighbour of a copy near the other).
    """
    centred = coords_nm - coords_nm.mean(axis=0)
    placed: list[np.ndarray] = []
    coms: list[np.ndarray] = []

    tries = 0
    while len(placed) < n_copies:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Could not place {n_copies} copies in a {edge_nm:.2f} nm box "
                f"with min_com_separation={min_com_nm} nm after {max_tries} "
                f"attempts. Lower box.min_com_separation_nm, or lower "
                f"box.peptide_concentration_mM to enlarge the box."
            )

        rot = random_rotation(rng)
        com = rng.random(3) * edge_nm
        cand = centred @ rot.T + com

        if coms:
            d = np.linalg.norm(min_image(np.array(coms) - com, edge_nm), axis=1)
            if d.min() < min_com_nm:
                continue

        ok = True
        for prev in placed:
            # (n_prev, 1, 3) - (1, n_cand, 3) -> all pairwise, min-imaged
            diff = min_image(prev[:, None, :] - cand[None, :, :], edge_nm)
            if np.sqrt((diff ** 2).sum(-1)).min() < min_atom_nm:
                ok = False
                break
        if not ok:
            continue

        placed.append(cand)
        coms.append(com)

    print(f"  placed {n_copies} copies in {tries} attempts", flush=True)
    return placed


# ----------------------------------------------------------------------------
# Step 3: solvate at a given edge, and correct for addSolvent's under-fill
# ----------------------------------------------------------------------------

def solvate_at(edge_nm: float, single: app.PDBFile, cfg: dict,
               packing_seed: int, quiet: bool = False
               ) -> tuple[app.Modeller, dict, int]:
    """Place the copies, add hydrogens, and solvate in a cubic box of `edge_nm`.

    Re-seeded from `packing_seed` every call, so the placement for a given
    (seed, edge) is identical no matter how many times this runs.
    """
    sysc, boxc = cfg["system"], cfg["box"]
    rng = np.random.default_rng(packing_seed)
    coords = np.array(single.positions.value_in_unit(unit.nanometer))

    placements = place_copies(
        coords, sysc["n_copies"], edge_nm,
        boxc["min_com_separation_nm"], boxc["min_heavy_atom_separation_nm"], rng,
    )

    modeller = app.Modeller(app.Topology(), [])
    for xyz in placements:
        modeller.add(single.topology, xyz * unit.nanometer)

    box_vectors = (openmm.Vec3(edge_nm, 0, 0), openmm.Vec3(0, edge_nm, 0),
                   openmm.Vec3(0, 0, edge_nm)) * unit.nanometer
    modeller.topology.setPeriodicBoxVectors(box_vectors)

    ff = app.ForceField(sysc["forcefield"]["protein"], sysc["forcefield"]["water"])
    modeller.addHydrogens(ff, pH=sysc.get("pH", 7.4))
    n_solute = modeller.topology.getNumAtoms()

    # OPC is a 4-site model with TIP4P-Ew topology.  OpenMM's addSolvent has no
    # 'opc' keyword (verified against 8.6); the correct idiom is to place
    # tip4pew geometry and let amber19/opc.xml supply parameters and the M-site.
    solvent_model = {"opc": "tip4pew", "opc3": "tip3p",
                     "tip4pd": "tip4pew"}.get(sysc["water_model"].lower(),
                                              sysc["water_model"].lower())

    modeller.addSolvent(
        ff, model=solvent_model, boxVectors=box_vectors,
        positiveIon=sysc["salt"]["positive_ion"],
        negativeIon=sysc["salt"]["negative_ion"],
        ionicStrength=sysc["salt"]["concentration_mM"] * unit.millimolar,
        neutralize=sysc["salt"]["neutralize"],
    )
    return modeller, _composition(modeller.topology), n_solute


def corrected_edge(edge_target_nm: float, edge_probe_nm: float,
                   n_waters_probe: int, v_solute_nm3: float,
                   rho_bulk_kg_m3: float) -> tuple[float, float, float]:
    """Edge to build at so the NPT-equilibrated box lands on the design volume.

    addSolvent achieves a packing number density rho_achieved below bulk.  NPT
    then compresses the box until the water reaches bulk density.  Requiring
    that the relaxed solvent volume equal the design solvent volume:

        n_target   = rho_bulk     * (V_target - V_solute)      waters we need
        V_0        = V_solute + n_target / rho_achieved        box to place them in

    Returns (edge_0_nm, rho_achieved_per_nm3, rho_bulk_per_nm3).
    """
    rho_bulk_n = (rho_bulk_kg_m3 * 1000.0 / boxmod.WATER_MOLAR_MASS) \
        * boxmod.N_A / boxmod.NM3_PER_M3
    v_probe_solvent = edge_probe_nm ** 3 - v_solute_nm3
    rho_achieved = n_waters_probe / v_probe_solvent

    v_target_solvent = edge_target_nm ** 3 - v_solute_nm3
    n_target = rho_bulk_n * v_target_solvent
    v_0 = v_solute_nm3 + n_target / rho_achieved
    return v_0 ** (1.0 / 3.0), rho_achieved, rho_bulk_n


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="../runs")
    p.add_argument("--replicate", type=int, default=1)
    a = p.parse_args(argv)

    cfg = mdlib.load_config(a.config)
    out = mdlib.run_dir(cfg, a.output_root, a.replicate)
    seeds = mdlib.derive_seeds(cfg["reproducibility"]["master_seed"], a.replicate)
    rng = np.random.default_rng(seeds["packing"])

    sysc, boxc = cfg["system"], cfg["box"]
    sequence, n_copies = sysc["sequence"], sysc["n_copies"]

    mdlib.banner(f"BUILD  {cfg['metadata']['name']}  replicate {a.replicate}")
    print(f"  config hash   {cfg['_config_hash']}")
    print(f"  output        {out}")
    print(f"  packing seed  {seeds['packing']}")

    # --- box from the concentration, checked against minimum image ----------
    pep = boxmod.Peptide(sequence, sysc["termini"])
    size = boxmod.size_box(
        pep, n_copies,
        boxc["peptide_concentration_mM"], boxc["nonbonded_cutoff_nm"],
        boxc["solvent_buffer_nm"], sysc["salt"]["concentration_mM"],
        sysc["water_model"], boxc["aggregate_scenario"],
    )
    edge = size.edge_nm
    print(f"\n  box edge      {edge:.4f} nm   (binds on {size.binding_constraint})")
    print(f"  box volume    {size.volume_nm3:,.1f} nm^3")
    print(f"  target conc   {boxc['peptide_concentration_mM']} mM")

    if size.binding_constraint == "minimum image":
        print(f"  !! requested concentration would violate minimum image; box "
              f"enlarged, actual concentration is {size.concentration_mM:.3f} mM")

    # --- one peptide --------------------------------------------------------
    print("\n  building peptide ...", flush=True)
    single = build_capped_peptide(sequence, sysc["termini"])
    scoords = np.array(single.positions.value_in_unit(unit.nanometer))
    n_res = single.topology.getNumResidues()
    print(f"  single peptide: {n_res} residues, {single.topology.getNumAtoms()} "
          f"heavy atoms, extended length "
          f"{np.linalg.norm(scoords.max(0) - scoords.min(0)):.2f} nm")

    v_solute = n_copies * pep.volume_nm3

    # --- solvate (pass 1: probe the achieved packing density) ---------------
    print(f"\n  solvating pass 1 at design edge {edge:.4f} nm ...", flush=True)
    modeller, counts, n_solute = solvate_at(edge, single, cfg, seeds["packing"])
    build_edge = edge
    correction = None

    if boxc.get("precompensate_solvent_density", False):
        edge_0, rho_ach, rho_bulk = corrected_edge(
            edge, edge, counts["HOH"], v_solute,
            boxc["reference_water_density_kg_m3"],
        )
        pct = 100.0 * (rho_ach / rho_bulk - 1.0)
        print(f"\n  packing density  {rho_ach:.2f} waters/nm^3  "
              f"({pct:+.2f}% vs bulk {rho_bulk:.2f})")
        print(f"  NPT would relax the design box to "
              f"{edge**3 * rho_ach / rho_bulk:,.1f} nm^3, i.e. "
              f"{boxmod.concentration_from_volume(n_copies, edge**3 * rho_ach / rho_bulk):.3f} mM")
        print(f"  -> rebuilding at edge {edge_0:.4f} nm so the equilibrated box "
              f"lands on {edge**3:,.1f} nm^3")
        print(f"\n  solvating pass 2 ...", flush=True)
        modeller, counts, n_solute = solvate_at(edge_0, single, cfg,
                                                seeds["packing"])
        build_edge = edge_0
        correction = {
            "applied": True,
            "design_edge_nm": round(edge, 6),
            "build_edge_nm": round(edge_0, 6),
            "achieved_packing_density_per_nm3": round(rho_ach, 4),
            "bulk_density_per_nm3": round(rho_bulk, 4),
            "deficit_percent": round(pct, 3),
        }
    else:
        correction = {"applied": False,
                      "warning": "equilibrated volume will fall below the "
                                 "design volume; concentration will read high"}

    box_vectors = (openmm.Vec3(build_edge, 0, 0),
                   openmm.Vec3(0, build_edge, 0),
                   openmm.Vec3(0, 0, build_edge)) * unit.nanometer

    total = modeller.topology.getNumAtoms()
    print(f"\n  solute atoms  {n_solute:,}")
    print(f"  waters        {counts['HOH']:,}")
    print(f"  Na+           {counts.get('NA', 0):,}")
    print(f"  Cl-           {counts.get('CL', 0):,}")
    print(f"  TOTAL ATOMS   {total:,}")
    print(f"  build edge    {build_edge:.4f} nm    "
          f"design edge {edge:.4f} nm")

    # --- parameterise -------------------------------------------------------
    intg = cfg["integration"]
    ff = app.ForceField(sysc["forcefield"]["protein"], sysc["forcefield"]["water"])
    print("\n  creating System ...", flush=True)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=boxc["nonbonded_cutoff_nm"] * unit.nanometer,
        constraints=getattr(app, intg["constraints"]),
        rigidWater=intg["rigid_water"],
        ewaldErrorTolerance=intg["ewald_error_tolerance"],
        hydrogenMass=intg["hydrogen_mass_amu"] * unit.amu,
    )
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(intg["use_dispersion_correction"])

    net_charge = _net_charge(system)
    print(f"  net charge    {net_charge:+.4f} e")
    if abs(net_charge) > 1e-3:
        raise SystemExit(
            f"System carries a net charge of {net_charge:+.4f} e. PME with a "
            f"net charge applies a neutralising background that biases "
            f"energies. Check salt.neutralize in the config."
        )

    # --- write --------------------------------------------------------------
    with open(out / "system.pdb", "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh,
                              keepIds=True)
    with open(out / "system.xml", "w") as fh:
        fh.write(openmm.XmlSerializer.serialize(system))

    solute = app.Modeller(modeller.topology, modeller.positions)
    solute.delete([r for r in solute.topology.residues()
                   if r.name in ("HOH", "NA", "CL", "WAT")])
    with open(out / "topology.pdb", "w") as fh:
        app.PDBFile.writeFile(solute.topology, solute.positions, fh, keepIds=True)

    summary = {
        "replicate": a.replicate,
        "sequence": sequence,
        "n_copies": n_copies,
        "n_residues_per_copy": n_res,
        "peptide_charge_each": pep.charge,
        "peptide_mass_g_per_mol": round(pep.mass, 3),
        "box": {
            "geometry": "cubic",
            # The volume the science is designed around; NPT should relax to it.
            "design_edge_nm": round(edge, 6),
            "design_volume_nm3": round(edge ** 3, 3),
            "design_concentration_mM": round(
                boxmod.concentration_from_volume(n_copies, edge ** 3), 5),
            # The volume actually built, enlarged to absorb addSolvent's
            # under-fill so that NPT lands on the design volume.
            "build_edge_nm": round(build_edge, 6),
            "build_volume_nm3": round(build_edge ** 3, 3),
            "binding_constraint": size.binding_constraint,
            "target_concentration_mM": boxc["peptide_concentration_mM"],
            "min_image_required_nm": round(size.min_image_edge_nm, 4),
            "d_max_nm": round(size.d_max_nm, 4),
            "density_correction": correction,
        },
        "composition": {
            "solute_atoms": n_solute, "waters": counts["HOH"],
            "na": counts.get("NA", 0), "cl": counts.get("CL", 0),
            "total_atoms": total,
        },
        "system_net_charge_e": round(net_charge, 6),
        "seeds": seeds,
    }
    with open(out / "build_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    mdlib.write_provenance(cfg, "build", out, {"build_summary": summary})

    print(f"\n  wrote {out}/system.pdb, system.xml, topology.pdb, "
          f"build_summary.json")
    return 0


def _composition(topology: app.Topology) -> dict[str, int]:
    counts: dict[str, int] = {}
    for res in topology.residues():
        counts[res.name] = counts.get(res.name, 0) + 1
    counts.setdefault("HOH", counts.get("WAT", 0))
    return counts


def _net_charge(system: openmm.System) -> float:
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return sum(force.getParticleParameters(i)[0].value_in_unit(
                unit.elementary_charge) for i in range(force.getNumParticles()))
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
