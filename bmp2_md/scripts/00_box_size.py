#!/usr/bin/env python3
"""
Periodic box sizing for a multi-copy peptide simulation.

The box edge is fixed by TWO independent physical constraints, and a defensible
box must satisfy BOTH:

  (A) CONCENTRATION.  The box volume *defines* the peptide concentration you are
      simulating.  This is the number that goes in the Methods section and the
      number a reviewer will check against your experiment:

          V = N_pep / (N_A * C)

  (B) MINIMUM IMAGE.  A solute must never see its own periodic image.  For a
      system that can aggregate, the relevant size is the largest assembly you
      expect to form, not the size of one peptide:

          L >= D_max + 2 * (r_cut + buffer)      and      L >= 2 * r_cut

      D_max is the longest dimension of the worst-case aggregate; the buffer is
      the solvent padding that keeps two images separated by real water.

Whichever constraint gives the larger edge is the one that binds.  This script
reports both, says which one binds, and then reports the derived system size
(waters, ions, atoms) and a wall-clock estimate, so the choice is reproducible
from the numbers rather than from a rule of thumb.

Stdlib only, by design: it must run anywhere, including a login node with no
conda environment loaded.

Usage:
    python 00_box_size.py --config ../config/bmp2_p24_10mer.yaml
    python 00_box_size.py --sequence KIPKASSVPTELSAISTLYL --n-copies 10 --scan
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Physical constants (CODATA 2018) and solvent properties
# ----------------------------------------------------------------------------

N_A = 6.02214076e23          # 1/mol, exact by SI definition
NM3_PER_LITRE = 1.0e24       # 1 L = 1e-3 m^3 = 1e24 nm^3

# TIP3P/OPC bulk water at 300 K, 1 bar.  Experimental density of water is
# 996.5 kg/m^3 at 300 K; TIP3P reproduces ~980-1000 kg/m^3 and OPC ~997 kg/m^3.
# The resulting number density is what sets how many waters the solvation step
# will actually place.
WATER_DENSITY_KG_M3 = 997.0
WATER_MOLAR_MASS = 18.01528  # g/mol
# molecules per nm^3.   1 m^3 = 1e27 nm^3.
#   997 kg/m^3 -> 997e3 g/m^3 / 18.015 g/mol = 55341 mol/m^3
#              -> x N_A = 3.333e28 /m^3  -> /1e27 = 33.33 molecules/nm^3
NM3_PER_M3 = 1.0e27
WATER_NUMBER_DENSITY = (WATER_DENSITY_KG_M3 * 1000.0 / WATER_MOLAR_MASS) * N_A / NM3_PER_M3

# Partial specific volume of a folded/compact polypeptide, Cohn & Edsall.
# 0.73 cm^3/g is the standard protein value; peptides sit in 0.70-0.75.
PARTIAL_SPECIFIC_VOLUME_CM3_G = 0.73

# Geometry of extended polypeptide, from fibre diffraction of beta structure.
RISE_PER_RESIDUE_BETA_NM = 0.335   # axial rise per residue in a beta strand
INTERSTRAND_SPACING_NM = 0.48      # H-bonded strand-to-strand spacing in a sheet
RISE_PER_RESIDUE_HELIX_NM = 0.15   # axial rise per residue in an alpha helix

# ----------------------------------------------------------------------------
# Residue chemistry.  Masses are average (not monoisotopic) residue masses,
# i.e. free amino acid minus one water, in the protonation state dominant at
# pH 7.4.  Atom counts are ALL-ATOM counts as Amber will build them (explicit H).
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Residue:
    code: str
    name: str
    mass: float      # g/mol, residue mass at pH 7.4
    atoms: int       # all-atom count at pH 7.4
    charge: int      # formal charge at pH 7.4

RESIDUES: dict[str, Residue] = {
    r.code: r for r in [
        Residue("G", "GLY",  57.0519,  7,  0),
        Residue("A", "ALA",  71.0788, 10,  0),
        Residue("S", "SER",  87.0782, 11,  0),
        Residue("P", "PRO",  97.1167, 14,  0),
        Residue("V", "VAL",  99.1326, 16,  0),
        Residue("T", "THR", 101.1051, 14,  0),
        Residue("C", "CYS", 103.1388, 11,  0),   # neutral thiol (CYS), not CYM/CYX
        Residue("L", "LEU", 113.1594, 19,  0),
        Residue("I", "ILE", 113.1594, 19,  0),
        Residue("N", "ASN", 114.1038, 14,  0),
        Residue("D", "ASP", 114.0806, 12, -1),   # ASP, deprotonated
        Residue("Q", "GLN", 128.1307, 17,  0),
        Residue("K", "LYS", 129.1820, 22, +1),   # LYS, protonated
        Residue("E", "GLU", 128.1075, 15, -1),   # GLU, deprotonated
        Residue("M", "MET", 131.1926, 17,  0),
        Residue("H", "HIS", 137.1411, 17,  0),   # HIE/HID, neutral (pKa ~6.0)
        Residue("F", "PHE", 147.1766, 20,  0),
        Residue("R", "ARG", 157.1950, 24, +1),   # ARG, protonated
        Residue("Y", "TYR", 163.1760, 21,  0),
        Residue("W", "TRP", 186.2132, 24,  0),
    ]
}

# Terminal caps.  Free (charged) termini add NH3+ / COO-; ACE/NME neutral caps
# are the usual choice for a peptide excised from a larger protein, because a
# free charged terminus is an artefact that is not present in the parent chain.
TERMINI = {
    #  name         extra_mass  extra_atoms  extra_charge   note
    "free":  dict(mass=18.0153, atoms=3,  charge=0,  desc="NH3+ / COO- (zwitterionic, net 0)"),
    "capped": dict(mass=57.0519 + 15.0146 + 14.027,
                   atoms=6 + 6, charge=0,
                   desc="ACE (CH3CO-) / NME (-NHCH3), neutral"),
}
# free termini: +H (N-term) +OH (C-term) = +H2O mass, +3 atoms (2H on N, O+H on C
# is already handled by the +H2O bookkeeping), net charge NH3+ (+1) + COO- (-1) = 0.
# capped: ACE = C(=O)CH3 -> 6 atoms, 43.04 g/mol ; NME = NH-CH3 -> 6 atoms, 30.05 g/mol


@dataclass
class Peptide:
    sequence: str
    termini: str = "capped"

    def __post_init__(self) -> None:
        self.sequence = self.sequence.strip().upper().replace(" ", "")
        bad = sorted(set(self.sequence) - set(RESIDUES))
        if bad:
            raise ValueError(f"Unknown one-letter residue code(s): {bad}")
        if self.termini not in TERMINI:
            raise ValueError(f"termini must be one of {sorted(TERMINI)}")

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    @property
    def mass(self) -> float:
        """Average molecular mass in g/mol at pH 7.4, including termini."""
        if self.termini == "capped":
            extra = 43.0446 + 30.0500   # ACE + NME
        else:
            extra = 18.0153             # +H2O for free termini
        return sum(RESIDUES[c].mass for c in self.sequence) + extra

    @property
    def n_atoms(self) -> int:
        """All-atom count as Amber builds it."""
        extra = 12 if self.termini == "capped" else 3
        return sum(RESIDUES[c].atoms for c in self.sequence) + extra

    @property
    def charge(self) -> int:
        """Formal net charge at pH 7.4."""
        return sum(RESIDUES[c].charge for c in self.sequence)

    @property
    def volume_nm3(self) -> float:
        """Excluded volume from the partial specific volume, in nm^3."""
        # v_bar [cm^3/g] * M [g/mol] / N_A  ->  cm^3 ; 1 cm^3 = 1e21 nm^3
        return PARTIAL_SPECIFIC_VOLUME_CM3_G * self.mass / N_A * 1.0e21

    @property
    def extended_length_nm(self) -> float:
        """End-to-end length of the fully extended (beta) conformation."""
        return self.n_residues * RISE_PER_RESIDUE_BETA_NM

    @property
    def helix_length_nm(self) -> float:
        return self.n_residues * RISE_PER_RESIDUE_HELIX_NM


# ----------------------------------------------------------------------------
# Constraint (A): concentration
# ----------------------------------------------------------------------------

def volume_from_concentration(n_copies: int, conc_mM: float) -> float:
    """Box volume in nm^3 that puts `n_copies` peptides at `conc_mM` millimolar."""
    conc_M = conc_mM * 1.0e-3                     # mol/L
    volume_L = n_copies / (N_A * conc_M)          # L
    return volume_L * NM3_PER_LITRE               # nm^3


def concentration_from_volume(n_copies: int, volume_nm3: float) -> float:
    """Inverse: millimolar concentration implied by a box volume in nm^3."""
    volume_L = volume_nm3 / NM3_PER_LITRE
    return n_copies / (N_A * volume_L) * 1.0e3


# ----------------------------------------------------------------------------
# Constraint (B): minimum image
# ----------------------------------------------------------------------------

@dataclass
class AggregateGeometry:
    """Worst-case dimensions of the assembly the box has to contain."""
    dispersed_nm: float      # one extended peptide, copies far apart
    globular_nm: float       # all copies collapsed into one compact droplet
    sheet_nm: float          # all copies in one extended beta sheet (worst case)

    @property
    def worst_case_nm(self) -> float:
        return max(self.dispersed_nm, self.globular_nm, self.sheet_nm)


def aggregate_geometry(pep: Peptide, n_copies: int) -> AggregateGeometry:
    """
    Three bounding scenarios for the largest solute dimension D_max.

    globular: all copies fuse into one sphere at protein packing density.
    sheet:    all copies line up as extended strands in a single beta sheet --
              the largest structure this many copies can physically make, and
              therefore the conservative bound for an aggregation study.
    """
    dispersed = pep.extended_length_nm

    total_volume = n_copies * pep.volume_nm3
    radius = (3.0 * total_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    globular = 2.0 * radius

    sheet_long = pep.extended_length_nm
    sheet_wide = n_copies * INTERSTRAND_SPACING_NM
    sheet = math.hypot(sheet_long, sheet_wide)   # diagonal of the ribbon

    return AggregateGeometry(dispersed, globular, sheet)


def min_image_edge(d_max_nm: float, cutoff_nm: float, buffer_nm: float) -> float:
    """
    Smallest cubic edge that satisfies the minimum-image convention.

    Two conditions:
      L >= D_max + 2*(r_cut + buffer)   solute never sees its own image through
                                        the nonbonded interaction
      L >= 2*r_cut                      direct-space PME sum is minimum-imageable
    """
    return max(d_max_nm + 2.0 * (cutoff_nm + buffer_nm), 2.0 * cutoff_nm)


# ----------------------------------------------------------------------------
# Derived system size
# ----------------------------------------------------------------------------

@dataclass
class SystemSize:
    edge_nm: float
    volume_nm3: float
    concentration_mM: float
    n_waters: int
    n_cation: int
    n_anion: int
    n_atoms: int
    solute_charge: int
    water_sites: int
    binding_constraint: str
    d_max_nm: float
    min_image_edge_nm: float
    conc_edge_nm: float
    notes: list[str] = field(default_factory=list)


def build_system_size(
    pep: Peptide,
    n_copies: int,
    edge_nm: float,
    salt_mM: float,
    water_model: str,
    binding_constraint: str,
    d_max_nm: float,
    min_image_edge_nm: float,
    conc_edge_nm: float,
) -> SystemSize:
    volume = edge_nm ** 3
    solvent_volume = volume - n_copies * pep.volume_nm3
    n_waters = int(round(WATER_NUMBER_DENSITY * solvent_volume))

    # Ion counting.  OpenMM's Modeller.addSolvent sets the salt count from the
    # number of water molecules, not from the box volume, because the ions
    # replace waters:   n_salt = n_waters * ionicStrength / 55.4 M
    n_salt = int(round(n_waters * (salt_mM * 1.0e-3) / 55.4))

    solute_charge = n_copies * pep.charge
    # neutralising counter-ions go on top of the background salt
    n_cation = n_salt + max(0, -solute_charge)
    n_anion = n_salt + max(0, solute_charge)

    sites = {"tip3p": 3, "tip3pfb": 3, "spce": 3, "tip4pew": 4, "opc": 4, "tip4pd": 4}
    water_sites = sites.get(water_model.lower().replace("-", "").replace("_", ""), 3)

    n_atoms = n_copies * pep.n_atoms + n_waters * water_sites + n_cation + n_anion

    return SystemSize(
        edge_nm=edge_nm,
        volume_nm3=volume,
        concentration_mM=concentration_from_volume(n_copies, volume),
        n_waters=n_waters,
        n_cation=n_cation,
        n_anion=n_anion,
        n_atoms=n_atoms,
        solute_charge=solute_charge,
        water_sites=water_sites,
        binding_constraint=binding_constraint,
        d_max_nm=d_max_nm,
        min_image_edge_nm=min_image_edge_nm,
        conc_edge_nm=conc_edge_nm,
    )


def size_box(
    pep: Peptide,
    n_copies: int,
    conc_mM: float,
    cutoff_nm: float,
    buffer_nm: float,
    salt_mM: float,
    water_model: str,
    aggregate_scenario: str = "sheet",
) -> SystemSize:
    """Apply both constraints and return the box that satisfies the binding one."""
    conc_volume = volume_from_concentration(n_copies, conc_mM)
    conc_edge = conc_volume ** (1.0 / 3.0)

    geom = aggregate_geometry(pep, n_copies)
    d_max = {
        "dispersed": geom.dispersed_nm,
        "globular": geom.globular_nm,
        "sheet": geom.sheet_nm,
        "worst": geom.worst_case_nm,
    }[aggregate_scenario]
    mi_edge = min_image_edge(d_max, cutoff_nm, buffer_nm)

    if mi_edge > conc_edge:
        edge, binding = mi_edge, "minimum image"
    else:
        edge, binding = conc_edge, "concentration"

    return build_system_size(
        pep, n_copies, edge, salt_mM, water_model,
        binding, d_max, mi_edge, conc_edge,
    )


# ----------------------------------------------------------------------------
# Wall-clock estimate
# ----------------------------------------------------------------------------

# Throughput anchors for OpenMM 8 / CUDA, PME, 0.9-1.0 nm cutoff, in
# (ns/day) * (atoms) -- roughly constant because MD throughput is ~1/N.
# Calibrated from published OpenMM benchmarks; treat as order-of-magnitude.
GPU_THROUGHPUT = {          # (ns/day * atoms) at 2 fs, no HMR
    "A100": 1.55e7,
    "V100": 8.5e6,
    "RTX3090": 8.0e6,
    "P100": 4.0e6,
    "T4": 2.4e6,
}


def estimate_walltime(n_atoms: int, ns: float, gpu: str, timestep_fs: float) -> dict:
    base = GPU_THROUGHPUT.get(gpu.upper(), GPU_THROUGHPUT["V100"])
    ns_per_day = base / n_atoms * (timestep_fs / 2.0)
    return {
        "gpu": gpu,
        "timestep_fs": timestep_fs,
        "ns_per_day": ns_per_day,
        "hours_for_run": ns / ns_per_day * 24.0 if ns_per_day > 0 else float("inf"),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def report(pep: Peptide, n_copies: int, s: SystemSize, prod_ns: float,
           gpu: str, timestep_fs: float, cutoff_nm: float, buffer_nm: float,
           salt_mM: float, water_model: str) -> None:
    geom = aggregate_geometry(pep, n_copies)
    w = estimate_walltime(s.n_atoms, prod_ns, gpu, timestep_fs)

    print("=" * 74)
    print("  PERIODIC BOX SIZING")
    print("=" * 74)
    print()
    print("PEPTIDE")
    print(f"  sequence              {pep.sequence}")
    print(f"  residues              {pep.n_residues}")
    print(f"  termini               {pep.termini}  ({TERMINI[pep.termini]['desc']})")
    print(f"  molecular mass        {pep.mass:,.2f} g/mol")
    print(f"  formal charge (pH7.4) {pep.charge:+d} e  per copy")
    print(f"  all-atom count        {pep.n_atoms} atoms per copy")
    print(f"  excluded volume       {pep.volume_nm3:.2f} nm^3 per copy"
          f"  (v_bar = {PARTIAL_SPECIFIC_VOLUME_CM3_G} cm^3/g)")
    print(f"  extended length       {pep.extended_length_nm:.2f} nm (beta)"
          f"   /  {pep.helix_length_nm:.2f} nm (alpha)")
    print(f"  copies in box         {n_copies}")
    print()
    print("CONSTRAINT A - CONCENTRATION")
    print(f"  target concentration  {concentration_from_volume(n_copies, s.conc_edge_nm**3):.3f} mM")
    print(f"  V = N/(N_A*C)         {s.conc_edge_nm**3:,.1f} nm^3")
    print(f"  cubic edge            {s.conc_edge_nm:.3f} nm")
    print()
    print("CONSTRAINT B - MINIMUM IMAGE")
    print(f"  D_max scenarios       dispersed (1 extended peptide)  {geom.dispersed_nm:6.2f} nm")
    print(f"                        globular  (10-mer droplet)      {geom.globular_nm:6.2f} nm")
    print(f"                        sheet     (10-strand ribbon)    {geom.sheet_nm:6.2f} nm  <- worst case")
    print(f"  D_max used            {s.d_max_nm:.2f} nm")
    print(f"  L >= D_max + 2*(r_cut + buffer)")
    print(f"     = {s.d_max_nm:.2f} + 2*({cutoff_nm:.2f} + {buffer_nm:.2f})"
          f"       {s.min_image_edge_nm:.3f} nm")
    print()
    print("BINDING CONSTRAINT")
    print(f"  -> {s.binding_constraint.upper()}"
          f"   (concentration needs {s.conc_edge_nm:.2f} nm,"
          f" minimum image needs {s.min_image_edge_nm:.2f} nm)")
    print()
    print("ADOPTED BOX")
    print(f"  geometry              cubic (no preferred solute axis in a")
    print(f"                        self-assembly system)")
    print(f"  edge                  {s.edge_nm:.3f} nm  ({s.edge_nm*10:.2f} A)")
    print(f"  VOLUME                {s.volume_nm3:,.1f} nm^3"
          f"   = {s.volume_nm3/1000:.3f} x 10^3 nm^3"
          f"   = {s.volume_nm3*1e-24:.3e} L")
    print(f"  peptide concentration {s.concentration_mM:.3f} mM")
    print()
    print("SYSTEM COMPOSITION")
    print(f"  water model           {water_model}  ({s.water_sites} sites/molecule)")
    print(f"  water molecules       ~{s.n_waters:,}")
    print(f"  solute net charge     {s.solute_charge:+d} e"
          f"   ({n_copies} x {pep.charge:+d})")
    print(f"  Na+                   {s.n_cation:,}")
    print(f"  Cl-                   {s.n_anion:,}"
          f"     (background {salt_mM:.0f} mM + neutralisation)")
    print(f"  TOTAL ATOMS           ~{s.n_atoms:,}")
    print()
    print("COST ESTIMATE")
    print(f"  production            {prod_ns:.0f} ns @ {timestep_fs:.0f} fs")
    print(f"  throughput ({w['gpu']})     ~{w['ns_per_day']:.1f} ns/day")
    print(f"  wall clock            ~{w['hours_for_run']:.1f} h"
          f"  ({w['hours_for_run']/24:.2f} days)")
    print("=" * 74)


def scan(pep: Peptide, n_copies: int, cutoff_nm: float, buffer_nm: float,
         salt_mM: float, water_model: str, prod_ns: float, gpu: str,
         timestep_fs: float, scenario: str) -> None:
    print()
    print("CONCENTRATION SCAN"
          f"   ({n_copies} copies, cutoff {cutoff_nm} nm, buffer {buffer_nm} nm,"
          f" D_max scenario '{scenario}')")
    print("-" * 96)
    print(f"{'C req':>7} {'C actual':>9} {'edge':>7} {'volume':>10} {'waters':>9}"
          f" {'atoms':>9} {'binds on':>14} {'ns/day':>8} {'wall h':>7}")
    print(f"{'(mM)':>7} {'(mM)':>9} {'(nm)':>7} {'(nm^3)':>10} {'':>9}"
          f" {'':>9} {'':>14} {'':>8} {'':>7}")
    print("-" * 96)
    for c in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0):
        s = size_box(pep, n_copies, c, cutoff_nm, buffer_nm, salt_mM,
                     water_model, scenario)
        w = estimate_walltime(s.n_atoms, prod_ns, gpu, timestep_fs)
        flag = "min-image*" if s.binding_constraint == "minimum image" else "concentration"
        print(f"{c:>7.1f} {s.concentration_mM:>9.2f} {s.edge_nm:>7.2f}"
              f" {s.volume_nm3:>10,.0f} {s.n_waters:>9,}"
              f" {s.n_atoms:>9,} {flag:>14} {w['ns_per_day']:>8.1f}"
              f" {w['hours_for_run']:>7.1f}")
    print("-" * 96)
    print("* the requested concentration would shrink the box past the minimum-image")
    print("  limit, so the box was enlarged and the ACTUAL concentration (column 2)")
    print("  is lower than requested.  You cannot simulate above ~"
          f"{concentration_from_volume(n_copies, min_image_edge(aggregate_geometry(pep, n_copies).sheet_nm, cutoff_nm, buffer_nm)**3):.1f} mM with")
    print(f"  {n_copies} copies without violating minimum image -- to go higher, use more copies.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config; CLI flags override it")
    p.add_argument("--sequence", help="one-letter peptide sequence")
    p.add_argument("--n-copies", type=int, default=10)
    p.add_argument("--concentration-mM", type=float, default=10.0)
    p.add_argument("--termini", default="capped", choices=sorted(TERMINI))
    p.add_argument("--cutoff-nm", type=float, default=1.0)
    p.add_argument("--buffer-nm", type=float, default=0.5)
    p.add_argument("--salt-mM", type=float, default=150.0)
    p.add_argument("--water-model", default="opc")
    p.add_argument("--aggregate-scenario", default="sheet",
                   choices=["dispersed", "globular", "sheet", "worst"])
    p.add_argument("--production-ns", type=float, default=10.0)
    p.add_argument("--gpu", default="A100")
    p.add_argument("--timestep-fs", type=float, default=4.0)
    p.add_argument("--scan", action="store_true", help="table over concentrations")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    a = p.parse_args(argv)

    if a.config:
        import yaml
        with open(a.config) as fh:
            cfg = yaml.safe_load(fh)
        sysc, boxc = cfg.get("system", {}), cfg.get("box", {})
        a.sequence = a.sequence or sysc.get("sequence")
        if "--n-copies" not in (argv or sys.argv):
            a.n_copies = sysc.get("n_copies", a.n_copies)
        a.termini = sysc.get("termini", a.termini)
        a.water_model = sysc.get("water_model", a.water_model)
        a.salt_mM = sysc.get("salt", {}).get("concentration_mM", a.salt_mM)
        a.concentration_mM = boxc.get("peptide_concentration_mM", a.concentration_mM)
        a.cutoff_nm = boxc.get("nonbonded_cutoff_nm", a.cutoff_nm)
        a.buffer_nm = boxc.get("solvent_buffer_nm", a.buffer_nm)
        a.aggregate_scenario = boxc.get("aggregate_scenario", a.aggregate_scenario)
        a.production_ns = cfg.get("production", {}).get("length_ns", a.production_ns)
        a.timestep_fs = cfg.get("production", {}).get("timestep_fs", a.timestep_fs)

    if not a.sequence:
        p.error("need --sequence or a --config that provides system.sequence")

    pep = Peptide(a.sequence, a.termini)
    s = size_box(pep, a.n_copies, a.concentration_mM, a.cutoff_nm, a.buffer_nm,
                 a.salt_mM, a.water_model, a.aggregate_scenario)

    if a.json:
        w = estimate_walltime(s.n_atoms, a.production_ns, a.gpu, a.timestep_fs)
        print(json.dumps({"peptide": {"sequence": pep.sequence,
                                      "n_residues": pep.n_residues,
                                      "mass_g_per_mol": round(pep.mass, 3),
                                      "charge": pep.charge,
                                      "n_atoms": pep.n_atoms,
                                      "volume_nm3": round(pep.volume_nm3, 3)},
                          "box": {"edge_nm": round(s.edge_nm, 4),
                                  "volume_nm3": round(s.volume_nm3, 2),
                                  "concentration_mM": round(s.concentration_mM, 4),
                                  "binding_constraint": s.binding_constraint,
                                  "d_max_nm": round(s.d_max_nm, 3),
                                  "min_image_edge_nm": round(s.min_image_edge_nm, 4),
                                  "concentration_edge_nm": round(s.conc_edge_nm, 4)},
                          "composition": {"n_waters": s.n_waters,
                                          "n_na": s.n_cation,
                                          "n_cl": s.n_anion,
                                          "n_atoms": s.n_atoms},
                          "cost": w}, indent=2))
        return 0

    report(pep, a.n_copies, s, a.production_ns, a.gpu, a.timestep_fs,
           a.cutoff_nm, a.buffer_nm, a.salt_mM, a.water_model)
    if a.scan:
        scan(pep, a.n_copies, a.cutoff_nm, a.buffer_nm, a.salt_mM, a.water_model,
             a.production_ns, a.gpu, a.timestep_fs, a.aggregate_scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
