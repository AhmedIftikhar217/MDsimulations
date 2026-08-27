# BMP-2 peptide MD pipeline

Reproducible OpenMM/Amber molecular dynamics for **10 copies of the BMP-2
knuckle-epitope peptide P24** in explicit solvent.

Everything is driven by one config file. Change a number there and the box
volume, the system composition, the protocol and the Methods numbers all follow.

```
config/bmp2_p24_10mer.yaml     <- single source of truth
scripts/00_box_size.py         <- derives the box from the concentration
scripts/01_build_system.py     <- peptide -> caps -> 10 copies -> solvate -> ions -> System
scripts/02_equilibrate.py      <- minimise, NVT heat, NPT with staged restraint release
scripts/03_production.py       <- 10 ns NPT, checkpointed and restartable
scripts/04_analyze.py          <- QC, structure, association, minimum-image check
scripts/run_all.py             <- runs all four, skipping completed stages
slurm/theia_discover.sh        <- run on a login node: prints your SBATCH lines
slurm/theia_preflight.sbatch   <- 1/200-length end-to-end check on a real GPU
slurm/theia_jobcomposer.sbatch <- production array, safe under Open OnDemand
slurm/theia_gpu.sbatch         <- same, for plain `sbatch` from bmp2_md/
env/environment.yml            <- conda environment
docs/methods.md                <- paper-ready Methods + derivations + caveats
```

## The headline numbers

For **10 peptides at 10 mM**:

| quantity | value |
|---|---|
| **box volume** | **1.6605 × 10³ nm³** (1,660.5 nm³) |
| **cubic edge** | **11.8418 nm** |
| binding constraint | concentration (needs 11.84 nm) over minimum image (needs 11.24 nm) |
| water molecules | ~54,400 (OPC) |
| ions | ~148 Na⁺ / ~158 Cl⁻ (150 mM + neutralising) |
| total atoms | ~221,000 |
| cost | ~1.7 h per 10 ns replicate on an A100 |

Regenerate, or explore other concentrations:

```bash
python scripts/00_box_size.py --config config/bmp2_p24_10mer.yaml
python scripts/00_box_size.py --config config/bmp2_p24_10mer.yaml --scan
```

`--scan` prints the concentration/box/cost trade-off, including the point where
minimum image — not concentration — starts to set the box.

## Install

```bash
conda env create -f env/environment.yml
conda activate bmp2md
python -m openmm.testInstallation      # MUST list CUDA before you submit a real job
```

## Run

One replicate, locally:

```bash
python scripts/run_all.py --config config/bmp2_p24_10mer.yaml --replicate 1 --platform CUDA
```

Quick end-to-end smoke test (seconds, physically meaningless, proves the
plumbing works):

```bash
python scripts/run_all.py --config config/bmp2_p24_10mer.yaml \
    --replicate 1 --platform CPU --scale 0.001
```

Three replicates on Theia, from a login node:

```bash
bash slurm/theia_discover.sh     # prints the partition/account lines to paste
sbatch slurm/theia_preflight.sbatch   # ~30 min: proves it works, benchmarks the GPU
sbatch slurm/theia_gpu.sbatch         # only after the preflight passes
```

**Using OnDemand's Job Composer instead?** Submit
`slurm/theia_preflight.sbatch` and then `slurm/theia_jobcomposer.sbatch`, and
follow [`docs/theia_jobcomposer.md`](docs/theia_jobcomposer.md). Job Composer
runs scripts from its own folder rather than from the repository, which breaks
relative paths -- those two scripts are written for it, `theia_gpu.sbatch` is
not.

## Outputs, per replicate

| file | contents |
|---|---|
| `system.pdb`, `system.xml` | solvated coordinates and the serialised System — the complete force-field assignment |
| `build_summary.json` | box, composition, achieved concentration, seeds |
| `equilibrated.xml` | full State after equilibration |
| `equil_summary.json` | **equilibrated volume vs design volume** — confirms the concentration you report |
| `traj.dcd`, `prod_log.csv` | trajectory and scalar log |
| `analysis.json`, `timeseries.csv` | all analysis numbers |
| `analysis.png`, `rdf.png` | figures |
| `provenance_*.json` | versions, git commit, hardware, SLURM env, seeds |

## Two things worth knowing

**The box is derived, not chosen.** It comes from `V = N/(N_A·C)` and is then
checked against the minimum-image requirement using a worst-case aggregate size.
Both constraints are reported, and the script says which one binds. See
`docs/methods.md` §2.1.

**The build compensates for solvation under-fill.** OpenMM's `addSolvent` packs
water ~5% below bulk density (measured here: 961 vs 993 kg/m³). Left alone, NPT
compresses the box and a 10.0 mM design silently becomes ~10.5 mM. The builder
solvates once to measure the deficit, then rebuilds in a box enlarged by exactly
the right amount, and equilibration verifies the result. Set
`box.precompensate_solvent_density: false` to disable.

## Scope

Ten nanoseconds is a **pilot**, not an aggregation study. It supports
conformational relaxation and early encounter events; it does not support
equilibrium aggregation propensity, β-sheet formation, or any free energy of
association. `docs/methods.md` §3 spells out what to claim and what not to.
Extending is a one-line config change (`production.length_ns`).

## Before you run production

- [x] **Peptide sequence verified.** `KIPKASSVPTELSAISTLYL` is residues 73-92 of
      mature human BMP-2 (UniProt P12643), the knuckle epitope. Its net charge of
      +1 per capped copy matches the +10 e that `00_box_size.py` derives for 10
      copies. Still worth a glance against your source paper.
- [ ] Create the conda env on a Theia **login** node (`conda env create -f env/environment.yml`).
      Compute nodes generally have no internet.
- [ ] Fill in the SLURM partition/account placeholders -- `slurm/theia_discover.sh` prints them.
- [ ] Run `slurm/theia_preflight.sbatch` and read its `PREFLIGHT REPORT`. It
      confirms CUDA on a real GPU node and measures the ns/day your `--time` needs.
- [ ] Decide whether 310 K and 150 mM NaCl match your experimental conditions.
