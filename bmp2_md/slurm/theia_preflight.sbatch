#!/bin/bash
# =============================================================================
# PREFLIGHT -- run this BEFORE the real thing.  Submit it from Job Composer
# exactly like the production script.
#
# It runs the complete pipeline end to end at 1/200th length: real system,
# real force field, real GPU, ~50 ps of dynamics instead of 10 ns.  Everything
# that can go wrong in a 12-hour production run goes wrong here in ~30 minutes
# instead, and it costs you almost no allocation.
#
# It also measures ns/day on the GPU you were actually given, which turns the
# "~1.7 h on an A100" estimate in the README into a number for YOUR hardware.
#
# Output goes to  runs_preflight/  -- run_all.py diverts it there automatically
# so it can never be mistaken for, or block, a production run.
# =============================================================================
#SBATCH --job-name=bmp2_preflight
#SBATCH --output=bmp2_preflight_%j.out
#SBATCH --error=bmp2_preflight_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00             # build solvates twice (density pre-compensation)
#SBATCH --mem=32G
#
# >>> FILL IN, same values as the production script <<<
# Run 'bash slurm/theia_discover.sh' on a Theia login node to get them.
###SBATCH --partition=<gpu-partition-name>
###SBATCH --account=<your-allocation>
###SBATCH --qos=<qos-if-required>

set -euo pipefail

# --- the one path you must set ----------------------------------------------
# ABSOLUTE path to your bmp2_md checkout.  Job Composer runs this script from
# ~/ondemand/data/sys/myjobs/projects/default/<n>/, NOT from the repository,
# so a relative path here will not find anything.
PROJECT_DIR="${PROJECT_DIR:-${HOME}/MDsimulations/bmp2_md}"
CONDA_ENV="${CONDA_ENV:-bmp2md}"

# module load cuda/12.4          # <-- uncomment/edit to match 'module avail'
# module load miniforge3         # <-- or anaconda3

echo "=========================================================="
echo " PREFLIGHT (not a production run)"
echo " job          : ${SLURM_JOB_ID:-local}"
echo " node         : $(hostname)"
echo " partition    : ${SLURM_JOB_PARTITION:-<none>}"
echo " started      : $(date -Is)"
echo "=========================================================="

source "${PROJECT_DIR}/slurm/theia_env.sh"

# --- the run -----------------------------------------------------------------
# 0.005 = 10 ps equilibration + 50 ps production.  Physically meaningless,
# but it exercises every line of code the production run will execute.
SCALE=0.005

python scripts/00_box_size.py --config config/bmp2_p24_10mer.yaml

python scripts/run_all.py \
    --config config/bmp2_p24_10mer.yaml \
    --replicate 1 \
    --platform CUDA \
    --scale "${SCALE}"

# --- the verdict -------------------------------------------------------------
python - "${SCALE}" <<'PYREPORT'
import json, sys
from pathlib import Path

scale = float(sys.argv[1])
runs = sorted(Path("runs_preflight").glob("*/rep01"))
if not runs:
    sys.exit("ERROR: no preflight output found under runs_preflight/")
out = runs[-1]

prod = json.loads((out / "prod_summary.json").read_text())
equil = json.loads((out / "equil_summary.json").read_text())
build = json.loads((out / "build_summary.json").read_text())

nspd = prod.get("ns_per_day")
line = "=" * 74
print(f"\n{line}\n  PREFLIGHT REPORT\n{line}")
print(f"  length scale        : {scale}  ({scale * 10_000:.0f} ps production)")
print(f"  platform            : {prod.get('platform')}")
print(f"  total atoms         : {build['composition']['total_atoms']:,}")
print(f"  measured throughput : {nspd} ns/day")

ok = True
if prod.get("platform") != "CUDA":
    print("  !! ran on the CPU platform -- do NOT submit production")
    ok = False

if nspd:
    hours = 10.0 / nspd * 24.0
    print(f"\n  PROJECTED 10 ns PRODUCTION: {hours:.1f} h per replicate")
    # The preflight is short, so its ns/day is pessimistic (startup dominates).
    # Even so, a wall limit under ~1.5x the projection is asking to be killed.
    print(f"  -> set --time to at least {max(2, int(hours * 1.5) + 1):02d}:00:00 "
          f"in theia_jobcomposer.sbatch")
    if hours > 12:
        print("  !! that exceeds the 12:00:00 default wall limit.")
        print("     Production checkpoints every 100 ps and resumes on")
        print("     resubmission, so this is survivable -- but raise --time")
        print("     if the partition allows it.")

print(f"\n  design volume       : {equil['design_volume_nm3']:,.1f} nm^3")
print(f"  equilibrated volume : {equil['equilibrated_volume_nm3']:,.1f} nm^3"
      f"  ({equil['deviation_percent']:+.2f} %)")
print(f"  -> concentration    : {equil['equilibrated_concentration_mM']:.3f} mM"
      f"  (design {build['box']['design_concentration_mM']:.3f} mM)")
print("  (50 ps of NPT is far too short for the density to converge, so a")
print("   large deviation here is EXPECTED and is not a failure. The real")
print("   check is equil_summary.json['within_tolerance'] in production.)")

print(f"\n{line}")
print("  PLUMBING OK -- ready for production." if ok else
      "  NOT READY -- fix the items marked !! above first.")
print(f"{line}")
sys.exit(0 if ok else 1)
PYREPORT

echo "finished: $(date -Is)"
