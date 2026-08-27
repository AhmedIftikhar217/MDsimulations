#!/bin/bash
# =============================================================================
# PRODUCTION -- 3 replicates x 10 ns of 10 x BMP-2 P24 at 10 mM.
#
# Written for Open OnDemand's Job Composer.  The difference that matters:
# Job Composer submits from its OWN job folder
# (~/ondemand/data/sys/myjobs/projects/default/<n>/), not from the repository.
# So this script uses an absolute PROJECT_DIR and writes its SLURM logs into
# the job folder, where they are guaranteed to be writable.
#
# Do not submit this until slurm/theia_preflight.sbatch has passed.
#
# See docs/theia_jobcomposer.md for the click-by-click walkthrough.
# =============================================================================
#SBATCH --job-name=bmp2p24
#SBATCH --output=bmp2p24_rep%a_%A.out
#SBATCH --error=bmp2p24_rep%a_%A.err
#SBATCH --array=1-3                 # one array task per replicate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00             # raise if the preflight projects more
#SBATCH --mem=32G
#
# >>> FILL IN BEFORE FIRST USE <<<
# Run 'bash slurm/theia_discover.sh' on a Theia login node; it prints these.
# Delete one '#' so the line reads '#SBATCH', not '###SBATCH'.
###SBATCH --partition=<gpu-partition-name>
###SBATCH --account=<your-allocation>
###SBATCH --qos=<qos-if-required>

set -euo pipefail

# --- the one path you must set ----------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-${HOME}/MDsimulations/bmp2_md}"
CONDA_ENV="${CONDA_ENV:-bmp2md}"

# module load cuda/12.4          # <-- uncomment/edit to match 'module avail'
# module load miniforge3         # <-- or anaconda3

REPLICATE="${SLURM_ARRAY_TASK_ID:-1}"
CONFIG="${CONFIG:-config/bmp2_p24_10mer.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"

echo "=========================================================="
echo " job          : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
echo " replicate    : ${REPLICATE} of ${SLURM_ARRAY_TASK_COUNT:-1}"
echo " node         : $(hostname)"
echo " partition    : ${SLURM_JOB_PARTITION:-<none>}"
echo " started      : $(date -Is)"
echo "=========================================================="

source "${PROJECT_DIR}/slurm/theia_env.sh"

# --- run ---------------------------------------------------------------------
# run_all.py skips any stage whose output already exists, so a job killed by
# the wall limit resumes from its last 100 ps checkpoint on resubmission
# instead of starting over.  Resubmitting is always safe.
#
# No 'srun' here: with a single task it adds nothing and, on clusters that bind
# GPUs per-step, it is a way to lose the GPU the job was allocated.
python scripts/run_all.py \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --replicate "${REPLICATE}" \
    --platform CUDA

echo
echo "finished: $(date -Is)"
echo "results in: ${PROJECT_DIR}/${OUTPUT_ROOT}/"
