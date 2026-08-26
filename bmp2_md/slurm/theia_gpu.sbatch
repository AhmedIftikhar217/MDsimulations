#!/bin/bash
#SBATCH --job-name=bmp2p24
#SBATCH --output=logs/%x_rep%a_%j.out
#SBATCH --error=logs/%x_rep%a_%j.err
#SBATCH --array=1-3                 # one array task per replicate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
# ---------------------------------------------------------------------------
# >>> FILL THESE IN BEFORE FIRST USE <<<
# I do not know Theia's partition names or module stack, so these are marked
# rather than guessed.  Run these on a Theia login node and paste the answers:
#
#   sinfo -o "%20P %10G %6D %N"      # partitions and which have GPUs
#   scontrol show partition          # time limits, account requirements
#   module avail 2>&1 | grep -i -E "cuda|conda|anaconda|miniforge"
#   sacctmgr show assoc user=$USER format=account,partition,qos
#
# Then set:
###SBATCH --partition=<gpu-partition-name>
###SBATCH --account=<your-allocation>
###SBATCH --qos=<qos-if-required>
# ---------------------------------------------------------------------------
#
# One 10 ns replicate of 10 x BMP-2 P24 at 10 mM is ~221k atoms.  Expect
# roughly 2 h on an A100 and 4-6 h on a V100 at 4 fs with HMR.  The 12 h wall
# limit above is deliberately generous; production checkpoints every 100 ps and
# resumes automatically, so a job that is cut off can simply be resubmitted.

set -euo pipefail

REPLICATE="${SLURM_ARRAY_TASK_ID:-1}"
CONFIG="${CONFIG:-config/bmp2_p24_10mer.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"

echo "=========================================================="
echo " job          : ${SLURM_JOB_ID:-local}"
echo " replicate    : ${REPLICATE}"
echo " node         : $(hostname)"
echo " started      : $(date -Is)"
echo "=========================================================="

# --- environment -----------------------------------------------------------
# module load cuda/12.4                 # <-- match what `module avail` shows
# module load miniforge3                # <-- or anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bmp2md

mkdir -p logs

# --- record the hardware, so a result can be traced to the GPU it ran on ----
nvidia-smi || echo "WARNING: nvidia-smi unavailable; is a GPU actually allocated?"
python -c "
import openmm
print('OpenMM', openmm.version.version)
print('platforms:', [openmm.Platform.getPlatform(i).getName()
                     for i in range(openmm.Platform.getNumPlatforms())])
"

# --- run -------------------------------------------------------------------
# run_all.py skips any stage whose output already exists, so resubmitting a
# preempted job resumes from the last checkpoint instead of starting over.
srun python scripts/run_all.py \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --replicate "${REPLICATE}" \
    --platform CUDA

echo "finished: $(date -Is)"
