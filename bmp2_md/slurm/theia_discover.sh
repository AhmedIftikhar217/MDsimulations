#!/bin/bash
# =============================================================================
# Run this ONCE on a Theia LOGIN node (not through Job Composer).
#
#     bash slurm/theia_discover.sh
#
# It answers the four questions the submit scripts cannot answer for you, and
# prints the exact #SBATCH lines to paste into them.  Nothing here is guessed:
# every value comes from your own cluster.
# =============================================================================
set -uo pipefail

hr() { printf '%.0s-' {1..74}; echo; }
sec() { echo; hr; echo "  $1"; hr; }

sec "1. GPU PARTITIONS  (pick one for --partition)"
# Gres column shows which partitions actually have GPUs attached.
sinfo -o "%-24P %-18G %-10l %-8D %N" 2>/dev/null \
  | awk 'NR==1 || tolower($2) ~ /gpu/' \
  || echo "  sinfo unavailable -- ask Theia support for the GPU partition name"
echo
echo "  (column 2 = Gres; a partition with no gpu:N there has NO GPUs)"
echo "  full list, including time limits:  scontrol show partition"

sec "2. YOUR ALLOCATIONS  (pick one for --account, and --qos if listed)"
sacctmgr -nP show assoc user="${USER}" format=Account,Partition,QOS 2>/dev/null \
  | column -t -s'|' \
  || echo "  sacctmgr unavailable -- your cluster may not require --account"

sec "3. MODULES  (uncomment the matching 'module load' lines)"
module avail 2>&1 | grep -i -E "cuda|conda|anaconda|miniforge|mamba" \
  || echo "  no cuda/conda modules found -- you may already have conda on PATH"
echo
echo "  conda on PATH now? $(command -v conda || echo 'NO -- module load one above')"

sec "4. DRIVER CUDA VERSION  (must be >= cuda-version in env/environment.yml)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -4
    echo "  NOTE: login nodes often have no GPU; the number that matters is the"
    echo "        'CUDA Version:' in the header, which is the DRIVER's."
else
    echo "  nvidia-smi not on the login node -- run this inside a GPU job instead:"
    echo "      srun --partition=<gpu-partition> --gres=gpu:1 --pty nvidia-smi"
fi

sec "5. PASTE THESE INTO slurm/theia_jobcomposer.sbatch"
cat <<'TEMPLATE'
  #SBATCH --partition=<from section 1>
  #SBATCH --account=<from section 2>
  #SBATCH --qos=<from section 2, only if your cluster requires it>

  and set, further down in the same file:

  PROJECT_DIR="<absolute path to your bmp2_md checkout>"
TEMPLATE
echo
echo "  this checkout is at: $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hr
