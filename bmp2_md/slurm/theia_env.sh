# =============================================================================
# Shared job preamble.  Sourced by theia_preflight.sbatch and
# theia_jobcomposer.sbatch after they have set PROJECT_DIR.
#
# Not executable on its own -- it expects to run inside a SLURM job that has
# already set PROJECT_DIR and (optionally) CONDA_ENV.
#
# What it does, in order:
#   1. enters the project directory            (Job Composer starts elsewhere)
#   2. activates the conda environment
#   3. REFUSES TO CONTINUE if OpenMM cannot see a GPU
#
# Step 3 is the important one.  Without it a mis-specified partition sends the
# job to a CPU node, OpenMM silently falls back to the CPU platform, and you
# discover 12 hours later that you produced ~0.2 ns instead of 10 ns.
# =============================================================================

# --- 1. project directory ----------------------------------------------------
if [[ ! -f "${PROJECT_DIR}/scripts/run_all.py" ]]; then
    echo "ERROR: PROJECT_DIR does not look like a bmp2_md checkout." >&2
    echo "       PROJECT_DIR = ${PROJECT_DIR}" >&2
    echo "       expected    = \${PROJECT_DIR}/scripts/run_all.py" >&2
    echo >&2
    echo "  Job Composer runs this script from its own job folder, not from the" >&2
    echo "  repository, so PROJECT_DIR must be an ABSOLUTE path.  Edit it at the" >&2
    echo "  top of this script.  On a login node, 'cd' into your checkout and" >&2
    echo "  run 'pwd' to get the right value." >&2
    exit 2
fi
cd "${PROJECT_DIR}"
mkdir -p logs
echo "  project dir  : $(pwd)"

# --- 2. conda ----------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' is not on PATH." >&2
    echo "       Uncomment the matching 'module load' line in this script." >&2
    echo "       'bash slurm/theia_discover.sh' on a login node lists what Theia has." >&2
    exit 2
fi
# conda's activate scripts predate 'set -u' and reference unset variables.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate "${CONDA_ENV}"; then
    echo "ERROR: conda environment '${CONDA_ENV}' does not exist." >&2
    echo "       Create it ON A LOGIN NODE, once, before submitting:" >&2
    echo "           conda env create -f ${PROJECT_DIR}/env/environment.yml" >&2
    exit 2
fi
set -u
echo "  conda env    : ${CONDA_ENV}  ($(command -v python))"

# --- 3. hardware, recorded so a result can be traced to the GPU it ran on -----
echo
nvidia-smi || echo "WARNING: nvidia-smi unavailable -- is a GPU actually allocated?"
echo

# --- 4. the guard ------------------------------------------------------------
python - <<'PYCHECK'
import sys
try:
    import openmm
except ImportError as exc:
    sys.exit(f"ERROR: cannot import openmm ({exc}). Is the right env active?")

names = [openmm.Platform.getPlatform(i).getName()
         for i in range(openmm.Platform.getNumPlatforms())]
print(f"  OpenMM       : {openmm.version.version}")
print(f"  platforms    : {', '.join(names)}")

if "CUDA" not in names:
    sys.exit(
        "\nERROR: OpenMM cannot see the CUDA platform -- refusing to start.\n"
        "  This job would run on the CPU and take weeks, not hours.\n"
        "  Usual causes, in order of likelihood:\n"
        "    1. --partition points at a CPU-only partition (no --gres=gpu:1 honoured)\n"
        "    2. the conda env was solved with a cuda-version newer than the driver\n"
        "       -> check 'CUDA Version' in the nvidia-smi header above, then match\n"
        "          cuda-version in env/environment.yml and recreate the env\n"
        "    3. a 'module load cuda/...' line is still commented out\n"
    )
PYCHECK
