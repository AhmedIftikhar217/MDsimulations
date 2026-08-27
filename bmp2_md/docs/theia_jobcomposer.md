# Running this on Theia via Open OnDemand's Job Composer

Job Composer is a thin web wrapper around `sbatch`. It has one behaviour that
breaks most scripts written for the command line:

> **Job Composer runs your script from its own job folder**
> (`~/ondemand/data/sys/myjobs/projects/default/<n>/`), **not from the
> repository.**

So every relative path in a normal submit script — `scripts/run_all.py`,
`config/…yaml`, and especially `#SBATCH --output=logs/…` — resolves to the
wrong place. The `--output` one is the nastiest: SLURM opens that file *before*
your script runs, so the job dies instantly with `Permission denied` or
`No such file or directory` and no other explanation.

`slurm/theia_jobcomposer.sbatch` and `slurm/theia_preflight.sbatch` are written
for this. They take an absolute `PROJECT_DIR`, `cd` into it themselves, and
write SLURM logs into the job folder, which always exists.

---

## Step 0 — get the code onto Theia (login node, once)

OnDemand → **Clusters → Theia Shell Access**:

```bash
git clone https://github.com/AhmedIftikhar217/MDsimulations.git
cd MDsimulations/bmp2_md
pwd          # <-- copy this. It is your PROJECT_DIR.
```

## Step 1 — find your cluster's names (login node, once)

```bash
bash slurm/theia_discover.sh
```

This prints your GPU partitions, your allocations, the available `cuda`/`conda`
modules, and the driver's CUDA version — then prints the exact `#SBATCH` lines
to paste. It guesses nothing; every value comes from Theia itself.

## Step 2 — build the conda environment (login node, once)

**Do this on a login node, not through Job Composer.** It takes 10–30 minutes
and needs internet, which compute nodes usually do not have.

```bash
# match cuda-version in env/environment.yml to the driver CUDA from step 1 first
conda env create -f env/environment.yml
conda activate bmp2md
python -m openmm.testInstallation
```

A login node may have no GPU, so `testInstallation` there may not list CUDA.
That is fine — step 4 tests it on a real GPU node.

## Step 3 — edit the two submit scripts

In **both** `slurm/theia_preflight.sbatch` and `slurm/theia_jobcomposer.sbatch`:

1. Paste the `#SBATCH --partition/--account/--qos` lines from step 1.
   **Delete one `#`** so they read `#SBATCH`, not `###SBATCH` — three hashes is
   a comment and SLURM ignores it silently.
2. Set `PROJECT_DIR` to the `pwd` from step 0.
3. Uncomment the `module load` lines that step 1 showed you actually have.

## Step 4 — preflight (Job Composer)

**Jobs → Job Composer → New Job → From Specified Path**

| field | value |
|---|---|
| Path | your `PROJECT_DIR` + `/slurm` |
| Script name | `theia_preflight.sbatch` |
| Account | leave blank if the script sets it |

Click **Submit**.

This runs the entire pipeline at 1/200 length — real system, real force field,
real GPU, ~50 ps instead of 10 ns — in about half an hour. It:

- refuses to continue if OpenMM cannot see the CUDA platform,
- builds the full 221k-atom system, so build errors surface here,
- **measures ns/day on the GPU you were actually given**, and tells you the
  `--time` to set for production.

Output goes to `runs_preflight/`, which can never collide with a real run.

Read the `PREFLIGHT REPORT` block at the bottom of the `.out` file. If it says
`PLUMBING OK`, continue. If it says `NOT READY`, fix the `!!` lines first.

## Step 5 — production (Job Composer)

Same procedure, script name `theia_jobcomposer.sbatch`. Set `--time` to what
the preflight recommended.

This submits an **array of 3 jobs**, one per replicate. Job Composer shows it
as a single entry; `squeue -u $USER` shows the three tasks.

Expect roughly 2 h per replicate on an A100, 4–6 h on a V100.

## Step 6 — when it finishes

```bash
cd "$PROJECT_DIR"
ls runs/bmp2_p24_10mer_*/rep*/

# the number that goes in the paper:
grep -h equilibrated_concentration_mM runs/*/rep*/equil_summary.json

# did the box land on the design volume?
grep -h within_tolerance runs/*/rep*/equil_summary.json     # want: true

# what it actually achieved
grep -h ns_per_day runs/*/rep*/prod_summary.json
```

Then read `docs/methods.md` §3 before writing anything up — it states what 10 ns
of 10 peptides does and does not support.

---

## If something goes wrong

| symptom | cause | fix |
|---|---|---|
| Job dies in <1 s, empty output | `--output` path not writable from the job folder | use the provided scripts; don't add `logs/` to `--output` |
| `PROJECT_DIR does not look like a bmp2_md checkout` | relative or wrong path | set it to the `pwd` from step 0 |
| `OpenMM cannot see the CUDA platform` | CPU partition, or `cuda-version` > driver | check `--partition`; rebuild env with a matching `cuda-version` |
| `conda environment 'bmp2md' does not exist` | step 2 not done, or done in a different account | rerun step 2 on a login node |
| Job pends forever | wrong partition/account/QOS combination | `squeue -u $USER --start`; the `NODELIST(REASON)` column names it |
| Killed at the wall limit | 10 ns needs longer than `--time` | just resubmit — it resumes from the last 100 ps checkpoint |
| `[skip] build: system.xml already present` but you wanted a fresh run | stage sentinels from an earlier run | add `--force`, or delete that `runs/…/repNN/` directory |

**Resubmitting is always safe.** Every stage is skipped if its output exists,
and production resumes from its checkpoint, so a job that was cut off picks up
where it stopped.
