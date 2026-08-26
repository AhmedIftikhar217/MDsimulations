#!/usr/bin/env python3
"""
Run the whole pipeline for one replicate: build -> equilibrate -> produce -> analyse.

Each stage is skipped if its output already exists, so re-running after an
interruption resumes rather than restarts.  Use --force to redo a stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdlib  # noqa: E402

HERE = Path(__file__).resolve().parent

STAGES = [
    ("build",       "01_build_system.py",   "system.xml"),
    ("equilibrate", "02_equilibrate.py",    "equilibrated.xml"),
    ("production",  "03_production.py",     "prod_summary.json"),
    ("analysis",    "04_analyze.py",        "analysis.json"),
]


def _recorded_scale(path: Path) -> float | None:
    """The --scale a stage output was produced at, if it recorded one."""
    try:
        return json.loads(path.read_text()).get("scale")
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="runs",
                   help="run outputs go here; paths are relative to the "
                        "directory you launch from (intended: bmp2_md/)")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--platform", default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--force", action="store_true", help="rerun completed stages")
    p.add_argument("--only", nargs="*", choices=[s[0] for s in STAGES])
    a = p.parse_args()

    cfg = mdlib.load_config(a.config)

    # A scaled run is a plumbing test, not science.  If it wrote into the
    # normal run directory, the next full run would find those sentinels and
    # SKIP the stages -- silently reporting a 10 ns result produced by a 0.1 ns
    # run.  Divert scaled runs to their own tree unless the user has already
    # chosen a separate one.
    output_root = a.output_root
    if a.scale != 1.0 and output_root == p.get_default("output_root"):
        output_root = f"{a.output_root}_preflight"
        print(f"[note] --scale {a.scale} is a preflight, not a production run.")
        print(f"[note] redirecting output to {output_root} so it cannot be "
              f"mistaken for, or block, a real run.")

    out = mdlib.run_dir(cfg, output_root, a.replicate)

    for name, script, sentinel in STAGES:
        if a.only and name not in a.only:
            continue
        if (out / sentinel).exists() and not a.force:
            # A sentinel written at a different --scale describes a different
            # run length, so it must not satisfy this one.
            recorded = _recorded_scale(out / sentinel)
            if recorded is not None and recorded != a.scale:
                print(f"[redo] {name}: {sentinel} was produced at scale "
                      f"{recorded}, not {a.scale}; rerunning")
            else:
                print(f"[skip] {name}: {sentinel} already present")
                continue

        cmd = [sys.executable, str(HERE / script),
               "--config", str(Path(a.config).resolve()),
               "--output-root", str(Path(output_root).resolve()),
               "--replicate", str(a.replicate)]
        # Only the stages that actually run dynamics take these.
        if name in ("equilibrate", "production"):
            if a.platform:
                cmd += ["--platform", a.platform]
            if a.scale != 1.0:
                cmd += ["--scale", str(a.scale)]

        print(f"\n[run ] {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=HERE).returncode
        if rc != 0:
            print(f"[FAIL] {name} exited {rc}", file=sys.stderr)
            return rc

    print(f"\nAll stages complete for replicate {a.replicate}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
