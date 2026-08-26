"""
Shared helpers: config loading, deterministic seeding, and provenance capture.

The point of this module is that a result can always be traced back to exactly
what produced it -- config, code version, software versions, hardware and seeds.
Anything that a reviewer could ask "what settings gave you that?" about is
written next to the output.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(Path(path).resolve())
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable SHA-256 over the physical content of the config.

    Keys beginning with '_' are bookkeeping, not physics, and are excluded so
    that the hash identifies the SIMULATION, not the file path it came from.
    """
    physical = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(physical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Deterministic seeding
# ----------------------------------------------------------------------------

def derive_seeds(master_seed: int, replicate: int) -> dict[str, int]:
    """Derive every seed a run needs from (master_seed, replicate).

    Deterministic and independent: replicate 3 is always the same run, and no
    two replicates share a stream.  Using a hash rather than master+replicate
    avoids the correlated low-order bits you get from adjacent integer seeds.
    """
    def sub(tag: str) -> int:
        h = hashlib.sha256(f"{master_seed}:{replicate}:{tag}".encode()).digest()
        # OpenMM seeds are 32-bit signed and 0 means "pick randomly", so keep
        # the value in [1, 2^31-1].
        return int.from_bytes(h[:4], "big") % (2**31 - 2) + 1

    return {
        "packing": sub("packing"),
        "solvent": sub("solvent"),
        "velocities": sub("velocities"),
        "integrator": sub("integrator"),
        "barostat": sub("barostat"),
    }


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------

def _git_info(repo_root: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(args, cwd=repo_root, capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the code on disk is NOT the code at that commit.
        # Recording it stops you from trusting a commit hash that lies.
        "dirty": bool(dirty) if dirty is not None else None,
    }


def _package_versions() -> dict:
    versions: dict[str, str] = {}
    for mod in ("openmm", "numpy", "scipy", "mdtraj", "MDAnalysis",
                "Bio", "PeptideBuilder", "yaml", "pandas", "matplotlib"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not installed"
    return versions


def _openmm_platforms() -> dict:
    try:
        import openmm
        return {
            "openmm_version": openmm.version.version,
            "available_platforms": [
                openmm.Platform.getPlatform(i).getName()
                for i in range(openmm.Platform.getNumPlatforms())
            ],
        }
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def provenance(cfg: dict, stage: str, extra: dict[str, Any] | None = None) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    rec = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg.get("_config_path"),
        "config_hash": cfg.get("_config_hash"),
        "config_version": cfg.get("metadata", {}).get("version"),
        "git": _git_info(repo_root),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "packages": _package_versions(),
        "openmm": _openmm_platforms(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")} or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if extra:
        rec.update(extra)
    return rec


def write_provenance(cfg: dict, stage: str, out_dir: Path,
                     extra: dict[str, Any] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"provenance_{stage}.json"
    with open(path, "w") as fh:
        json.dump(provenance(cfg, stage, extra), fh, indent=2)
    return path


def run_dir(cfg: dict, root: str | Path, replicate: int) -> Path:
    """Output directory for one replicate.  Encodes the config hash so two
    different configs can never write into the same place."""
    name = cfg.get("metadata", {}).get("name", "run")
    d = Path(root) / f"{name}_{cfg['_config_hash']}" / f"rep{replicate:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def banner(title: str) -> None:
    print("\n" + "=" * 74, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 74, flush=True)
