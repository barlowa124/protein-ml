"""Provenance manifest: hash the inputs + code state that produced a result."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def write_manifest(path: str, inputs: list, config_path: str = None):
    from protstab.config import config_path as _default_config
    cfg_path = config_path or str(_default_config())
    entry = {
        "python": sys.version.split()[0],
        "git_rev": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "config_sha256": hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
        if Path(cfg_path).exists()
        else None,
        "inputs": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in inputs
            if Path(p).exists()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
