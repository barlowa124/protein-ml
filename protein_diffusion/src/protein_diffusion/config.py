"""Shared config loader. DIFFUSION_CONFIG env var overrides the default."""

import os
from pathlib import Path

import yaml


def config_path() -> Path:
    return Path(os.environ.get("DIFFUSION_CONFIG", "config/config.yaml"))


def load_config() -> dict:
    with open(config_path()) as f:
        return yaml.safe_load(f)
