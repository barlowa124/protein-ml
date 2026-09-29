"""Shared config loader. AL_CONFIG env var overrides the default path."""

import os
from pathlib import Path

import yaml


def config_path() -> Path:
    return Path(os.environ.get("AL_CONFIG", "config/config.yaml"))


def load_config() -> dict:
    with open(config_path()) as f:
        return yaml.safe_load(f)
