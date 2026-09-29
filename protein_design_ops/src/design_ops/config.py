"""Shared config loading: config/config.yaml is the single source of truth.

DESIGN_CONFIG env var overrides the path so alternate configs can drive
the same DAG entry points (workflow/Snakefile honors the same variable).
"""

from __future__ import annotations

import os

import yaml

DEFAULT_CONFIG = "config/config.yaml"
ENV_VAR = "DESIGN_CONFIG"

CONFIG = os.environ.get(ENV_VAR, DEFAULT_CONFIG)


def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG) as f:
        return yaml.safe_load(f)
