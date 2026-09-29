"""Content-addressed cache for ESM-family model outputs.

One on-disk store shared by protein-design-ops, dti-fusion, and
active-learning-loop so an embedding or PLL score computed once is
reused in every repo. The same file is vendored into each repo — edit
one copy and sync the others.

Store layout: CACHE_DIR/<sha256>.npy where the key is
sha256("kind|model|sequence"). kind distinguishes outputs ("esm2-embed",
"esm2-pll"); sequence is the exact string the model saw (truncated or
context-substituted upstream — callers key on the effective input).

Env: ESM_CACHE_DIR overrides the default ~/.cache/esm2-outputs/.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(
    os.environ.get("ESM_CACHE_DIR", Path.home() / ".cache" / "esm2-outputs")
)


def _path(kind: str, model: str, seq: str) -> Path:
    key = hashlib.sha256(f"{kind}|{model}|{seq}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.npy"


def get(kind: str, model: str, seq: str) -> np.ndarray | None:
    """Cached value for this (kind, model, sequence), or None."""
    p = _path(kind, model, seq)
    if p.exists():
        return np.load(p)
    return None


def put(kind: str, model: str, seq: str, value) -> None:
    """Store a vector or scalar (float32). Atomic via tmp+rename."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(kind, model, seq)
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, np.asarray(value, dtype=np.float32))
    os.replace(tmp, p)


def get_many(kind: str, model: str, seqs) -> tuple[list, list[int]]:
    """Per-sequence cache lookup. Returns (values, miss_indices) where
    values[i] is the cached array or None and misses lists the indices
    needing computation."""
    values, misses = [], []
    for i, s in enumerate(seqs):
        v = get(kind, model, s)
        values.append(v)
        if v is None:
            misses.append(i)
    return values, misses
