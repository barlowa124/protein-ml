"""Canonical split-conformal helpers shared across the portfolio.

The conformal guarantee is the finite-sample corrected quantile:
q-hat is the ceil((n+1)(1-alpha))-th smallest calibration score, which
the formula below reads directly off the sorted array (index k-1).
Vendored identically into comp-tox-pipeline, protein-stability-uncertainty,
and cultivated-meat-multiomic — edit one copy and sync the others.
"""
from __future__ import annotations

import math

import numpy as np


def qhat(scores: np.ndarray, alpha: float) -> float:
    """Canonical split-conformal level on nonconformity scores."""
    scores = np.asarray(scores, dtype=float)
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration set")
    k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(np.sort(scores)[k - 1])


def class_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 - p(true class): nonconformity for a (n, K) probability matrix."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    return 1.0 - probs[np.arange(len(y)), y]


def class_sets(probs: np.ndarray, q: float) -> np.ndarray:
    """Boolean (n, K) set membership at threshold q."""
    return np.asarray(probs, dtype=float) >= (1.0 - q)


def abs_residuals(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float))


def interval(yhat: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric conformal interval [yhat - q, yhat + q]."""
    yhat = np.asarray(yhat, dtype=float)
    return yhat - q, yhat + q


def covered_interval(y: np.ndarray, yhat: np.ndarray, q: float) -> np.ndarray:
    lo, hi = interval(yhat, q)
    return (np.asarray(y, dtype=float) >= lo) & (y <= hi)


def covered_set(y: np.ndarray, sets: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    return sets[np.arange(len(y)), y]


def mondrian_qhat(scores: np.ndarray, groups: np.ndarray, alpha: float,
                  min_group: int = 30) -> dict:
    """Per-group conformal levels with global fallback for sparse groups.
    Returns {group: q} including a 'global' entry."""
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    out = {"global": qhat(scores, alpha)}
    for g in np.unique(groups):
        sel = scores[groups == g]
        out[g] = qhat(sel, alpha) if len(sel) >= min_group else out["global"]
    return out
