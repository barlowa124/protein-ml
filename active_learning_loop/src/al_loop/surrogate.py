"""Surrogate models mapping variant encoding -> predicted fitness.

gp: GaussianProcessRegressor with RBF + white-noise kernel, normalized
    targets. The protein-AL literature standard (Wu et al. used GPs on
    these same combinatorial libraries); gives calibrated mean/std.
rf: RandomForestRegressor; uncertainty = std across trees. Cheaper and
    more robust to weird landscapes, but the std is heuristic.
"""

import numpy as np


def fit_predict(X_train, y_train, X_pool, kind: str,
                length_scale: float = 1.5):
    """Return (mean, std) surrogate predictions over the unlabeled pool."""
    if kind == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        # Fixed hyperparameters: per-round kernel optimization degenerates on
        # this spiky landscape (bounds-hitting, matmul overflow -> NaN scores).
        # length_scale ~1.5 matches the one-hot metric: variants differing at
        # one site are sqrt(2) apart, at all four sites sqrt(8). Embedding
        # encoders need their own scale, set per-config from a measured
        # distance diagnostic, not tuned on results.
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=length_scale)
            + WhiteKernel(noise_level=0.1),
            normalize_y=True,
            optimizer=None,
        )
        gp.fit(X_train, y_train)
        mean, std = gp.predict(X_pool, return_std=True)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=0
        )
        rf.fit(X_train, y_train)
        tree_preds = np.stack(
            [t.predict(X_pool) for t in rf.estimators_], axis=0
        )
        mean, std = tree_preds.mean(axis=0), tree_preds.std(axis=0)
    else:
        raise ValueError(f"unknown surrogate kind {kind!r}")
    return mean, np.maximum(std, 1e-8)
