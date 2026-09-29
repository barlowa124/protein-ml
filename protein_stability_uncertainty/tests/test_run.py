import numpy as np
import pandas as pd
import pytest

from protstab.run import (conformal_eval, metrics, mondrian_eval,
                          _descriptive_table)


class _ConstModel:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_conformal_coverage_exact_when_residuals_uniform():
    # cal residuals all equal to 1 -> q=1; test residuals of 0.5 covered,
    # 1.5 not
    m = _ConstModel(0.0)
    X_cal = np.zeros((10, 2))
    y_cal = np.ones(10)  # |0-1|=1 residual each
    X_te = np.zeros((4, 2))
    y_te = np.array([0.5, -0.5, 1.5, -1.5])
    q, yhat, cov = conformal_eval(m, X_cal, y_cal, X_te, y_te, alpha=0.10)
    assert q == pytest.approx(1.0)
    assert cov.tolist() == [True, True, False, False]


def test_metrics_exact():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["spearman"] == 1.0


def _edges(vals, bins):
    e = np.quantile(vals, np.linspace(0, 1, bins + 1))
    e[0], e[-1] = -1e-9, np.inf
    return e


def test_descriptive_table_bins_partition():
    dist = np.arange(12, dtype=float)
    cov = np.ones(12, dtype=bool)
    cov[-3:] = False  # far points uncovered
    rows = _descriptive_table(dist, cov, np.ones(12), _edges(dist, 4))
    assert sum(r["n"] for r in rows) == 12
    assert rows[-1]["coverage"] < rows[0]["coverage"]


def test_descriptive_table_empty_bin_skipped():
    dist = np.zeros(5)
    rows = _descriptive_table(dist, np.ones(5, dtype=bool), np.ones(5),
                              _edges(dist, 4))
    assert all(r["n"] > 0 for r in rows)


def test_mondrian_per_bin_quantiles():
    # bin 0 residuals all 1, bin 1 residuals all 10. Global q would be 10;
    # per-bin calibration tightens the near bin (q=1) without widening it.
    n = 40
    m = _ConstModel(0.0)
    X_cal = np.zeros((2 * n, 2))
    y_cal = np.r_[np.ones(n), np.full(n, 10.0)]
    dist_cal = np.r_[np.full(n, 0.5), np.full(n, 1.5)]
    # test: 2 near points (resid 0.5 and 5), 1 far point (resid 5)
    yhat = np.zeros(3)
    y_te = np.array([0.5, 5.0, 5.0])
    dist_te = np.array([0.4, 0.6, 1.4])
    rows, cov, edges, q_bins = mondrian_eval(m, X_cal, y_cal, dist_cal,
                                         dist_te, yhat, y_te, alpha=0.10,
                                         bins=2, min_cal=10)
    assert rows[0]["q"] == pytest.approx(1.0)
    assert rows[1]["q"] == pytest.approx(10.0)
    # near bin uses its own tight q: resid 0.5 covered, resid 5.0 not
    assert cov.tolist() == [True, False, True]


def test_mondrian_sparse_bin_falls_back():
    n = 40
    m = _ConstModel(0.0)
    X_cal = np.zeros((n + 3, 2))
    y_cal = np.r_[np.ones(n), np.full(3, 10.0)]  # bin1 sparse: 3 pts
    dist_cal = np.r_[np.full(n, 0.5), np.full(3, 1.5)]
    yhat = np.zeros(2)
    y_te = np.array([5.0, 5.0])
    dist_te = np.array([0.5, 1.5])
    rows, cov, _, _ = mondrian_eval(m, X_cal, y_cal, dist_cal, dist_te,
                                yhat, y_te, alpha=0.10, bins=2,
                                min_cal=10)
    from protstab.run import _conformal_quantile
    q_global = _conformal_quantile(np.abs(y_cal), 0.10)
    assert rows[-1]["fallback"] is True
    assert rows[-1]["q"] == pytest.approx(q_global)


def test_export_deployable_schema(tmp_path):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from protstab.run import export_deployable
    import json

    X = np.arange(24, dtype=float).reshape(8, 3)
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr"))
    m.fit(X, np.arange(8, dtype=float))
    out = tmp_path / "deploy.json"
    export_deployable(m, "esm2", np.zeros(3), np.array([-1e-9, 1.0, np.inf]),
                      np.array([1.0, 2.0]), 1.5, {"kind": "esm2"}, 0.10,
                      str(out))
    doc = json.loads(out.read_text())
    assert set(doc) == {"feature", "linear_model", "applicability_domain",
                        "encoder", "conformal_alpha", "note"}
    lm = doc["linear_model"]
    assert len(lm["coef"]) == 3 and len(lm["scaler_mean"]) == 3
    ad = doc["applicability_domain"]
    assert ad["bin_edges_inner"] == [1.0] and ad["bin_q"] == [1.0, 2.0]
