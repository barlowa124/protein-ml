"""Parity between _conformal_quantile and the vendored canonical
module (src/protstab/conformal.py).

The repo level is (1-a)(n+1)/n at np.quantile(method='higher') —
the canonical index or one order statistic higher, never lower.
"""
import numpy as np

from protstab import conformal
from protstab.run import _conformal_quantile


def test_repo_quantile_never_below_canonical():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(5, 400))
        s = rng.exponential(1.0, n)
        for alpha in (0.05, 0.1, 0.2):
            assert _conformal_quantile(s, alpha) >= conformal.qhat(s, alpha) - 1e-12


def test_mondrian_falls_back_for_sparse_groups():
    scores = np.arange(100, dtype=float)
    groups = np.zeros(100, dtype=int)
    groups[:5] = 1  # sparse group: n=5 < min_group
    qs = conformal.mondrian_qhat(scores, groups, 0.1, min_group=30)
    assert qs[1] == qs["global"]
    assert qs[0] >= qs["global"]


def test_interval_coverage():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.1, 2.0, 4.0])
    cov = conformal.covered_interval(y, yhat, q=0.5)
    assert list(cov) == [True, True, False]
