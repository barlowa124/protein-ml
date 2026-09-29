"""Robustness battery: degenerate inputs, partition invariants, conformal
edge cases, and deploy-artifact consistency."""

import numpy as np
import pandas as pd
import pytest

from protstab.encode import _token_budget_batches, composition
from protstab.run import (conformal_eval, fit_ridge, metrics, mondrian_eval,
                          predict)


class TestCompositionEdges:
    def test_empty_sequence_no_nan(self):
        X = composition(pd.Series([""]))
        assert np.isfinite(X).all()
        # n = max(len,1) guards the division: empty seq reports as length 1
        assert X[0, -1] == pytest.approx(np.log1p(1))
        assert X[0, :-1].sum() == 0.0

    def test_unknown_residues_dont_crash_or_count(self):
        X = composition(pd.Series(["ACXB ZUO"]))
        # 8 chars (space counts in length); only A,C canonical -> 2/8
        assert X[0, :-1].sum() == pytest.approx(2 / 8)
        assert np.isfinite(X).all()

    def test_lowercase_counts_as_unknown(self):
        X = composition(pd.Series(["acde"]))
        assert X[0, :-1].sum() == 0.0

    def test_frequencies_bounded(self):
        rng = np.random.RandomState(0)
        seqs = pd.Series(["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWYXB"),
                                             n)) for n in [1, 5, 500]])
        X = composition(seqs)
        assert (X[:, :-1] >= 0).all() and (X[:, :-1] <= 1).all()
        assert (X[:, -1] >= 0).all()  # log length

    def test_single_residue(self):
        X = composition(pd.Series(["W"]))
        assert X[0, list("ACDEFGHIKLMNPQRSTVWY").index("W")] == 1.0


class TestBatchPartition:
    def _slices(self, lengths, bs, tb):
        return list(_token_budget_batches(lengths, bs, tb))

    def test_partition_covers_all_exactly_once(self):
        lengths = sorted(np.random.RandomState(0).randint(10, 3000, 500))
        sl = self._slices(lengths, 64, 16384)
        idx = np.concatenate([np.arange(s.start, s.stop) for s in sl])
        assert np.array_equal(np.sort(idx), np.arange(500))

    def test_oversized_sequence_gets_singleton(self):
        # a single seq longer than the token budget can't be co-batched
        sl = self._slices([10, 10, 999999, 10], 8, 1000)
        big = [s for s in sl if s.stop - s.start == 1]
        assert any(lengths_idx == 2 for s in big
                   for lengths_idx in range(s.start, s.stop))

    def test_batch_size_cap(self):
        sl = self._slices([5] * 100, 8, 10**9)
        assert all(s.stop - s.start <= 8 for s in sl)

    def test_token_budget_respected_when_possible(self):
        lengths = [10] * 50 + [20] * 50
        for s in self._slices(lengths, 128, 200):
            n = s.stop - s.start
            longest = lengths[s.stop - 1]  # sorted ascending
            assert longest * n <= 200 or n == 1

    def test_empty_input(self):
        assert self._slices([], 8, 100) == []


class TestConformalEdges:
    def _toy(self, n_cal=50, n_te=30, seed=0):
        rng = np.random.RandomState(seed)
        m = fit_ridge(rng.randn(100, 5), rng.randn(100), 1.0)
        return (m, rng.randn(n_cal, 5), rng.randn(n_cal),
                rng.randn(n_te, 5), rng.randn(n_te))

    def test_coverage_is_boolean_and_bounded(self):
        m, Xc, yc, Xt, yt = self._toy()
        q, yhat, cov = conformal_eval(m, Xc, yc, Xt, yt, 0.1)
        assert cov.dtype == bool and cov.shape == (30,)
        assert 0.0 <= cov.mean() <= 1.0
        assert q >= 0

    def test_alpha_one_gives_min_residual(self):
        m, Xc, yc, Xt, yt = self._toy()
        # alpha=1 -> 0th quantile = min |cal resid|, not 0: pin the contract
        q, _, cov = conformal_eval(m, Xc, yc, Xt, yt, 1.0)
        resid_cal = np.abs(predict(m, Xc) - yc)
        assert q == pytest.approx(resid_cal.min())
        assert cov.mean() < 0.3  # near-zero coverage, not a crash

    def test_mondrian_falls_back_on_sparse_bin(self):
        m, Xc, yc, Xt, yt = self._toy(n_cal=60)
        dist_cal = np.linalg.norm(Xc, axis=1)
        # bin 0 gets exactly 1 cal point -> below min_cal -> global q
        dist_cal = np.sort(dist_cal)
        dist_te = np.linalg.norm(Xt, axis=1)
        yhat = predict(m, Xt)
        rows, cov, edges, q_bins = mondrian_eval(
            m, Xc, yc, dist_cal, dist_te, yhat, yt, 0.1, 4, min_cal=58)
        assert all(r["fallback"] for r in rows) or len(rows) == 0
        assert cov.shape == (30,)

    def test_mondrian_edges_cover_full_range(self):
        m, Xc, yc, Xt, yt = self._toy()
        dist_cal = np.linalg.norm(Xc, axis=1)
        dist_te = np.linalg.norm(Xt, axis=1)
        yhat = predict(m, Xt)
        _, cov, edges, _ = mondrian_eval(
            m, Xc, yc, dist_cal, dist_te, yhat, yt, 0.1, 4)
        # every test point lands in exactly one bin (sentinel edges)
        binned = np.zeros(30)
        for b in range(4):
            binned += ((dist_te > edges[b]) & (dist_te <= edges[b + 1]))
        assert (binned == 1).all()

    def test_mondrian_constant_distances(self):
        m, Xc, yc, Xt, yt = self._toy()
        dist = np.ones(len(yc))
        dist_t = np.ones(len(yt))
        yhat = predict(m, Xt)
        rows, cov, edges, _ = mondrian_eval(
            m, Xc, yc, dist, dist_t, yhat, yt, 0.1, 4)
        # degenerate edges must not crash or double-count coverage
        assert np.isfinite(edges[:-1]).all()
        assert cov.dtype == bool

    def test_metrics_constant_targets(self):
        # spearman on constant y is NaN upstream; surface it, don't crash
        r = metrics(np.ones(10), np.random.RandomState(0).randn(10))
        assert np.isfinite(r["rmse"]) and np.isfinite(r["mae"])

    def test_ridge_rejects_nan_features(self):
        # sklearn input validation must reject NaN — silent propagation
        # would produce a fitted model that nan-poisons every prediction
        X = np.random.RandomState(0).randn(50, 4)
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            fit_ridge(X, np.random.RandomState(1).randn(50), 1.0)
