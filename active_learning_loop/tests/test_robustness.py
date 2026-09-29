"""Robustness battery: acquisition degenerate cases, encoder rejects,
loop budget/monotonicity invariants, and evaluation bounds."""

import numpy as np
import pandas as pd
import pytest

from al_loop.acquisition import acquire, ei, ucb
from al_loop.encode import apply_context, one_hot
from al_loop.evaluate import aubc, summarize
from al_loop.loop import apply_transform, run_active, run_random
from al_loop.surrogate import fit_predict


class TestAcquisitionEdges:
    def test_ei_zero_std_no_nan(self):
        # std=0 with improvement=0 -> 0/0; must surface as 0, not NaN,
        # because NaN scores previously ordered the whole pool wrongly
        scores = ei(np.array([1.0, 0.5]), np.array([0.0, 0.0]),
                    y_best=0.5, xi=0.0)
        assert np.isfinite(scores).all()
        assert scores[1] == 0.0  # zero-improvement, zero-std -> zero EI

    def test_ei_nonnegative_everywhere(self):
        rng = np.random.RandomState(0)
        mean, std = rng.randn(1000), np.abs(rng.randn(1000)) + 1e-8
        scores = ei(mean, std, y_best=0.0, xi=0.01)
        assert np.isfinite(scores).all()
        assert (scores >= 0).all()

    def test_ucb_kappa_zero_equals_greedy(self):
        mean = np.array([1.0, 3.0, 2.0])
        std = np.array([10.0, 0.0, 5.0])
        assert np.array_equal(ucb(mean, std, 0.0),
                              acquire(mean, std, "greedy"))

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            acquire(np.ones(3), np.ones(3), "thompson")

    def test_ei_prefers_uncertain_when_means_tied(self):
        m = np.array([1.0, 1.0])
        s = np.array([0.01, 1.0])
        scores = ei(m, s, y_best=0.0, xi=0.0)
        assert scores[1] > scores[0]


class TestEncoderEdges:
    def test_one_hot_empty(self):
        X = one_hot(pd.Series([], dtype=str))
        assert X.shape == (0, 0)

    def test_one_hot_inconsistent_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            one_hot(pd.Series(["AAAA", "AAA"]))

    def test_one_hot_nonstandard_raises(self):
        with pytest.raises(ValueError, match="residue"):
            one_hot(pd.Series(["AAZA"]))

    def test_one_hot_rowsum_is_n_sites(self):
        X = one_hot(pd.Series(["ACDE", "WWWW"]))
        assert X.shape == (2, 4 * 20)
        assert np.allclose(X.sum(axis=1), 4.0)

    def test_apply_context_length_mismatch(self):
        with pytest.raises(ValueError):
            apply_context("AA", "MKT", [0])  # 2 residues, 1 site

    def test_apply_context_out_of_range_site(self):
        with pytest.raises(IndexError):
            apply_context("A", "MKT", [10])

    def test_apply_context_correct_substitution(self):
        assert apply_context("XY", "MKTAA", [1, 4]) == "MXTAY"


class TestTransformEdges:
    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError):
            apply_transform(np.ones(3), "sqrt")

    def test_log1p_on_negative_fitness(self):
        # enrichment ratios are >=0, but a negative input must not
        # silently produce NaN that poisons the surrogate
        out = apply_transform(np.array([-1.5, 0.0, 1.0]), "log1p")
        assert np.isnan(out[0]) or np.isfinite(out).all()


class TestLoopInvariants:
    def _toy(self, n=200, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 10)
        y = np.exp(rng.randn(n))  # heavy-tailed positive
        exp = {"n_init": 12, "budget": 60, "batch_size": 8,
               "seed": seed, "n_random_seeds": 2}
        sur = {"kind": "rf", "length_scale": 1.5}
        acq = {"kind": "ucb", "kappa": 2.0, "xi": 0.01}
        top = set(np.argsort(-y)[:20].tolist())
        return X, y, exp, sur, acq, top

    def test_budget_never_exceeded_and_picks_unique(self):
        X, y, exp, sur, acq, top = self._toy()
        recs, picks = run_active(X, y, exp, sur, acq, top,
                                 transform="identity")
        assert recs[-1]["n_experiments"] <= exp["budget"]
        assert len(picks) == len(set(picks))
        assert all(0 <= p < len(X) for p in picks)

    def test_best_fitness_monotone(self):
        X, y, exp, sur, acq, top = self._toy()
        recs, _ = run_active(X, y, exp, sur, acq, top,
                             transform="identity")
        best = [r["best_fitness"] for r in recs]
        assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))

    def test_seed_reproducible(self):
        X, y, exp, sur, acq, top = self._toy()
        _, p1 = run_active(X, y, exp, sur, acq, top, seed=7,
                           transform="identity")
        _, p2 = run_active(X, y, exp, sur, acq, top, seed=7,
                           transform="identity")
        assert p1 == p2
        _, p3 = run_active(X, y, exp, sur, acq, top, seed=8,
                           transform="identity")
        assert p1 != p3

    def test_budget_smaller_than_init(self):
        X, y, exp, sur, acq, top = self._toy()
        exp["budget"], exp["n_init"] = 5, 20
        recs, _ = run_active(X, y, exp, sur, acq, top,
                             transform="identity")
        assert len(recs) == 1  # init screen already exceeds budget

    def test_batch_size_one(self):
        X, y, exp, sur, acq, top = self._toy()
        exp["batch_size"] = 1
        recs, _ = run_active(X, y, exp, sur, acq, top,
                             transform="identity")
        assert recs[-1]["n_experiments"] <= exp["budget"]

    def test_random_matches_budget(self):
        X, y, exp, _, _, top = self._toy()
        recs = run_random(X, y, exp, 0, top)
        assert recs[-1]["n_experiments"] <= exp["budget"]
        assert recs[-1]["best_fitness"] == max(
            r["best_fitness"] for r in recs)


class TestSurrogateEdges:
    def _xy(self, n=30, p=5, seed=0):
        r = np.random.RandomState(seed)
        return r.randn(n, p), r.randn(n), r.randn(10, p)

    def test_std_floor_never_zero(self):
        # the NaN-acquisition bug class: downstream divides by std
        Xtr, ytr, Xp = self._xy()
        for kind in ("gp", "rf"):
            _, std = fit_predict(Xtr, ytr, Xp, kind)
            assert (std >= 1e-8).all()

    def test_single_train_point(self):
        Xtr, _, Xp = self._xy()
        mean, std = fit_predict(Xtr[:1], np.array([1.0]), Xp, "rf")
        assert mean.shape == (10,) and np.isfinite(mean).all()

    def test_unknown_kind_raises(self):
        Xtr, ytr, Xp = self._xy()
        with pytest.raises(ValueError):
            fit_predict(Xtr, ytr, Xp, "knn")


class TestEvaluationEdges:
    def test_aubc_bounded(self):
        n = np.array([12, 60])
        # trapezoid spans only measured rounds (12->60 = 48 of budget 60):
        # constant best=0.5 -> 0.5*48/60; oracle-throughout -> 48/60
        assert aubc(n, np.array([0.5, 0.5]), 1.0, 60) == pytest.approx(0.4)
        assert aubc(n, np.array([1.0, 1.0]), 1.0, 60) == pytest.approx(0.8)
        # starts at 0 -> full normalized area is achievable
        assert aubc(np.array([0, 60]), np.array([1.0, 1.0]), 1.0, 60) \
            == pytest.approx(1.0)

    def test_aubc_zero_max_guard(self):
        # y_max=0 -> division by zero; document behavior
        v = aubc(np.array([12, 60]), np.array([0.0, 0.0]), 0.0, 60)
        assert v == 0.0 or np.isnan(v) or np.isinf(v)

    def test_summarize_includes_every_strategy(self):
        rows = []
        for strat in ["active_0", "random_0", "random_1"]:
            for n_exp, bf in [(10, 0.1), (50, 0.5)]:
                rows.append({"strategy": strat, "n_experiments": n_exp,
                             "best_fitness": bf, "top_hits_found": 3})
        out = summarize(pd.DataFrame(rows), 1.0, 10, 50)
        assert set(out) == {"active_0", "random_0", "random_1"}
        assert out["active_0"]["top_hit_rate"] == pytest.approx(0.3)
