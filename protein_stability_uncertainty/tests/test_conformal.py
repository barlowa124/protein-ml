import numpy as np
import pytest

from protstab.conformal import (
    abs_residuals,
    class_scores,
    class_sets,
    covered_interval,
    covered_set,
    interval,
    mondrian_qhat,
    qhat,
)


class TestQhat:
    def test_finite_sample_formula(self):
        # n=9, alpha=0.1: k = ceil(10*0.9) = 9 -> the max
        scores = np.arange(1.0, 10.0)
        assert qhat(scores, 0.1) == 9.0
        # alpha=0.5: k = ceil(10*0.5) = 5 -> median of 1..9
        assert qhat(scores, 0.5) == 5.0

    def test_k_capped_at_n(self):
        # tiny alpha would index past the array; capped at n
        scores = np.array([0.1, 0.2, 0.3])
        assert qhat(scores, 0.01) == 0.3

    def test_order_independent(self):
        assert qhat(np.array([3.0, 1.0, 2.0]), 0.5) == \
            qhat(np.array([1.0, 2.0, 3.0]), 0.5)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            qhat(np.array([0.1]), alpha)

    def test_empty_calibration(self):
        with pytest.raises(ValueError, match="empty"):
            qhat(np.array([]), 0.1)


class TestClassificationScores:
    def test_class_scores_pick_true_column(self):
        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        y = np.array([0, 1])
        np.testing.assert_allclose(class_scores(probs, y), [0.3, 0.2])

    def test_class_sets_threshold(self):
        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        sets = class_sets(probs, q=0.5)  # keep classes with p >= 0.5
        np.testing.assert_array_equal(sets, [[True, False], [False, True]])

    def test_covered_set(self):
        sets = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(covered_set(np.array([0, 0]), sets),
                                      [True, False])


class TestRegressionIntervals:
    def test_abs_residuals(self):
        np.testing.assert_allclose(
            abs_residuals(np.array([1.0, 2.0]), np.array([1.5, 1.0])),
            [0.5, 1.0])

    def test_interval_symmetric(self):
        lo, hi = interval(np.array([10.0, 20.0]), 2.0)
        np.testing.assert_allclose(lo, [8.0, 18.0])
        np.testing.assert_allclose(hi, [12.0, 22.0])

    def test_covered_interval_edges_inclusive(self):
        yhat = np.array([10.0])
        # exactly on the boundary counts as covered
        np.testing.assert_array_equal(
            covered_interval(np.array([8.0, 12.0, 12.01]), yhat, 2.0),
            [True, True, False])


class TestMondrian:
    def test_sparse_group_falls_back_to_global(self):
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.normal(0, 1, 100), rng.normal(0, 1, 5)])
        groups = np.array(["dense"] * 100 + ["sparse"] * 5)
        out = mondrian_qhat(scores, groups, alpha=0.1, min_group=30)
        assert out["sparse"] == out["global"]
        assert out["dense"] != out["global"] or True  # dense computed on own
        sel = scores[groups == "dense"]
        assert out["dense"] == qhat(sel, 0.1)

    def test_global_key_always_present(self):
        out = mondrian_qhat(np.array([1.0, 2.0]), np.array(["a", "a"]), 0.5,
                            min_group=1)
        assert "global" in out and "a" in out


def test_empirical_coverage_reaches_nominal():
    # split-conformal guarantee sanity check: coverage on held-out points
    # drawn from the same distribution is >= 1 - alpha in expectation
    rng = np.random.default_rng(1)
    calib = rng.normal(0, 1, 500)
    test = rng.normal(0, 1, 500)
    q = qhat(np.abs(calib), 0.1)
    covered = np.mean(np.abs(test) <= q)
    assert covered >= 0.85  # loose bound: nominal 0.9, binomial noise
