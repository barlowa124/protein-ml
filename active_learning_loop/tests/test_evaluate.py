import numpy as np
import pandas as pd
import pytest

from al_loop.evaluate import aubc, plot_curves, summarize
from al_loop.surrogate import fit_predict


def _records(rows):
    return pd.DataFrame(rows, columns=[
        "strategy", "n_experiments", "best_fitness", "top_hits_found"])


class TestAubc:
    def test_perfect_curve_is_one(self):
        n = np.array([0, 25, 50])
        best = np.array([1.0, 1.0, 1.0])
        assert aubc(n, best, y_max=1.0, budget=50) == pytest.approx(1.0)

    def test_late_discovery_scores_lower(self):
        n = np.array([0, 25, 50])
        early = np.array([0.5, 1.0, 1.0])
        late = np.array([0.5, 0.5, 1.0])
        assert aubc(n, early, 1.0, 50) > aubc(n, late, 1.0, 50)


class TestSummarize:
    def test_groups_by_strategy_prefix(self):
        recs = _records([
            ["active_s0", 10, 0.6, 5], ["active_s0", 50, 0.9, 30],
            ["active_s1", 10, 0.5, 4], ["active_s1", 50, 0.8, 20],
            ["random_s0", 10, 0.2, 1], ["random_s0", 50, 0.6, 10],
            ["random_s1", 10, 0.3, 2], ["random_s1", 50, 0.5, 5],
        ])
        out = summarize(recs, y_max=1.0, top_k=100, budget=50)
        assert set(out) == {"active_s0", "active_s1",
                            "random_s0", "random_s1"}
        assert out["active_s0"]["best_fitness"] == 0.9
        assert out["random_s1"]["top_hit_rate"] == pytest.approx(0.05)
        # each key carries its own trajectory; aggregation into
        # active-vs-random means happens in main()
        assert out["active_s0"]["aubc"] > out["random_s0"]["aubc"]

    def test_final_row_read_after_sort(self):
        # unsorted input: last row must come from n_experiments order
        recs = _records([
            ["active", 50, 0.9, 30],
            ["active", 10, 0.4, 5],
        ])
        out = summarize(recs, y_max=1.0, top_k=100, budget=50)
        assert out["active"]["best_fitness"] == 0.9
        assert out["active"]["top_hits_at_budget"] == 30


class TestPlotCurves:
    def test_writes_png(self, tmp_path):
        recs = _records([
            ["active", 10, 0.5, 3], ["active", 50, 0.9, 30],
            ["random", 10, 0.3, 1], ["random", 50, 0.5, 8],
        ])
        png = tmp_path / "curves.png"
        plot_curves(recs, y_max=1.0, out_png=str(png))
        assert png.exists() and png.stat().st_size > 1000


class TestSurrogate:
    @pytest.fixture
    def toy(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 8))
        y = X[:, 0] * 2 + rng.normal(0, 0.05, 40)
        X_pool = rng.normal(size=(10, 8))
        return X, y, X_pool

    @pytest.mark.parametrize("kind", ["gp", "rf"])
    def test_shapes_and_finite(self, toy, kind):
        X, y, X_pool = toy
        mean, std = fit_predict(X, y, X_pool, kind)
        assert mean.shape == (10,) and std.shape == (10,)
        assert np.all(np.isfinite(mean))
        assert np.all(std >= 1e-8)  # floor the module promises

    def test_gp_std_lower_at_training_density(self, toy):
        X, y, X_pool = toy
        _, std_in = fit_predict(X, y, X, "gp")
        _, std_out = fit_predict(X, y, X_pool + 8.0, "gp")
        # GP confidence grows away from training data
        assert std_out.mean() > std_in.mean()

    def test_unknown_kind(self, toy):
        X, y, X_pool = toy
        with pytest.raises(ValueError, match="surrogate"):
            fit_predict(X, y, X_pool, "xgboost")
