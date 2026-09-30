import numpy as np
import pytest

from protein_diffusion.run import (
    _agg,
    _cond_transform,
    _eval_batch,
    _stats,
)


class TestStats:
    def test_fitness_summary(self):
        fit = np.array([0.2, 0.6, 1.2, 0.9])
        st = _stats(fit, top_set={0, 2}, idx=np.array([0, 1, 2, 3]),
                    train_set=set())
        assert st["n"] == 4
        assert st["frac_ge_1"] == pytest.approx(0.25)
        assert st["frac_ge_05"] == pytest.approx(0.75)
        assert st["top_hits"] == 2


class TestEvalBatch:
    """The AGENTS rules: unmeasured variants count as zero fitness,
    not dropped rows."""

    def _landscape(self):
        variants = ["AAAA", "AAAC", "AACA"]
        fitness = [1.0, 0.5, 0.1]
        df_index = {v: i for i, v in enumerate(variants)}
        fit_map = dict(zip(variants, fitness))
        return variants, df_index, fit_map

    def test_unmeasured_counted_as_zero(self):
        variants, df_index, fit_map = self._landscape()
        generated = ["AAAA", "ZZZZ"]  # ZZZZ is not in the landscape
        st = _eval_batch(generated, df_index, fit_map, top_set={0})
        # ZZZZ contributes 0.0 to the mean, not dropped from n
        assert st["n"] == 2
        assert st["fitness_mean"] == pytest.approx(0.5)
        assert st["frac_unmeasured"] == pytest.approx(0.5)

    def test_all_measured(self):
        variants, df_index, fit_map = self._landscape()
        st = _eval_batch(["AAAA", "AAAC"], df_index, fit_map,
                         top_set={0})
        assert st["frac_unmeasured"] == 0.0
        assert st["fitness_mean"] == pytest.approx(0.75)
        assert st["top_hits"] == 1

    def test_uniqueness_reported(self):
        variants, df_index, fit_map = self._landscape()
        st = _eval_batch(["AAAA", "AAAA", "AAAC"],
                         df_index, fit_map, top_set=set())
        assert st["n"] == 3 and st["n_unique"] == 2


class TestAgg:
    def test_mean_and_std_keys(self):
        stats = [
            {"n": 4, "fitness_mean": 1.0, "top_hits": 2},
            {"n": 4, "fitness_mean": 0.0, "top_hits": 0},
        ]
        out = _agg(stats)
        assert out["n"] == 4.0
        assert out["fitness_mean"] == pytest.approx(0.5)
        assert out["fitness_mean_std"] == pytest.approx(0.5)
        assert out["top_hits_std"] == pytest.approx(1.0)
        # std of n is meaningless and not emitted
        assert "n_std" not in out


class TestCondTransform:
    def test_log1p_identity(self):
        y = np.array([0.0, 1.0])
        yc, to_cond = _cond_transform(y, "log1p")
        np.testing.assert_allclose(yc, np.log1p(y))
        # conditioning channel stays >= 0 so -1 remains a valid null token
        assert yc.min() >= 0
        assert to_cond(2.0) == pytest.approx(np.log1p(2.0))

    def test_shift_log1p_handles_negative_fitness(self):
        # AAV log-viability goes negative; the shift keeps the
        # conditioning channel above the -1 null token
        y = np.array([-11.0, -5.0, -2.0])
        yc, to_cond = _cond_transform(y, "shift_log1p")
        assert np.isfinite(yc).all()
        assert yc.min() >= 0
        # to_cond applies the same shift to a new conditioning value
        assert to_cond(-11.0) == pytest.approx(yc[0])

    def test_unknown_transform(self):
        with pytest.raises(ValueError, match="transform"):
            _cond_transform(np.array([1.0]), "zscore")
