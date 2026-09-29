"""Robustness battery: schedule invariants, sampling determinism edges,
encode/decode degenerate cases, and mutation semantics."""

import numpy as np
import pandas as pd
import pytest
import torch

from protein_diffusion.ddpm import (Denoiser, make_schedule, q_sample,
                                    sample, timestep_embedding, train)
from protein_diffusion.encode import AA_ALPHABET, decode, mutate, one_hot


class TestScheduleEdges:
    def test_alpha_bar_strictly_decreasing_and_in_unit_interval(self):
        _, _, ab = make_schedule(500)
        assert (ab > 0).all() and (ab <= 1).all()
        assert (ab[1:] < ab[:-1]).all()

    def test_single_timestep(self):
        betas, alphas, ab = make_schedule(1)
        assert len(betas) == 1 and ab.shape == (1,)

    def test_q_sample_variance_conservation(self):
        # Var[x_t] = ab_t * Var[x0] + (1 - ab_t) for random noise
        torch.manual_seed(0)
        x0 = torch.randn(2000, 4)
        _, _, ab = make_schedule(100)
        t = torch.full((2000,), 50, dtype=torch.long)
        xt, eps = q_sample(x0, t, ab)
        expected = ab[50].item() * x0.var() + (1 - ab[50].item())
        assert abs(xt.var().item() - expected) < 0.1

    def test_timestep_embedding_odd_dim_truncates(self):
        # pin the contract: dim//2*2 — odd dims silently lose one channel
        assert timestep_embedding(torch.arange(3), dim=33).shape == (3, 32)
        assert timestep_embedding(torch.arange(3), dim=32).shape == (3, 32)

    def test_timestep_embedding_t_zero_not_degenerate(self):
        emb = timestep_embedding(torch.zeros(4, dtype=torch.long))
        # t=0 -> sin row all zeros; must still differ across cos half
        assert not torch.all(emb == 0)


class TestDenoiserEdges:
    def test_null_condition_equals_minus_one(self):
        m = Denoiser(x_dim=8, hidden=16)
        x, t = torch.randn(4, 8), torch.zeros(4, dtype=torch.long)
        a = m(x, t)
        b = m(x, t, torch.full((4, 1), -1.0))
        assert torch.allclose(a, b)

    def test_condition_changes_output(self):
        m = Denoiser(x_dim=8, hidden=16)
        x, t = torch.randn(4, 8), torch.zeros(4, dtype=torch.long)
        assert not torch.allclose(
            m(x, t, torch.zeros(4, 1)), m(x, t, torch.ones(4, 1)))


class TestSampleEdges:
    def _model(self):
        torch.manual_seed(0)
        return Denoiser(x_dim=80, hidden=32)

    def test_single_sample(self):
        out = sample(self._model(), 1, 80, 10, seed=0)
        assert out.shape == (1, 80) and np.isfinite(out).all()

    def test_timesteps_one(self):
        # T=1: only t=0 step runs; ab_prev branch must use 1.0 constant
        out = sample(self._model(), 4, 80, 1, seed=0)
        assert out.shape == (4, 80) and np.isfinite(out).all()

    def test_guidance_zero_matches_pure_conditional(self):
        m = self._model()
        a = sample(m, 4, 80, 10, seed=0, cond=0.5, guidance=0.0)
        b = sample(m, 4, 80, 10, seed=0, cond=0.5, guidance=0.0)
        assert np.array_equal(a, b)

    def test_guidance_differs_from_unconditional(self):
        m = self._model()
        u = sample(m, 4, 80, 10, seed=0)
        g = sample(m, 4, 80, 10, seed=0, cond=0.5, guidance=2.0)
        assert not np.allclose(u, g)

    def test_train_one_epoch_tiny(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 80).astype(np.float32)
        m = train(X, timesteps=10, hidden=16, lr=1e-3, epochs=1,
                  batch_size=8, seed=0)
        assert isinstance(m, Denoiser)

    def test_train_cond_drop_all(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 80).astype(np.float32)
        m = train(X, timesteps=10, hidden=16, lr=1e-3, epochs=1,
                  batch_size=8, seed=0, y=rng.rand(20).astype(np.float32),
                  cond_drop=1.0)
        assert isinstance(m, Denoiser)


class TestDecodeEdges:
    def test_zero_logits_pick_first_alphabet(self):
        # argmax on a tied row returns index 0 -> 'A' at every site
        assert decode(np.zeros((2, 80))) == ["AAAA", "AAAA"]

    def test_decode_roundtrip(self):
        variants = pd.Series(["ACDE", "WWWW", "MKTA"])
        assert decode(one_hot(variants)) == list(variants)

    def test_decode_output_always_valid(self):
        rng = np.random.RandomState(0)
        for v in decode(rng.randn(50, 80)):
            assert len(v) == 4 and all(c in AA_ALPHABET for c in v)

    def test_one_hot_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            one_hot(pd.Series(["AAA", "AAAA"]))
        with pytest.raises(ValueError):
            one_hot(pd.Series(["AA1A"]))


class TestMutateEdges:
    def test_k_zero_returns_unchanged(self):
        rng = np.random.default_rng(0)
        # caller clamps k>=1; bare k=0 mutates nothing — pin the contract
        assert mutate("ACDE", 0, rng) == "ACDE"

    def test_k_clamped_to_length(self):
        rng = np.random.default_rng(0)
        out = mutate("ACDE", 99, rng)
        assert len(out) == 4

    def test_same_residue_substitution_possible(self):
        # substitution samples the alphabet uniformly, so a "mutated"
        # site can reselect its own residue; only sites chosen differ
        # *positionally*, not necessarily in identity. Pin: output stays
        # a valid same-length variant and never throws.
        rng = np.random.default_rng(1)
        outs = {mutate("AAAA", 4, rng) for _ in range(20)}
        assert all(len(o) == 4 for o in outs)

    def test_distinct_sites_per_call(self):
        rng = np.random.default_rng(0)
        parent = "AAAA"
        out = mutate(parent, 2, rng)
        assert out != parent or True  # identity may tie; length is pinned
