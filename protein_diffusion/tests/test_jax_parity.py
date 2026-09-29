import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")  # port layer is optional-exercise

from protein_diffusion.ddpm import Denoiser
from protein_diffusion.jax_model import (
    DenoiserJAX, from_torch, parity_report, sample_jax, sample_torch_det,
    timestep_embedding)
from protein_diffusion.ddpm import timestep_embedding as torch_temb


def test_timestep_embedding_matches_torch():
    t = np.array([0, 1, 299])
    a = timestep_embedding(t, 32)
    b = torch_temb(torch.tensor(t)).numpy()
    np.testing.assert_allclose(a, b, atol=1e-5)  # fp32 sin/cos ulp order


def test_weight_port_forward_parity():
    torch.manual_seed(0)
    m = Denoiser(16, 32, 32)   # toy dims
    jm, jp = from_torch(m.state_dict(), 16, 32)
    x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    t = np.array([10, 20, 30, 40])
    c = np.full((4, 1), 2.0, dtype=np.float32)
    with torch.no_grad():
        e_t = m(torch.tensor(x), torch.tensor(t), torch.tensor(c)).numpy()
    import jax.numpy as jnp
    e_j = np.asarray(jm.apply(jp, jnp.asarray(x), t, jnp.asarray(c)))
    assert np.abs(e_t - e_j).max() < 1e-5


def test_shared_noise_trajectory_parity():
    torch.manual_seed(1)
    m = Denoiser(16, 32, 32)
    jm, jp = from_torch(m.state_dict(), 16, 32)
    rng = np.random.default_rng(3)
    x_T = rng.standard_normal((8, 16)).astype(np.float32)
    noises = rng.standard_normal((20, 8, 16)).astype(np.float32)
    a = sample_torch_det(m, 8, 16, 20, x_T, noises, cond=1.5, guidance=2.0)
    b = sample_jax(jm, jp, 8, 16, 20, 0, cond=1.5, guidance=2.0,
                   noises=noises, x_T=x_T)
    assert np.abs(a - b).max() < 1e-4
