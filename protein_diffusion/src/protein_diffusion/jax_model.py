"""JAX/Flax port of the DDPM denoiser with checkpoint-level parity.

`DenoiserJAX` mirrors `ddpm.Denoiser` layer-for-layer; `from_torch` maps
the committed torch state_dict into a Flax param tree. The parity claim
is numerical, not narrative: identical (x, t, c) inputs produce epsilon
predictions matching torch to fp32 reduction order, and a shared noise
schedule drives both samplers to the same x0.
"""

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn


def timestep_embedding(t: np.ndarray, dim: int = 32) -> np.ndarray:
    """Matches ddpm.timestep_embedding (sin/cos, half-dim each)."""
    half = dim // 2
    freq = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t.astype(np.float32)[:, None] * freq[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1)


class DenoiserJAX(nn.Module):
    x_dim: int
    hidden: int
    t_dim: int = 32

    @nn.compact
    def __call__(self, x, t, c=None):
        emb = jnp.asarray(timestep_embedding(np.asarray(t), self.t_dim))
        if c is None:
            c = jnp.full((x.shape[0], 1), -1.0)
        h = jnp.concatenate([x, emb, c], axis=-1)
        h = nn.silu(nn.Dense(self.hidden)(h))
        h = nn.silu(nn.Dense(self.hidden)(h))
        return nn.Dense(self.x_dim)(h)


def from_torch(state_dict: dict, x_dim: int, hidden: int):
    """Map torch `net.{0,2,4}.{weight,bias}` into a Flax param pytree."""
    model = DenoiserJAX(x_dim=x_dim, hidden=hidden)
    layers = {}
    for tk, fl in ((0, "Dense_0"), (2, "Dense_1"), (4, "Dense_2")):
        w = state_dict[f"net.{tk}.weight"].numpy()  # (out, in)
        b = state_dict[f"net.{tk}.bias"].numpy()
        layers[fl] = {"kernel": jnp.asarray(w.T), "bias": jnp.asarray(b)}
    return model, {"params": layers}


def _denoise(params, model, x, t_arr, c_arr):
    return model.apply(params, x, t_arr, c_arr)


def sample_jax(model, params, n: int, x_dim: int, timesteps: int,
               seed: int, cond: float = None, guidance: float = 0.0,
               noises: np.ndarray = None, x_T: np.ndarray = None):
    """Ancestral DDPM sampling, matching ddpm.sample step-for-step.

    `noises`: optional (timesteps, n, x_dim) array of per-step draws —
    when supplied, the trajectory is fully determined and comparable
    across frameworks. Otherwise jax.random fills them from `seed`.
    """
    betas = np.linspace(1e-4, 0.02, timesteps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    rng = jax.random.PRNGKey(seed)
    if noises is None:
        noises = np.asarray(
            jax.random.normal(rng, (timesteps, n, x_dim)), dtype=np.float32)
    x = (np.asarray(x_T, dtype=np.float32) if x_T is not None
         else np.asarray(jax.random.normal(rng, (n, x_dim)),
                         dtype=np.float32))

    for t in range(timesteps - 1, -1, -1):
        tt = np.full((n,), t, dtype=np.int64)
        if cond is None:
            eps = _denoise(params, model, jnp.asarray(x), tt, None)
        else:
            c = jnp.full((n, 1), float(cond))
            eps_c = _denoise(params, model, jnp.asarray(x), tt, c)
            if guidance:
                eps_u = _denoise(params, model, jnp.asarray(x), tt, None)
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = eps_c
        eps = np.asarray(eps)
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else 1.0
        coef = (1 - alphas[t]) / np.sqrt(1 - ab_t)
        mean = (x - coef * eps) / np.sqrt(alphas[t])
        if t > 0:
            sigma = np.sqrt((1 - ab_prev) / (1 - ab_t) * betas[t])
            x = mean + sigma * noises[t]
        else:
            x = mean
    return x


def sample_torch_det(model, n: int, x_dim: int, timesteps: int,
                     x_T: np.ndarray, noises: np.ndarray,
                     cond: float = None, guidance: float = 0.0):
    """Torch sampler driven by pre-drawn noise, for cross-framework parity.

    Replicates ddpm.sample's update equations with `noises[t]` injected
    instead of an internal RNG draw, and `x_T` as the shared start point.
    """
    import torch
    from protein_diffusion.ddpm import make_schedule

    betas, alphas, alpha_bar = make_schedule(timesteps)
    x = torch.tensor(x_T, dtype=torch.float32)
    with torch.no_grad():
        for t in range(timesteps - 1, -1, -1):
            tt = torch.full((n,), t, dtype=torch.long)
            if cond is None:
                eps = model(x, tt)
            else:
                c = torch.full((n, 1), float(cond))
                eps_c = model(x, tt, c)
                if guidance:
                    eps_u = model(x, tt)
                    eps = eps_u + guidance * (eps_c - eps_u)
                else:
                    eps = eps_c
            ab_t = alpha_bar[t]
            ab_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
            coef = (1 - alphas[t]) / torch.sqrt(1 - ab_t)
            mean = (x - coef * eps) / torch.sqrt(alphas[t])
            if t > 0:
                sigma = torch.sqrt(
                    (1 - ab_prev) / (1 - ab_t) * betas[t])
                x = mean + sigma * torch.tensor(noises[t])
            else:
                x = mean
    return x.numpy()


def parity_report(model_path: str, n: int = 64, timesteps: int = 30,
                  seed: int = 0):
    """Shared-noise trajectory comparison: torch vs JAX on one checkpoint.

    Returns max abs deviation of x0 after a full ancestral pass, plus the
    max per-call epsilon deviation. Timesteps default small; the parity
    argument is per-step, not scale-dependent.
    """
    import torch
    from protein_diffusion.ddpm import Denoiser

    sd = torch.load(model_path, weights_only=True)
    w0 = sd["net.0.weight"]
    hidden, x_in = w0.shape
    x_dim, t_dim = x_in - 32 - 1, 32
    torch_model = Denoiser(x_dim, hidden, t_dim)
    torch_model.load_state_dict(sd)
    torch_model.eval()
    jmodel, jparams = from_torch(sd, x_dim, hidden)

    rng = np.random.default_rng(seed)
    x_T = rng.standard_normal((n, x_dim)).astype(np.float32)
    noises = rng.standard_normal((timesteps, n, x_dim)).astype(np.float32)

    # per-call epsilon parity on random mid-trajectory inputs
    x_mid = rng.standard_normal((n, x_dim)).astype(np.float32)
    t_mid = np.full((n,), timesteps // 2, dtype=np.int64)
    with torch.no_grad():
        e_torch = torch_model(
            torch.tensor(x_mid), torch.tensor(t_mid)).numpy()
    e_jax = np.asarray(jmodel.apply(
        jparams, jnp.asarray(x_mid), t_mid, None))
    eps_dev = float(np.abs(e_torch - e_jax).max())

    xt = sample_torch_det(torch_model, n, x_dim, timesteps, x_T, noises)
    xj = sample_jax(jmodel, jparams, n, x_dim, timesteps, seed,
                    noises=noises, x_T=x_T)
    x0_dev = float(np.abs(xt - xj).max())
    return {"eps_max_abs_dev": eps_dev, "x0_max_abs_dev": x0_dev,
            "n": n, "timesteps": timesteps}
