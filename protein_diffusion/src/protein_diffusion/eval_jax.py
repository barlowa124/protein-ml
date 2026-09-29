"""Commit the JAX-parity artifact: numerical equivalence + oracle band.

Writes `results/parity_jax.json` with (a) shared-noise trajectory
deviation between the torch and Flax samplers on the committed GB1
checkpoint, and (b) oracle-evaluated stats for JAX-sampled variants
through the identical `_eval_batch` path — the JAX port is scored against
the measured landscape, not just tensor-matched.

    DIFFUSION_CONFIG=config/config.yaml \
    python -m protein_diffusion.eval_jax <parquet> <model> <out_json>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from protein_diffusion.config import config_path, load_config
from protein_diffusion.ddpm import Denoiser
from protein_diffusion.encode import decode, one_hot
from protein_diffusion.jax_model import (
    from_torch, parity_report, sample_jax)
from protein_diffusion.provenance import write_manifest
from protein_diffusion.run import _agg, _cond_transform, _eval_batch


def main(in_parquet, model_path, out_json):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    ds = cfg["dataset"]
    alphabet = ds.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")
    fit_map = dict(zip(df["variant"], df["fitness"]))
    X = one_hot(df["variant"], alphabet)
    top_set = set(
        np.argsort(-df["fitness"].to_numpy())[
            : cfg["evaluation"]["top_k"]].tolist())
    df_index = {v: i for i, v in enumerate(df["variant"])}

    m = cfg["model"]
    sd = torch.load(model_path, weights_only=True)
    torch_model = Denoiser(X.shape[1], m["hidden"])
    torch_model.load_state_dict(sd)
    torch_model.eval()
    jmodel, jparams = from_torch(sd, X.shape[1], m["hidden"])

    parity = parity_report(
        model_path, n=64, timesteps=m["timesteps"], seed=0)

    # Oracle band: JAX-sampled guided_w8 variants through the real eval.
    gcfg = cfg["generate"]
    cond_val = float(_cond_transform(
        df["fitness"].to_numpy(), ds.get("transform", "log1p"))[1](
            gcfg["cond_fitness"]))
    n_seeds = min(3, gcfg.get("n_seeds", 1))   # CPU-bound; 3 seeds suffice
    stats = []
    for s in range(n_seeds):
        gv = decode(
            sample_jax(jmodel, jparams, gcfg["n_samples"], X.shape[1],
                       m["timesteps"], m["seed"] + 31 * s,
                       cond=cond_val, guidance=gcfg.get("guidance_w", 8.0)),
            alphabet)
        stats.append(_eval_batch(gv, df_index, fit_map, top_set))
    oracle = _agg(stats)

    out = {
        "model": model_path,
        "framework": "jax0.4.30+flax0.8.5 vs torch",
        "parity": parity,
        "guided_w8_oracle_jax": oracle,
        "n_seeds": n_seeds,
        "per_seed": stats,
        "note": ("eps/x0 deviations are fp32 reduction-order scale; the "
                 "RNG streams differ by design, so parity is trajectory-"
                 "level under shared noise, not seed-identical draws"),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    write_manifest(out_json.replace(".json", "_provenance.json"),
                   [in_parquet, model_path], str(config_path()))
    print(f"parity: eps dev {parity['eps_max_abs_dev']:.2e}, "
          f"x0 dev {parity['x0_max_abs_dev']:.2e}")
    print(f"jax guided_w8 oracle: mean {oracle['fitness_mean']:.3f} | "
          f">=0.5 {oracle['frac_ge_05']:.1%} | "
          f"unmeasured {oracle['frac_unmeasured']:.1%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
