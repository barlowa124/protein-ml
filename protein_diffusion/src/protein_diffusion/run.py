"""Conditional DDPM over the full GB1 landscape, oracle-evaluated.

v2 design, after the v1 failure (see README debugging trail): the
unconditional model trained only on fit variants could not produce novel
fit variants. GB1's functional region is an archipelago (Hamming-1
neighbors of fit variants measure ~0.087, i.e. dead), so density
estimation over a sparse fit set has nothing to interpolate toward and
memorizes instead.

The fix is fitness conditioning + classifier-free guidance, trained on
the FULL landscape. The model must see the dead variants to learn where
the boundary is. Evaluation is a steering experiment:

- unconditioned samples should reproduce the landscape distribution
  (sanity: mean fitness ~ random)
- samples conditioned on high fitness, with guidance w, should shift
  toward the functional region, measured against the oracle

Because 93% of the 20^4 space is measured, virtually every decoded
variant exists in the training data: the claim is steering, not
novelty. Diversity (unique variants, top-hit concentration) is reported
to show conditioning isn't just replaying the top rows.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protein_diffusion.config import config_path, load_config
from protein_diffusion.ddpm import sample, train
from protein_diffusion.encode import decode, mutate, one_hot
from protein_diffusion.provenance import write_manifest


def _stats(fitness: np.ndarray, top_set: set, idx: np.ndarray, train_set: set):
    return {
        "n": int(len(fitness)),
        "fitness_mean": float(np.nanmean(fitness)),
        "fitness_median": float(np.nanmedian(fitness)),
        "frac_ge_1": float(np.mean(fitness >= 1.0)),
        "frac_ge_05": float(np.mean(fitness >= 0.5)),
        "top_hits": int(sum(i in top_set for i in idx)),
    }


def _eval_batch(variants, df_index, fit_map, top_set):
    gi = np.array([df_index.get(v, -1) for v in variants])
    meas = gi >= 0
    gf = np.array(
        [fit_map[v] if i >= 0 else np.nan for v, i in zip(variants, gi)]
    )
    st = _stats(np.nan_to_num(gf, nan=0.0), top_set, gi[meas], set())
    st["n_unique"] = len(set(variants))
    st["frac_unmeasured"] = float(1 - meas.mean())
    return st


def _agg(stats_list):
    keys = [k for k in stats_list[0] if k != "n"]
    return {
        **{k: float(np.mean([s[k] for s in stats_list])) for k in
           ["n", *keys]},
        **{f"{k}_std": float(np.std([s[k] for s in stats_list]))
           for k in keys},
    }


def _cond_transform(y: np.ndarray, transform: str):
    """Map raw fitness to the conditioning channel.

    The denoiser's null-condition token is -1, so the transformed channel
    must be >= ~0: log1p suits GB1's nonnegative heavy tail; AAV's
    log-viability score goes to -11, so it is shifted into the positive
    range first."""
    if transform == "log1p":
        return np.log1p(y), lambda v: np.log1p(v)
    if transform == "shift_log1p":
        shift = y.min() - 0.01
        return np.log1p(y - shift), lambda v: np.log1p(v - shift)
    raise ValueError(f"unknown transform {transform!r}")


def main(in_parquet: str, out_json: str, out_model: str):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    ds = cfg["dataset"]
    alphabet = ds.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")
    fit_map = dict(zip(df["variant"], df["fitness"]))
    X = one_hot(df["variant"], alphabet)
    y_all = df["fitness"].to_numpy()
    y_cond, to_cond = _cond_transform(
        y_all, ds.get("transform", "log1p"))
    top_set = set(np.argsort(-y_all)[: cfg["evaluation"]["top_k"]].tolist())
    df_index = {v: i for i, v in enumerate(df["variant"])}

    m = cfg["model"]
    model = train(X, m["timesteps"], m["hidden"], m["lr"], m["epochs"],
                  m["batch_size"], m["seed"], y=y_cond,
                  cond_drop=cfg["generate"].get("cond_drop", 0.15))
    import torch

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model)

    n_gen = cfg["generate"].get("n_seeds", 1)
    cond_val = float(to_cond(cfg["generate"]["cond_fitness"]))
    gcfg = cfg["generate"]

    sets = {}
    for label, cond, w in (
        ("unconditioned", None, 0.0),
        ("conditioned", cond_val, 0.0),
        ("guided_w4", cond_val, gcfg.get("guidance_w_low", 4.0)),
        ("guided_w8", cond_val, gcfg.get("guidance_w", 8.0)),
    ):
        stats = []
        for s in range(n_gen):
            gv = decode(
                sample(model, gcfg["n_samples"], X.shape[1],
                       m["timesteps"], m["seed"] + 31 * s,
                       cond=cond, guidance=w),
                alphabet,
            )
            stats.append(_eval_batch(gv, df_index, fit_map, top_set))
        sets[label] = {**_agg(stats), "per_seed": stats}
        print(
            f"{label}: mean fitness {sets[label]['fitness_mean']:.3f} "
            f"| >=0.5 {sets[label]['frac_ge_05']:.2%} "
            f"| top-100 hits {sets[label]['top_hits']:.1f}"
        )

    rng_stats = []
    for s in range(cfg["evaluation"]["n_random_seeds"]):
        r = np.random.default_rng(m["seed"] + 100 + s)
        ri = r.choice(len(df), size=gcfg["n_samples"], replace=False)
        rng_stats.append(_stats(y_all[ri], top_set, ri, set()))
    rand_mean = {k: float(np.mean([s[k] for s in rng_stats]))
                 for k in rng_stats[0]}

    # Mutational baseline: the "why diffuse?" counterfactual:
    # sample a fit parent (fitness >= the conditioning target), apply
    # max(1, Poisson(mu)) random substitutions, oracle-score the children.
    # If guidance only replays local neighborhoods, this trivial baseline
    # should match it; the v1 deconstruction says it can't.
    parents = df.loc[y_all >= gcfg["cond_fitness"], "variant"].tolist()
    for mu in gcfg.get("baseline_mu", [1.0, 2.0]):
        stats = []
        for s in range(cfg["evaluation"]["n_random_seeds"]):
            r = np.random.default_rng(m["seed"] + 500 + s)
            pv = r.choice(parents, size=gcfg["n_samples"], replace=True)
            gv = [mutate(v, max(1, r.poisson(mu)), r, alphabet)
                  for v in pv]
            stats.append(_eval_batch(gv, df_index, fit_map, top_set))
        sets[f"mutate_mu{mu:g}"] = {**_agg(stats), "per_seed": stats}
        print(
            f"mutate_mu{mu:g}: mean fitness "
            f"{sets[f'mutate_mu{mu:g}']['fitness_mean']:.3f} | >=0.5 "
            f"{sets[f'mutate_mu{mu:g}']['frac_ge_05']:.2%}"
        )

    result = {
        "config": {
            "n_train": int(len(df)),
            "timesteps": m["timesteps"],
            "epochs": m["epochs"],
            "n_samples": gcfg["n_samples"],
            "cond_fitness": cfg["generate"]["cond_fitness"],
            "guidance_w": gcfg.get("guidance_w", 2.0),
        },
        "generated": sets,
        "random": {"n_seeds": len(rng_stats), "mean": rand_mean},
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    manifest = (
        Path(out_json).parent
        / ("provenance" + Path(out_json).stem.removeprefix("summary") + ".json")
    )
    write_manifest(str(manifest), inputs=[in_parquet], config_path=str(config_path()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
