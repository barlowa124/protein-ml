"""Oracle-evaluate the DDP-trained checkpoint against the measured landscape.

Same scoring path as `run.main` (decode -> _eval_batch -> _agg) so the
DDP-vs-single-process comparison is an apples-to-apples artifact, not a
separately computed anecdote. Writes `results/eval_ddp.json`.
"""

import json

import numpy as np
import pandas as pd
import torch

from protein_diffusion.config import load_config
from protein_diffusion.ddpm import Denoiser, sample
from protein_diffusion.encode import decode, one_hot
from protein_diffusion.run import _agg, _cond_transform, _eval_batch
from protein_diffusion.provenance import write_manifest


def main(in_parquet: str, model_path: str, out_json: str):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    ds = cfg["dataset"]
    alphabet = ds.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")
    fit_map = dict(zip(df["variant"], df["fitness"]))
    X = one_hot(df["variant"], alphabet)
    y_all = df["fitness"].to_numpy()
    _, to_cond = _cond_transform(y_all, ds.get("transform", "log1p"))
    top_set = set(np.argsort(-y_all)[: cfg["evaluation"]["top_k"]].tolist())
    df_index = {v: i for i, v in enumerate(df["variant"])}

    m = cfg["model"]
    model = Denoiser(X.shape[1], m["hidden"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    gcfg = cfg["generate"]
    cond_val = float(to_cond(gcfg["cond_fitness"]))
    stats = []
    for s in range(gcfg.get("n_seeds", 1)):
        gv = decode(
            sample(model, gcfg["n_samples"], X.shape[1],
                   m["timesteps"], m["seed"] + 31 * s,
                   cond=cond_val, guidance=gcfg.get("guidance_w", 8.0)),
            alphabet,
        )
        stats.append(_eval_batch(gv, df_index, fit_map, top_set))

    out = {
        "model": model_path,
        "mode": "guided_w8",
        "n_seeds": gcfg.get("n_seeds", 1),
        "guidance_w": gcfg.get("guidance_w", 8.0),
        **{f"guided_w8_{k}": v for k, v in _agg(stats).items() if k != "per_seed"},
        "per_seed": stats,
    }
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    write_manifest(out_json.replace(".json", "_provenance.json"),
                   [in_parquet, model_path])
    print(f"guided_w8: mean {out['guided_w8_fitness_mean']:.3f} | "
          f">=0.5 {out['guided_w8_frac_ge_05']:.1%} | "
          f"unmeasured {out['guided_w8_frac_unmeasured']:.1%}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
