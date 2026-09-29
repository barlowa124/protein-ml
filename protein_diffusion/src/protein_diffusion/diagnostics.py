"""Landscape diagnostics the README quotes, committed as data.

Computes the library-density and generated-vs-library distance stats for
a configured landscape + checkpoint pair (AAV is the failing case these
numbers describe). Writes a compact JSON so the prose stays auditable.

    python -m protein_diffusion.diagnostics <parquet> <model> <out_json>

Config comes from DIFFUSION_CONFIG like every other entry point.
Pairwise stats use seeded subsamples of the library (full pairwise over
38k sequences is wasted compute for a median to one decimal).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from protein_diffusion.config import config_path, load_config
from protein_diffusion.ddpm import Denoiser, sample
from protein_diffusion.encode import decode, one_hot
from protein_diffusion.provenance import write_manifest
from protein_diffusion.run import _cond_transform

N_PAIRS = 4000          # sampled library pairs for pairwise distance
N_MEMBER = 512          # library members probed for nearest-member distance
N_REF = 4096            # reference subset for nearest-distance queries


def _char_matrix(variants, alphabet):
    idx = {c: i for i, c in enumerate(alphabet)}
    return np.array([[idx[c] for c in v] for v in variants], dtype=np.uint8)


def _site_stats(lib_chars):
    """Per-site Shannon entropy in nats."""
    n_sites = lib_chars.shape[1]
    entropy = np.zeros(n_sites)
    for s in range(n_sites):
        p = np.bincount(lib_chars[:, s], minlength=lib_chars.max() + 1)
        p = p[p > 0] / lib_chars.shape[0]
        entropy[s] = -(p * np.log(p)).sum()
    return entropy


def _min_hamming(query, refs):
    """Min Hamming distance from each query row to the ref set."""
    return np.array(
        [(q != refs).sum(1).min() for q in query], dtype=np.int32)


def main(in_parquet, model_path, out_json):
    cfg = load_config()
    ds = cfg["dataset"]
    alphabet = ds["alphabet"]
    rng = np.random.default_rng(2026)

    df = pd.read_parquet(in_parquet)
    variants = df["variant"].tolist()
    lib = _char_matrix(variants, alphabet)
    modal_seq = np.array(
        [np.bincount(lib[:, s], minlength=len(alphabet)).argmax()
         for s in range(lib.shape[1])])

    entropy = _site_stats(lib)
    uni_entropy = np.log(len(alphabet))
    variant_set = set(variants)

    # Library density: sampled pairwise distance + nearest-neighbor probe.
    ia = rng.integers(0, len(lib), N_PAIRS)
    ib = rng.integers(0, len(lib), N_PAIRS)
    pairwise = (lib[ia] != lib[ib]).sum(1)
    probe_idx = rng.choice(len(lib), N_MEMBER, replace=False)
    ref_idx = rng.choice(len(lib), N_REF, replace=False)
    ref = lib[ref_idx]
    nn = np.array([
        ((lib[i] != ref).sum(1).min()) for i in probe_idx], dtype=np.int32)
    member_modal = (lib[probe_idx] == modal_seq).mean(1)

    # Generated side: unconditioned + guided samples from the checkpoint.
    m = cfg["model"]
    gcfg = cfg["generate"]
    X = one_hot(df["variant"][:1], alphabet)
    model = Denoiser(X.shape[1], m["hidden"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    cond_val = float(_cond_transform(
        df["fitness"].to_numpy(), ds.get("transform", "log1p")
    )[1](gcfg["cond_fitness"]))

    generated = {}
    for label, cond, w in (
        ("unconditioned", None, 0.0),
        ("guided", cond_val, gcfg.get("guidance_w", 8.0)),
    ):
        gv = decode(sample(model, gcfg["n_samples"], X.shape[1],
                           m["timesteps"], m["seed"], cond=cond,
                           guidance=w), alphabet)
        g = _char_matrix(gv, alphabet)
        generated[label] = {
            "n": len(gv),
            "n_unique": int(len(set(gv))),
            "min_library_dist_median": float(np.median(_min_hamming(g, ref))),
            "modal_match_mean": float((g == modal_seq).mean(1).mean()),
            "measured_fraction": float(
                np.mean([v in variant_set for v in gv])),
        }

    out = {
        "dataset": {"n_variants": len(lib), "seq_len": int(lib.shape[1]),
                    "alphabet_size": len(alphabet)},
        "library": {
            "pairwise_hamming_median": float(np.median(pairwise)),
            "nearest_member_dist_median": float(np.median(nn)),
            "site_entropy_median_nats": float(np.median(entropy)),
            "uniform_entropy_nats": float(uni_entropy),
            "modal_match_mean": float(member_modal.mean()),
            "sampled_pairs": N_PAIRS, "probe_members": N_MEMBER,
            "ref_subset": N_REF, "seed": 2026,
        },
        "generated": generated,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    write_manifest(
        out_json.replace(".json", "_provenance.json"),
        [in_parquet, model_path],
        str(config_path()))
    print(json.dumps({k: v for k, v in out.items() if k != "generated"},
                     indent=1))
    for k, v in generated.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
