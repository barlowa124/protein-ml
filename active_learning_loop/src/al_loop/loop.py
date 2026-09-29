"""The experiment-selection simulation.

Oracle model: `y` contains *measured* fitness for every variant. Querying a
variant costs one experiment and reveals its true value. The active loop
starts from `n_init` random labels and iterates: fit surrogate -> score
unlabeled pool with the acquisition function -> measure the top `batch_size`
candidates -> repeat until `budget` experiments are spent.

The random baseline follows the identical schedule but picks candidates
uniformly, replicated `n_random_seeds` times for a distribution. This is
the matched counterfactual: same budget and schedule, no model. (Each
trajectory, active or random, draws its own initial screen. The
multi-seed replication is what makes the comparison fair.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from al_loop.acquisition import acquire
from al_loop.config import load_config
from al_loop.encode import build_features
from al_loop.surrogate import fit_predict


def _snapshot(labeled: set, y: np.ndarray, top_set: set) -> dict:
    lab = np.array(sorted(labeled))
    return {
        "n_experiments": len(lab),
        "best_fitness": float(y[lab].max()),
        "top_hits_found": len(top_set & labeled),
    }


def apply_transform(y, transform: str):
    """Model-space fitness transform. GB1's enrichment is a heavy-tailed
    ratio (log1p); AAV's log-viability score is already symmetric
    (identity). Reported metrics stay on the raw scale."""
    if transform == "log1p":
        return np.log1p(y)
    if transform == "identity":
        return y
    raise ValueError(f"unknown transform {transform!r}")


def run_active(X, y, exp, sur, acq, top_set, seed=None,
               transform="log1p"):
    """One AL trajectory. Returns (round snapshots, ordered pick indices)."""
    rng = np.random.default_rng(exp["seed"] if seed is None else seed)
    labeled = set(rng.choice(len(X), size=exp["n_init"], replace=False).tolist())
    picks = sorted(labeled)
    records = [_snapshot(labeled, y, top_set)]
    y_model = apply_transform(y, transform)
    while len(labeled) < exp["budget"]:
        lab = np.array(sorted(labeled))
        unl = np.array([i for i in range(len(X)) if i not in labeled])
        mean, std = fit_predict(
            X[lab], y_model[lab], X[unl], sur["kind"],
            length_scale=sur.get("length_scale", 1.5),
        )
        scores = acquire(
            mean,
            std,
            acq["kind"],
            acq["kappa"],
            acq["xi"],
            float(y_model[lab].max()),
        )
        k = min(exp["batch_size"], exp["budget"] - len(labeled))
        batch = unl[np.argsort(-scores)[:k]]
        labeled.update(batch.tolist())
        picks.extend(batch.tolist())
        records.append(_snapshot(labeled, y, top_set))
    return records, picks


def run_random(X, y, exp, seed, top_set):
    """Random-selection trajectory under the identical experiment schedule."""
    rng = np.random.default_rng(seed)
    labeled = set(rng.choice(len(X), size=exp["n_init"], replace=False).tolist())
    records = [_snapshot(labeled, y, top_set)]
    while len(labeled) < exp["budget"]:
        unl = np.array([i for i in range(len(X)) if i not in labeled])
        k = min(exp["batch_size"], exp["budget"] - len(labeled))
        labeled.update(rng.choice(unl, size=k, replace=False).tolist())
        records.append(_snapshot(labeled, y, top_set))
    return records


def main(in_parquet: str, out_records: str, out_picks: str):
    cfg = load_config()
    exp, sur, acq = cfg["experiment"], cfg["surrogate"], cfg["acquisition"]
    df = pd.read_parquet(in_parquet)
    enc = cfg.get("encoder") or {
        "kind": "onehot", "alphabet": cfg["dataset"]["alphabet"]
    }
    stem = str(Path(in_parquet).with_suffix(""))
    X, y = build_features(df["variant"], enc, cache_stem=stem), \
        df["fitness"].to_numpy()
    top_set = set(np.argsort(-y)[: cfg["evaluation"]["top_k"]].tolist())

    # Active policy is replicated over seeds too. A single trajectory vs a
    # 20-seed random distribution was the old comparison and it wasn't fair.
    n_active = exp.get("n_active_seeds", 1)
    rows, first_picks = [], None
    best_last = []
    for s in range(n_active):
        seed = exp["seed"] + 1000 * s
        recs, picks = run_active(X, y, exp, sur, acq, top_set, seed=seed,
                                 transform=cfg["dataset"].get("transform",
                                                              "log1p"))
        for r in recs:
            rows.append({"strategy": f"active_{s}", **r})
        best_last.append(recs[-1]["best_fitness"])
        if s == 0:
            first_picks = picks
    for s in range(exp["n_random_seeds"]):
        for r in run_random(X, y, exp, exp["seed"] + 1 + s, top_set):
            rows.append({"strategy": f"random_{s}", **r})
    Path(out_records).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_records, index=False)

    if first_picks is None:
        raise ValueError("n_active_seeds must be >= 1 to record picks")
    # What the seed-13 active trajectory chose to measure, in acquisition
    # order, inspectable evidence of the policy's decisions.
    picks = df.iloc[first_picks].copy()
    picks["acquisition_order"] = range(len(picks))
    picks["initial_screen"] = picks["acquisition_order"] < exp["n_init"]
    Path(out_picks).parent.mkdir(parents=True, exist_ok=True)
    picks.to_parquet(out_picks, index=True)
    print(
        f"records: {len(rows)} rows | {n_active} active trajectories, "
        f"best found {min(best_last):.3f}..{max(best_last):.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
