"""Compare active vs random trajectories and render the summary artifacts.

Metrics:
- best-fitness curve: max measured fitness found vs experiments spent
- top-k hit curve: how many of the landscape's true top-k variants were
  discovered vs experiments spent
- AUBC: area under the best-fitness curve (normalized by oracle max and
  budget), a single scalar "learning speed" summary per trajectory
- hit rate @ budget: fraction of true top-k found when the budget is spent

Random baseline is aggregated over seeds with mean +/- std bands; the
active policy is a single seeded trajectory (reported as such).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from al_loop.config import config_path, load_config
from al_loop.provenance import write_manifest


def aubc(n_exp: np.ndarray, best: np.ndarray, y_max: float, budget: int) -> float:
    """Area under best-fitness curve, normalized to [0,1] vs the oracle."""
    auc = np.trapezoid(best, n_exp)
    return float(auc / (y_max * budget))


def summarize(records: pd.DataFrame, y_max: float, top_k: int, budget: int):
    out = {}
    for strategy, g in records.groupby("strategy"):
        g = g.sort_values("n_experiments")
        final = g.iloc[-1]
        out[strategy] = {
            "aubc": aubc(
                g["n_experiments"].to_numpy(),
                g["best_fitness"].to_numpy(),
                y_max,
                budget,
            ),
            "best_fitness": float(final["best_fitness"]),
            "top_hits_at_budget": int(final["top_hits_found"]),
            "top_hit_rate": float(final["top_hits_found"]) / top_k,
        }
    return out


def plot_curves(records: pd.DataFrame, y_max: float, out_png: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rand = records[records.strategy.str.startswith("random")]
    active = records[records.strategy.str.startswith("active")]
    for ax, col, title in zip(
        axes,
        ["best_fitness", "top_hits_found"],
        ["best measured fitness found", "true top-100 variants discovered"],
    ):
        for subset, color, name in (
            (rand, "gray", "random"),
            (active, "crimson", "active (UCB-GP)"),
        ):
            pivot = subset.pivot_table(
                index="n_experiments", columns="strategy", values=col
            ).sort_index()
            m, s = pivot.mean(axis=1), pivot.std(axis=1)
            ax.plot(m.index, m, color=color,
                    label=f"{name} (n={pivot.shape[1]})")
            ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.2)
        if col == "best_fitness":
            ax.axhline(y_max, ls=":", color="k", alpha=0.4)
        ax.set_xlabel("experiments spent")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)


def main(records_path: str, landscape_path: str, out_json: str, out_png: str):
    import json

    cfg = load_config()
    records = pd.read_parquet(records_path)
    y = pd.read_parquet(landscape_path)["fitness"].to_numpy()
    y_max = float(y.max())

    summary = summarize(
        records, y_max, cfg["evaluation"]["top_k"], cfg["experiment"]["budget"]
    )
    rand = {k: v for k, v in summary.items() if k.startswith("random")}
    act = {k: v for k, v in summary.items() if k.startswith("active")}

    def _agg(g):
        return {
            "n_seeds": len(g),
            "aubc_mean": float(np.mean([v["aubc"] for v in g.values()])),
            "aubc_std": float(np.std([v["aubc"] for v in g.values()])),
            "best_fitness_mean": float(
                np.mean([v["best_fitness"] for v in g.values()])
            ),
            "top_hits_mean": float(
                np.mean([v["top_hits_at_budget"] for v in g.values()])
            ),
        }

    result = {
        "config": {
            "surrogate": cfg["surrogate"]["kind"],
            "acquisition": cfg["acquisition"]["kind"],
            "n_init": cfg["experiment"]["n_init"],
            "batch_size": cfg["experiment"]["batch_size"],
            "budget": cfg["experiment"]["budget"],
            "top_k": cfg["evaluation"]["top_k"],
        },
        "landscape": {
            "n_variants": int(len(y)),
            "oracle_max_fitness": y_max,
        },
        "active": _agg(act),
        "random": _agg(rand),
        "per_trajectory": summary,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    plot_curves(records, y_max, out_png)
    manifest_name = "provenance" + Path(out_json).stem.removeprefix("summary") + ".json"
    write_manifest(
        str(Path(out_json).parent / manifest_name),
        inputs=[records_path, landscape_path],
        config_path=str(config_path()),
    )
    print(
        f"active AUBC {result['active']['aubc_mean']:.3f}±"
        f"{result['active']['aubc_std']:.3f} vs random "
        f"{result['random']['aubc_mean']:.3f}±{result['random']['aubc_std']:.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
