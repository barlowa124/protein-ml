"""Report: cross-model consensus + diversity for designed candidates.

Each designed sequence has two independent scores: MPNN's own
backbone-conditioned likelihood and ESM-2's zero-shot sequence fitness.
Consensus candidates, strong on both, are the defensible shortlist.
"""

from __future__ import annotations

import json
import subprocess
import sys


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from design_ops.config import load_config


def identity(a: str, b: str) -> float:
    """Fraction of identical positions between equal-length sequences."""
    assert len(a) == len(b)
    return sum(x == y for x, y in zip(a, b)) / len(a)


def pairwise_identities(seqs: list[str]) -> list[float]:
    return [
        identity(a, b)
        for i, a in enumerate(seqs)
        for b in seqs[i + 1 :]
    ]


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation: Pearson correlation of scipy rankdata."""
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])




def _git_head(path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _pkg_versions() -> dict:
    from importlib.metadata import version

    out = {}
    for pkg in ("torch", "transformers", "numpy", "scipy"):
        try:
            out[pkg] = version(pkg)
        except Exception:
            pass
    return out


def _provenance(backbone: dict, cfg: dict) -> dict:
    """Reproducibility record: model ids, upstream commit, run params."""
    m = cfg["mpnn"]
    return {
        "backbone": {
            "id": cfg["backbone"].get("id"),
            "pdb": backbone["pdb"],
            "chain": backbone["chain"],
            "n_residues": backbone["n_residues"],
        },
        "generator": {
            "tool": "ProteinMPNN (dauparas/ProteinMPNN, external)",
            "repo_commit": _git_head(m["repo_path"]),
            "model_name": m["model_name"],
            "weights": m["weights"],
            "num_seq_per_target": m["num_seq_per_target"],
            "sampling_temp": m["sampling_temp"],
            "seed": m["seed"],
        },
        "scorer": {"tool": "ESM-2 masked-marginal PLL", "model": cfg["esm"]["model"]},
        "git_commit": _git_head("."),
        "versions": _pkg_versions(),
    }


def report(records: list[dict], backbone: dict, cfg: dict,
           out_json: str, out_png: str, folds: list[dict] = None) -> dict:
    designed = [r for r in records if not r["is_native"]]
    natives = [r for r in records if r["is_native"]]
    if not natives:
        raise ValueError("no native record in input, check generate parsing")
    if not designed:
        raise ValueError("no designed candidates in input, nothing to rank")
    native = natives[0]
    seqs = [r["seq"] for r in designed]
    mpnn = np.array([r["mpnn_score"] for r in designed])
    esm = np.array([r["esm_pll"] for r in designed])
    rec = np.array([r["seq_recovery"] for r in designed])

    # Consensus rank: mean of per-model ranks (lower mpnn score is better
    # in ProteinMPNN's convention? score is -logP -> lower is better;
    # esm_pll is logP -> higher is better).
    mpnn_rank = mpnn.argsort().argsort()          # 0 = best (lowest)
    esm_rank = (-esm).argsort().argsort()          # 0 = best (highest)
    consensus = mpnn_rank + esm_rank
    top_k = int(cfg.get("report", {}).get("top_k", 3))
    if top_k < 1:
        raise ValueError(f"report.top_k must be >= 1, got {top_k}")
    top_k = min(top_k, len(designed))
    top_idx = consensus.argsort()[:top_k]

    # Optional ESMFold confidence from the fold stage: pLDDT is the
    # model's own folding confidence, not an experimental structure.
    fold_by_seq = {}
    if folds:
        for fr in folds:
            if fr.get("seq") and fr.get("fold"):
                fold_by_seq[fr["seq"]] = fr["fold"]

    # Rank statistics need >=2 candidates; below that report nulls rather
    # than letting NaN fail at json.dump(allow_nan=False).
    pw = pairwise_identities(seqs)
    result = {
        "provenance": _provenance(backbone, cfg),
        "n_designed": len(designed),
        "native_seq": native["seq"],
        "native_esm_pll": native["esm_pll"],
        "mean_pairwise_identity": round(float(np.mean(pw)), 3) if pw else None,
        "mean_seq_recovery": round(float(rec.mean()), 3),
        "score_correlation_spearman": (
            round(spearman(list(mpnn), list(esm)), 3)
            if len(designed) >= 2
            else None
        ),
        f"consensus_top{top_k}": [
            {
                "seq": designed[i]["seq"],
                "mpnn_score": designed[i]["mpnn_score"],
                "esm_pll": designed[i]["esm_pll"],
                "seq_recovery": designed[i]["seq_recovery"],
                **({"plddt_mean": round(
                        fold_by_seq[designed[i]["seq"]]["plddt_mean"], 1),
                    "ptm": round(
                        fold_by_seq[designed[i]["seq"]]["ptm"], 3)}
                   if designed[i]["seq"] in fold_by_seq else {}),
            }
            for i in top_idx
        ],
        "designed_esm_pll": {
            "min": round(float(esm.min()), 3),
            "max": round(float(esm.max()), 3),
            "mean": round(float(esm.mean()), 3),
        },
    }
    if folds is not None:
        dpl = [fold_by_seq[r["seq"]]["plddt_mean"]
               for r in designed if r["seq"] in fold_by_seq]
        result["fold_screen"] = {
            "n_folded": len(dpl),
            "n_unfolded": len(designed) - len(dpl),
            "native_plddt": (round(
                fold_by_seq[native["seq"]]["plddt_mean"], 1)
                if native["seq"] in fold_by_seq else None),
            # >70 is the conventional "confident" pLDDT band
            "designed_plddt_mean": (
                round(float(np.mean(dpl)), 1) if dpl else None),
            "n_confident_ge70": int(sum(v >= 70 for v in dpl)),
        }

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mpnn, esm, c=rec, cmap="viridis", s=40)
    ax.scatter(
        [native["mpnn_score"]], [native["esm_pll"]],
        marker="*", s=200, c="red", label="native",
    )
    for i in top_idx:
        ax.annotate(
            f"#{i}", (mpnn[i], esm[i]), fontsize=8, xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("MPNN score (-logP, lower = better)")
    ax.set_ylabel("ESM-2 mean PLL (higher = better)")
    ax.set_title(
        f"cross-model consensus, spearman={result['score_correlation_spearman']}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result


def main() -> None:
    in_path, bb_path, out_json, out_png = sys.argv[1:5]
    cfg = load_config()
    with open(in_path) as f:
        records = json.load(f)
    with open(bb_path) as f:
        backbone = json.load(f)
    folds = None
    if len(sys.argv) > 5:
        with open(sys.argv[5]) as f:
            folds = json.load(f)
    result = report(records, backbone, cfg, out_json, out_png, folds)
    print(
        f"report: {result['n_designed']} designed, consensus "
        f"spearman={result['score_correlation_spearman']} -> {out_json}"
    )


if __name__ == "__main__":
    main()
