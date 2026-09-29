"""Structure screen: fold designed sequences with ESMFold and record pLDDT.

Scores (MPNN score, ESM-2 pseudo-loglikelihood) are sequence-space
signals; pLDDT is a *structure* confidence. A design that ranks well on
both is stronger evidence than either alone. pLDDT here is ESMFold's
own confidence estimate, not an experimental structure — the README
says so wherever numbers are quoted.

Usage:
    python -m design_ops.fold scores.json fold_out.json folds_dir/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _lazy_model(model_id: str = "facebook/esmfold_v1"):
    """Import + load only when actually folding (~8 GB weights)."""
    import torch
    from transformers import EsmForProteinFolding
    m = EsmForProteinFolding.from_pretrained(model_id)
    return m.eval(), torch


def fold_one(seq: str, model, torch) -> dict:
    """Fold a single sequence; returns confidence metrics + PDB text."""
    with torch.no_grad():
        out = model.infer(seq)
    # transformers returns plddt on 0-1 (categorical bins linspace 0..1);
    # report on the conventional 0-100 scale so >70 means "confident"
    plddt = out["plddt"].squeeze() * 100.0
    return {
        "plddt_mean": float(plddt.mean()),
        "plddt_min": float(plddt.min()),
        "ptm": float(out["ptm"]) if "ptm" in out else None,
        # output_to_pdb is a batched API: returns a list of PDB strings,
        # one per input; infer() is called on a single sequence
        "pdb": (lambda p: p[0] if isinstance(p, list) else p)(
            model.output_to_pdb(out)),
    }


def fold_records(records: list[dict], model=None, torch=None,
                 model_id: str = "facebook/esmfold_v1") -> list[dict]:
    """Attach fold metrics to every record; native folds too as reference."""
    if model is None:
        model, torch = _lazy_model(model_id)
    out = []
    for i, rec in enumerate(records):
        if not rec.get("seq"):
            out.append({**rec, "fold": None, "fold_error": "empty seq"})
            continue
        try:
            f = fold_one(rec["seq"], model, torch)
            print(f"  folded {i + 1}/{len(records)} "
                  f"plddt={f['plddt_mean']:.1f}", flush=True)
            out.append({**rec, "fold": {k: f[k] for k in
                                        ("plddt_mean", "plddt_min", "ptm")},
                        "pdb": f["pdb"]})
        except Exception as e:  # fold failures are data, not crashes
            out.append({**rec, "fold": None, "fold_error": str(e)})
    return out


def main() -> None:
    scores_json, out_json, folds_dir = sys.argv[1:4]
    from design_ops.config import load_config
    cfg = load_config()
    records = json.loads(Path(scores_json).read_text())
    folded = fold_records(
        records,
        model_id=cfg.get("fold", {}).get("model", "facebook/esmfold_v1"))
    folds = Path(folds_dir)
    folds.mkdir(parents=True, exist_ok=True)
    clean = []
    for i, rec in enumerate(folded):
        pdb = rec.pop("pdb", None)
        if pdb is not None:
            # a failed PDB write must not lose the metrics for all records
            try:
                (folds / f"{i:03d}_{'native' if rec.get('is_native') else 'design'}.pdb"
                 ).write_text(pdb)
            except Exception as e:
                rec["fold_error"] = f"pdb write failed: {e}"
        clean.append(rec)
    Path(out_json).write_text(json.dumps(clean, indent=1))
    n_ok = sum(1 for r in clean if r["fold"])
    print(f"folded {n_ok}/{len(clean)} sequences -> {out_json}")


if __name__ == "__main__":
    main()
