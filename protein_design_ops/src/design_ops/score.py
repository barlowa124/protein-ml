"""Independent rescoring: ESM-2 mean pseudo-log-likelihood per sequence.

For each candidate, mask every position in turn and take the model's
log-probability of the actual residue; the mean over positions is the
sequence's zero-shot fitness score. This is an independent second opinion
on MPNN's own score. The cross-model agreement is the report's signal.
"""

from __future__ import annotations

import json
import sys
from design_ops.config import load_config




def pll(seq: str, tok, model) -> float:
    """Mean masked-marginal log-prob of a sequence. Pure w.r.t. model+tok.

    All len(seq) single-position masks are scored in one batched forward
    pass instead of N sequential ones.
    """
    import torch

    n = len(seq)
    masked = [seq[:i] + tok.mask_token + seq[i + 1 :] for i in range(n)]
    inputs = tok(masked, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (n, L, V)
    mask_pos = (inputs["input_ids"] == tok.mask_token_id).nonzero()
    target_ids = torch.tensor(
        [tok.convert_tokens_to_ids(c) for c in seq]
    )
    lp = torch.log_softmax(
        logits[mask_pos[:, 0], mask_pos[:, 1]], dim=-1
    )
    return float(lp[torch.arange(n), target_ids].sum() / n)


def score_candidates(records: list[dict], model_name: str) -> list[dict]:
    from transformers import AutoTokenizer, EsmForMaskedLM

    from design_ops.esm_cache import get, put

    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).eval()
    for r in records:
        cached = get("esm2-pll", model_name, r["seq"])
        if cached is None:
            score = round(pll(r["seq"], tok, model), 4)
            put("esm2-pll", model_name, r["seq"], score)
        else:
            score = round(float(cached), 4)
        r["esm_pll"] = score
    return records


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    cfg = load_config()
    with open(in_path) as f:
        records = json.load(f)
    records = score_candidates(records, cfg["esm"]["model"])
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    designed = [r for r in records if not r["is_native"]]
    print(
        f"score: {len(designed)} designed + "
        f"{len(records) - len(designed)} native scored -> {out_path}"
    )


if __name__ == "__main__":
    main()
