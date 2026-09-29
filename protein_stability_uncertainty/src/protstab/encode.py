"""Sequence encoders.

esm2:        mean-pooled ESM-2 last hidden state (the claim under test:
             a protein LM should carry biophysical signal).
composition: the no-LM baseline: 20 amino-acid frequencies + log length.
             If ESM-2 can't beat this, the embedding earns nothing.
"""

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def composition(seqs: pd.Series) -> np.ndarray:
    idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    X = np.zeros((len(seqs), len(AA_ALPHABET) + 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = max(len(s), 1)
        counts = np.zeros(len(AA_ALPHABET))
        for c in s:
            j = idx.get(c)
            if j is not None:
                counts[j] += 1
        X[i, :-1] = counts / n
        X[i, -1] = np.log1p(n)
    return X


def _token_budget_batches(lengths, batch_size, token_budget):
    """Yield index slices: at most `batch_size` seqs AND at most
    `token_budget` total padded tokens per batch. Attention memory scales
    with batch_len^2 x batch_count, which a fixed-size batch can't bound
    on a protein corpus (10k+ aa outliers OOM MPS)."""
    i, n = 0, len(lengths)
    while i < n:
        # lengths are sorted ascending: padded cost of batch i..j is
        # count * lengths[j] (the batch's longest member)
        j = i
        while (j < n and j - i < batch_size
               and lengths[j] * (j - i + 1) <= token_budget):
            j += 1
        yield slice(i, max(j, i + 1))
        i = max(j, i + 1)


def esm2_embed(seqs: pd.Series, model_name: str = "facebook/esm2_t6_8M_UR50D",
               batch_size: int = 128, max_len: int = 1024,
               token_budget: int = 16384) -> np.ndarray:
    """Mean-pooled last hidden state over residue positions.
    Sequences longer than `max_len` are truncated (counted by caller)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    seqs = [str(s)[:max_len] for s in seqs]
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    # length-sorted, token-budgeted batching: padding waste collapses when
    # batch members share a length, and memory stays bounded on outliers
    order = np.argsort([len(s) for s in seqs])
    sorted_seqs = [seqs[i] for i in order]
    lens = [len(s) + 2 for s in sorted_seqs]  # +2 for BOS/EOS
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for sl in _token_budget_batches(lens, batch_size, token_budget):
            enc = tok(sorted_seqs[sl], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=max_len).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out[order[sl]] = (
                (hidden * mask).sum(1) / mask.sum(1)
            ).cpu().numpy()
            del enc, hidden, mask
            if device == "mps":
                torch.mps.empty_cache()  # MPS allocator retains across
                                         # batches; flush or it OOMs
    return out


def build_features(seqs: pd.Series, encoder_cfg: dict,
                   cache_stem: str = None) -> np.ndarray:
    kind = encoder_cfg.get("kind", "esm2")
    if kind == "composition":
        return composition(seqs)
    if kind != "esm2":
        raise ValueError(f"unknown encoder kind {kind!r}")
    if cache_stem:
        from pathlib import Path
        import hashlib
        import json

        npy = Path(f"{cache_stem}_esm2.npy")
        key_f = Path(f"{cache_stem}_esm2.key")
        # key covers encoder params + exact sequence list. A config change
        # must not silently reuse stale embeddings
        key = hashlib.sha256(
            (json.dumps(encoder_cfg, sort_keys=True)
             + "\n" + "\n".join(seqs)).encode()
        ).hexdigest()
        if npy.exists() and key_f.exists() and key_f.read_text() == key:
            return np.load(npy)
        X = esm2_embed(
            seqs,
            **{k: v for k, v in encoder_cfg.items() if k != "kind"})
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, X)
        key_f.write_text(key)
        return X
    return esm2_embed(
        seqs, **{k: v for k, v in encoder_cfg.items() if k != "kind"})
