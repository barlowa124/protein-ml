"""Variant encoders.

Default: position-wise one-hot over the mutated sites. GB1's library is a
4-site combinatorial design (V39, D40, G41, V54), so one-hot is 4*20 = 80
dims and captures all single-site effects; interaction terms are learnable
by the GP's kernel.

esm2: mean-pooled ESM-2 embeddings. A bare mutated-region string (e.g.
GB1's 4-AA combos) carries no signal for a protein LM, so variants can be
embedded *in context*: `context_seq` is the WT parent sequence and
`context_sites` the 0-indexed positions the variant residues replace.
AAV's 28-aa mutated_region is long enough to embed directly.
"""

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids, fixed order


def one_hot(variants: pd.Series, alphabet: str = AA_ALPHABET) -> np.ndarray:
    """(n_variants, n_sites * len(alphabet)) one-hot over mutated positions."""
    idx = {aa: i for i, aa in enumerate(alphabet)}
    if len(variants) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n_sites = len(variants.iloc[0])
    X = np.zeros((len(variants), n_sites * len(alphabet)), dtype=np.float32)
    for i, v in enumerate(variants):
        if len(v) != n_sites:
            raise ValueError(f"inconsistent variant length: {v!r}")
        for site, aa in enumerate(v):
            j = idx.get(aa)
            if j is None:
                raise ValueError(f"non-standard residue {aa!r} in {v!r}")
            X[i, site * len(alphabet) + j] = 1.0
    return X


def apply_context(variant: str, context_seq: str, context_sites) -> str:
    """Substitute variant residues into the WT parent at 0-indexed sites."""
    seq = list(context_seq)
    if len(variant) != len(context_sites):
        raise ValueError(
            f"variant {variant!r} has {len(variant)} residues for "
            f"{len(context_sites)} context sites"
        )
    for site, aa in zip(context_sites, variant):
        seq[site] = aa
    return "".join(seq)


def esm2_embed(variants: pd.Series, model_name: str = "facebook/esm2_t6_8M_UR50D",
               context_seq: str = None, context_sites=None,
               batch_size: int = 256) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, (n_variants, d_model).

    Requires the optional `esm` extras (torch + transformers)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    seqs = [
        apply_context(v, context_seq, context_sites) if context_seq else v
        for v in variants
    ]
    # Per-sequence shared cache (vendored esm_cache): embeddings reuse
    # across repos and across variant-list changes. Keyed on the exact
    # string the model sees, post context substitution.
    from al_loop.esm_cache import get_many, put
    values, misses = get_many("esm2-embed", model_name, seqs)
    if not misses:
        return np.stack(values).astype(np.float32)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = np.zeros((len(misses), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(misses), batch_size):
            chunk = misses[i : i + batch_size]
            enc = tok([seqs[j] for j in chunk], return_tensors="pt",
                      padding=True).to(device)
            hidden = model(**enc).last_hidden_state  # (b, L, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1)
            out[i : i + batch_size] = emb.cpu().numpy()
    for k, j in enumerate(misses):
        values[j] = out[k]
        put("esm2-embed", model_name, seqs[j], out[k])
    return np.stack(values).astype(np.float32)


def build_features(variants: pd.Series, encoder_cfg: dict,
                   cache_stem: str = None) -> np.ndarray:
    """Dispatch on encoder kind; esm2 embeddings are cached to disk keyed
    by the exact variant list so reruns skip the forward passes."""
    kind = encoder_cfg.get("kind", "onehot")
    if kind == "onehot":
        return one_hot(variants, encoder_cfg.get("alphabet", AA_ALPHABET))
    if kind != "esm2":
        raise ValueError(f"unknown encoder kind {kind!r}")
    if cache_stem:
        from pathlib import Path
        import hashlib
        import json

        npy = Path(f"{cache_stem}_esm2.npy")
        key_f = Path(f"{cache_stem}_esm2.key")
        # the key must cover every input to the embedding: variant list
        # AND encoder params. A changed model/context with an unchanged
        # variant list must not silently reuse stale embeddings
        key = hashlib.sha256(
            (json.dumps(encoder_cfg, sort_keys=True)
             + "\n" + "\n".join(variants)).encode()
        ).hexdigest()
        if npy.exists() and key_f.exists() and key_f.read_text() == key:
            return np.load(npy)
        X = esm2_embed(variants, **{k: v for k, v in encoder_cfg.items()
                                  if k not in ("kind", "alphabet")})
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, X)
        key_f.write_text(key)
        return X
    return esm2_embed(variants, **{k: v for k, v in encoder_cfg.items()
                                  if k not in ("kind", "alphabet")})
