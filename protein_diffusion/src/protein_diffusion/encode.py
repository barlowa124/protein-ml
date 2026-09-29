"""One-hot encode/decode for combinatorial variant spaces.

The DDPM operates on the continuous relaxation of the (n_sites, alphabet)
one-hot tensor; generation decodes by per-site argmax, which always yields
a valid variant string by construction. Site count is inferred from the
data. GB1 is 4x20, AAV's mutated_region is 28x21 (incl. '*' stops).
"""

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def one_hot(variants: pd.Series, alphabet: str = AA_ALPHABET) -> np.ndarray:
    """(n, n_sites*len(alphabet)) float32 one-hot."""
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


def decode(X: np.ndarray, alphabet: str = AA_ALPHABET) -> list:
    """Per-site argmax over the (n, n_sites, |alphabet|) logits."""
    X = np.asarray(X)
    n_sites = X.shape[1] // len(alphabet)
    X = X.reshape(-1, n_sites, len(alphabet))
    return [
        "".join(alphabet[j] for j in row.argmax(axis=1)) for row in X
    ]


def mutate(variant: str, k: int, rng: np.random.Generator,
           alphabet: str = AA_ALPHABET) -> str:
    """k random substitutions at distinct sites (k is clamped to >=1 by
    the caller. The baseline tests *novel* mutants of fit parents, not
    parent resampling)."""
    v = list(variant)
    sites = rng.choice(len(v), size=min(k, len(v)), replace=False)
    for s in sites:
        v[s] = alphabet[rng.integers(len(alphabet))]
    return "".join(v)
