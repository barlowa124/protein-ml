import numpy as np
import pandas as pd
import pytest

from al_loop.encode import AA_ALPHABET, one_hot


def test_one_hot_shape_and_normalization():
    X = one_hot(pd.Series(["VDGV", "AAAA"]))
    assert X.shape == (2, 4 * len(AA_ALPHABET))
    # each site contributes exactly one active bit
    for site in range(4):
        block = X[:, site * len(AA_ALPHABET) : (site + 1) * len(AA_ALPHABET)]
        assert np.all(block.sum(axis=1) == 1.0)


def test_one_hot_deterministic_and_distinct():
    a = one_hot(pd.Series(["VDGV"]))
    b = one_hot(pd.Series(["VDGV"]))
    c = one_hot(pd.Series(["ADGV"]))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_one_hot_rejects_bad_input():
    with pytest.raises(ValueError, match="non-standard"):
        one_hot(pd.Series(["VDG*"]))
    with pytest.raises(ValueError, match="inconsistent"):
        one_hot(pd.Series(["VDGV", "AAA"]))


def test_one_hot_extended_alphabet_accepts_stop():
    # AAV landscape keeps '*' (stop) as a real measured dead variant
    X = one_hot(pd.Series(["VDGV", "V*GV"]), alphabet=AA_ALPHABET + "*")
    assert X.shape == (2, 4 * 21)


def test_apply_context_substitutes_sites():
    from al_loop.encode import apply_context
    wt = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    sites = [38, 39, 40, 53]
    assert "".join(wt[i] for i in sites) == "VDGV"
    out = apply_context("AAAA", wt, sites)
    assert len(out) == len(wt)
    assert "".join(out[i] for i in sites) == "AAAA"
    # positions outside the sites are untouched
    assert out[:38] == wt[:38]
    with pytest.raises(ValueError):
        apply_context("AAA", wt, sites)


def test_build_features_dispatch_and_cache(tmp_path):
    import numpy as np
    from al_loop.encode import build_features

    variants = pd.Series(["VDGV", "AAAA"])
    X = build_features(variants, {"kind": "onehot"})
    assert X.shape == (2, 80)
    with pytest.raises(ValueError, match="unknown encoder"):
        build_features(variants, {"kind": "bogus"})
