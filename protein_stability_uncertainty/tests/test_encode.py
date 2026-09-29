import numpy as np
import pandas as pd
import pytest

from protstab.encode import AA_ALPHABET, build_features, composition


def test_composition_shape_and_normalization():
    X = composition(pd.Series(["ACDE", "AAAA"]))
    assert X.shape == (2, len(AA_ALPHABET) + 1)
    # AA fractions sum to 1 for clean strings
    assert np.isclose(X[0, :20].sum(), 1.0)
    assert X[1, 0] == 1.0  # all alanine
    assert X[1, -1] == pytest.approx(np.log1p(4), rel=1e-6)  # log length


def test_composition_ignores_nonstandard():
    # 'X' and '*' are not counted but must not crash
    X = composition(pd.Series(["AXA*"]))
    assert np.isclose(X[0, :20].sum() * 4, 2.0, atol=0.1) or \
        np.isclose(X[0, :20].sum(), 2 / 4)
    assert X.shape == (1, 21)


def test_build_features_dispatch():
    seqs = pd.Series(["ACDE", "FGHI"])
    X = build_features(seqs, {"kind": "composition"})
    assert X.shape == (2, 21)
    with pytest.raises(ValueError, match="unknown encoder"):
        build_features(seqs, {"kind": "bogus"})
