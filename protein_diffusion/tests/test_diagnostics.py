import numpy as np

from protein_diffusion.diagnostics import _char_matrix, _min_hamming, _site_stats


def test_char_matrix_maps_alphabet():
    m = _char_matrix(["AC", "DA"], "ACD")
    assert m.tolist() == [[0, 1], [2, 0]]


def test_site_stats_entropy_and_modal():
    # Site 0 uniform over 2 symbols (ln2 nats), site 1 fixed (0 nats).
    lib = _char_matrix(["AX", "AX", "DX", "DX"], "AXDY")
    entropy = _site_stats(lib)
    assert np.isclose(entropy[0], np.log(2))
    assert entropy[1] == 0.0


def test_min_hamming_finds_nearest():
    ref = _char_matrix(["AAAA", "TTTT"], "AT")
    q = _char_matrix(["AAAT"], "AT")
    assert _min_hamming(q, ref)[0] == 1
