"""Shared ESM output cache: correctness of the content-addressed store."""
import importlib

import numpy as np
import pytest


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ESM_CACHE_DIR", str(tmp_path))
    import design_ops.esm_cache as esm_cache
    importlib.reload(esm_cache)  # rebind CACHE_DIR from the env var
    return esm_cache


def test_roundtrip_vector(cache):
    cache.put("esm2-embed", "m", "ACDE", np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(cache.get("esm2-embed", "m", "ACDE"),
                                  np.ones(4, dtype=np.float32))


def test_miss_returns_none(cache):
    assert cache.get("esm2-embed", "m", "XXXX") is None


def test_kind_and_model_partition(cache):
    cache.put("esm2-embed", "m1", "AC", np.array([1.0]))
    cache.put("esm2-pll", "m1", "AC", np.array([2.0]))
    cache.put("esm2-embed", "m2", "AC", np.array([3.0]))
    assert cache.get("esm2-pll", "m1", "AC") == 2.0
    assert cache.get("esm2-embed", "m2", "AC") == 3.0
    assert cache.get("esm2-pll", "m2", "AC") is None


def test_get_many_reports_miss_indices(cache):
    cache.put("esm2-embed", "m", "AA", np.array([9.0]))
    values, misses = cache.get_many("esm2-embed", "m", ["AA", "BB", "CC"])
    assert misses == [1, 2]
    assert values[0] == 9.0 and values[1] is None
