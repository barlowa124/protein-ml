import numpy as np
import pytest

from al_loop.acquisition import acquire, ei, ucb


def test_ucb_linear_in_std():
    mean = np.array([1.0, 0.0])
    std = np.array([0.1, 0.5])
    s = ucb(mean, std, kappa=2.0)
    assert np.isclose(s[0], 1.2) and np.isclose(s[1], 1.0)
    # uncertain candidate outranks the certain one at this kappa ordering
    assert np.isclose(s[1] - s[0], -0.2)


def test_ei_properties():
    mean = np.array([1.0, 0.0])
    std = np.array([0.1, 0.1])
    e = ei(mean, std, y_best=0.5, xi=0.01)
    assert np.all(e >= 0)
    assert e[0] > e[1]  # candidate above incumbent improves more
    # zero variance, below incumbent -> no improvement possible
    assert ei(np.array([0.0]), np.array([1e-8]), 0.5, 0.0)[0] == pytest.approx(0.0, abs=1e-6)


def test_acquire_dispatch_and_unknown():
    mean = np.array([0.5, 0.7])
    std = np.array([0.1, 0.1])
    assert np.array_equal(acquire(mean, std, "greedy"), mean)
    with pytest.raises(ValueError, match="unknown acquisition"):
        acquire(mean, std, "thompson-ish")
