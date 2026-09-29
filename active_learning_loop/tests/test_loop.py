import numpy as np
import pandas as pd
import pytest

from al_loop.encode import one_hot
from al_loop.loop import run_active, run_random

# small synthetic landscape: fitness = fraction of 'A' residues, so the
# oracle optimum is AAAA and a learned surrogate should find it
VARIANTS = pd.Series(
    [a + b + c + d for a in "AV" for b in "AV" for c in "AV" for d in "AV"]
)
Y = np.array([v.count("A") / 4.0 for v in VARIANTS])
X = one_hot(VARIANTS)

EXP = {"n_init": 4, "batch_size": 2, "budget": 10, "seed": 0}
SUR = {"kind": "rf"}
ACQ = {"kind": "ucb", "kappa": 1.0, "xi": 0.01}


def test_active_respects_budget_and_monotone_best():
    top = {int(np.argmax(Y))}
    records, picks = run_active(X, Y, EXP, SUR, ACQ, top)
    assert records[-1]["n_experiments"] == EXP["budget"]
    assert len(picks) == EXP["budget"]
    best = [r["best_fitness"] for r in records]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
    hits = [r["top_hits_found"] for r in records]
    assert all(h2 >= h1 for h1, h2 in zip(hits, hits[1:]))


def test_active_deterministic_same_seed():
    top = set()
    _, p1 = run_active(X, Y, EXP, SUR, ACQ, top)
    _, p2 = run_active(X, Y, EXP, SUR, ACQ, top)
    assert p1 == p2


def test_active_picks_unique():
    _, picks = run_active(X, Y, EXP, SUR, ACQ, set())
    assert len(set(picks)) == len(picks)


def test_random_same_schedule():
    recs = run_random(X, Y, EXP, seed=1, top_set=set())
    assert recs[-1]["n_experiments"] == EXP["budget"]
    assert len(recs) == (EXP["budget"] - EXP["n_init"]) // EXP["batch_size"] + 1


def test_active_finds_optimum_on_easy_landscape():
    # sanity that the loop learns: on this separable landscape it
    # should discover the global optimum within the budget
    top = {int(np.argmax(Y))}
    records, _ = run_active(X, Y, EXP, SUR, ACQ, top)
    assert records[-1]["top_hits_found"] == 1


def test_apply_transform():
    from al_loop.loop import apply_transform
    y = np.array([0.0, 1.0, np.e - 1])
    assert np.allclose(apply_transform(y, "log1p"), [0, np.log(2), 1])
    # AAV scores are negative-heavy; identity keeps them as-is
    neg = np.array([-11.2, -1.4, 9.5])
    assert np.array_equal(apply_transform(neg, "identity"), neg)
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transform(y, "bogus")
