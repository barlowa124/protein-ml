"""Distributed + serving battery.

Shard-partition invariants hold without a process group; the 2-proc
torchrun run is exercised end-to-end in test_ddp_checkpoint_parity by
spawning the real entry point (skipped when torch is absent).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from protein_diffusion.ddpm import Denoiser, sample  # noqa: E402
from protein_diffusion.train_ddp import _rank_epoch_perm  # noqa: E402


class TestSharding:
    def test_union_covers_every_row(self):
        n, world = 997, 4
        for ep in (0, 5, 17):
            shards = [_rank_epoch_perm(n, ep, 7, r, world)
                      for r in range(world)]
            assert sorted(np.concatenate(shards).tolist()) == list(range(n))

    def test_shards_disjoint(self):
        n = 512
        for world in (2, 3, 8):
            seen = set()
            for r in range(world):
                s = set(_rank_epoch_perm(n, 0, 42, r, world).tolist())
                assert seen.isdisjoint(s)
                seen |= s

    def test_epochs_reshuffle(self):
        a = _rank_epoch_perm(256, 0, 7, 0, 2)
        b = _rank_epoch_perm(256, 1, 7, 0, 2)
        assert not np.array_equal(a, b)

    def test_deterministic_across_calls(self):
        a = _rank_epoch_perm(100, 3, 9, 1, 4)
        b = _rank_epoch_perm(100, 3, 9, 1, 4)
        assert np.array_equal(a, b)


@pytest.fixture(scope="module")
def ddp_checkpoint(tmp_path_factory):
    """Real 2-proc torchrun training on a toy landscape (~1 min)."""
    pytest.importorskip("pandas")
    import pandas as pd
    from protein_diffusion.encode import one_hot

    tmp = tmp_path_factory.mktemp("ddp")
    rng = np.random.default_rng(0)
    alpha = "ACDE"
    variants = ["".join(rng.choice(list(alpha), 4)) for _ in range(64)]
    df = pd.DataFrame({"variant": variants,
                       "fitness": rng.uniform(0, 3, 64)})
    pq = tmp / "toy.parquet"
    df.to_parquet(pq)

    model_p = tmp / "ddp.pt"
    rec_p = tmp / "rec.json"
    cfg = tmp / "cfg.yaml"
    cfg.write_text(
        "dataset:\n  transform: log1p\n  alphabet: ACDE\n"
        "model:\n  timesteps: 8\n  hidden: 32\n  lr: 0.001\n"
        "  epochs: 3\n  batch_size: 16\n  seed: 7\n"
        "generate:\n  cond_drop: 0.15\n  cond_fitness: 2.0\n"
        "evaluation:\n  top_k: 10\n")
    env = {**os.environ, "PYTHONPATH": "src",
           "DIFFUSION_CONFIG": str(cfg)}
    venv = Path(sys.executable).parent
    torchrun = venv / "torchrun"
    subprocess.run(
        [str(torchrun), "--nproc_per_node=2",
         "-m", "protein_diffusion.train_ddp",
         str(pq), str(model_p), str(rec_p)],
        env=env, check=True, capture_output=True, text=True, timeout=300)
    return model_p, rec_p, df


class TestDdpRun:
    def test_provenance_record(self, ddp_checkpoint):
        _, rec_p, _ = ddp_checkpoint
        rec = json.loads(rec_p.read_text())
        assert rec["ddp"]["world_size"] == 2
        assert rec["ddp"]["backend"] == "gloo"
        assert rec["inputs"][0]["sha256"]
        assert all(np.isfinite(v) for v in rec["ddp"]["rank0_losses_tail"])

    def test_checkpoint_loads_and_samples(self, ddp_checkpoint):
        model_p, _, df = ddp_checkpoint
        sd = torch.load(model_p, map_location="cpu", weights_only=True)
        x_dim = sd["net.0.weight"].shape[1] - 33
        m = Denoiser(x_dim, 32)
        m.load_state_dict(sd)
        out = sample(m, 4, x_dim, 8, 7)
        assert out.shape[0] == 4


class TestServe:
    @pytest.fixture(scope="class")
    def client(self, ddp_checkpoint):
        pytest.importorskip("fastapi")
        model_p, _, df = ddp_checkpoint
        pq = model_p.parent / "toy.parquet"
        saved = dict(os.environ)
        os.environ.update(MODEL_PATH=str(model_p), LANDSCAPE_PATH=str(pq),
                          TIMESTEPS="8", ALPHABET="ACDE")
        import protein_diffusion.serve as serve
        serve._ckpt = str(model_p)
        serve._load()
        os.environ.clear()
        os.environ.update(saved)
        from fastapi.testclient import TestClient
        return TestClient(serve.app)

    def test_health_and_model(self, client):
        assert client.get("/health").json()["loaded"] is True
        info = client.get("/model").json()
        assert info["x_dim"] == 4 * 4
        assert info["oracle_loaded"] is True

    def test_sample_measured_flag(self, client):
        r = client.post("/sample", json={"n": 8, "seed": 1}).json()
        assert r["n"] == 8
        for v in r["variants"]:
            assert set(v["variant"]) <= set("ACDE")
            assert v["status"] in ("measured", "unmeasured")
            assert (v["fitness"] is not None) == (v["status"] == "measured")

    def test_n_bounds(self, client):
        assert client.post("/sample", json={"n": 0}).status_code == 422
        assert client.post("/sample", json={"n": 9999}).status_code == 422

    def test_guidance_accepted(self, client):
        r = client.post("/sample", json={"n": 4, "seed": 2, "cond": 1.5,
                                       "guidance": 4.0})
        assert r.status_code == 200

    def test_metrics_endpoint(self, client):
        client.post("/sample", json={"n": 4, "seed": 3})
        body = client.get("/metrics").text
        assert "diffusion_requests_total" in body
        assert "diffusion_samples_total" in body
        assert "diffusion_measured_fraction" in body
        m = re.search(r"diffusion_requests_total (\d+)", body)
        assert int(m.group(1)) >= 1
