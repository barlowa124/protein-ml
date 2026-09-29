"""Distributed (DDP) training entry point.

Run under torchrun for a real multi-process run:

    torchrun --nproc_per_node=2 -m protein_diffusion.train_ddp \
        data/processed/gb1.parquet data/processed/ddpm.pt \
        results/train_ddp.json

Backend is read from DIFFUSION_DDP_BACKEND (default "gloo"). On GPU
clusters use "nccl"; CPU gloo is a genuine distributed run, not a stub —
two OS processes exchange gradients each step. Single-rank fallback
(--nproc_per_node=1) still exercises the same code path.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from protein_diffusion.config import load_config
from protein_diffusion.ddpm import (Denoiser, make_schedule,
                                    q_sample)
from protein_diffusion.encode import one_hot
from protein_diffusion.provenance import git_commit, sha256_file
from protein_diffusion.run import _cond_transform


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _rank_epoch_perm(n: int, epoch: int, seed: int, rank: int,
                     world: int) -> np.ndarray:
    """Deterministic per-epoch shuffle, interleaved sharded by rank.
    Rank r sees perm[r::world], so every row is owned by exactly one
    rank each epoch and the union is a full permutation."""
    rng = np.random.default_rng(seed * 100003 + epoch)
    perm = rng.permutation(n)
    return perm[rank::world]


def train_ddp(X: np.ndarray, y_cond: np.ndarray, timesteps: int,
              hidden: int, lr: float, epochs: int, batch_size: int,
              seed: int, cond_drop: float) -> tuple[Denoiser, dict]:
    """DDP training loop; returns the synced model plus a run record."""
    rank, world = _env_int("RANK", 0), _env_int("WORLD_SIZE", 1)
    backend = os.environ.get("DIFFUSION_DDP_BACKEND", "gloo")
    dist.init_process_group(backend)
    torch.manual_seed(seed)

    betas, _, alpha_bar = make_schedule(timesteps)
    x0 = torch.tensor(X, dtype=torch.float32)
    c0 = torch.tensor(y_cond, dtype=torch.float32).reshape(-1, 1)

    raw = Denoiser(x0.shape[1], hidden)
    model = DistributedDataParallel(raw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(x0)
    losses = []
    t0 = time.time()
    for ep in range(epochs):
        shard = _rank_epoch_perm(n, ep, seed, rank, world)
        total, seen = 0.0, 0
        for i in range(0, len(shard), batch_size):
            idx = shard[i:i + batch_size]
            xb, cb = x0[idx], c0[idx].clone()
            if cond_drop:
                drop = torch.rand(len(cb)) < cond_drop
                cb[drop] = -1.0
            t = torch.randint(0, timesteps, (len(xb),))
            xt, eps = q_sample(xb, t, alpha_bar)
            loss = nn.functional.mse_loss(model(xt, t, cb), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        losses.append(total / max(seen, 1))
        if rank == 0 and (ep % 10 == 9 or ep == 0):
            print(f"epoch {ep + 1}/{epochs} rank-local mse {total/seen:.4f}")
    # sync so every rank leaves with identical weights
    for p in raw.parameters():
        dist.broadcast(p.data, src=0)
    dist.destroy_process_group()
    model.module.eval()
    record = {
        "backend": backend, "world_size": world, "rank": rank,
        "epochs": epochs, "batch_size": batch_size,
        "rank0_losses_tail": losses[-5:],
        "wall_seconds": round(time.time() - t0, 1),
    }
    return model.module, record


def main() -> None:
    in_parquet, out_model, out_json = sys.argv[1:4]
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    ds = cfg["dataset"]
    X = one_hot(df["variant"], ds.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"))
    y_cond, _ = _cond_transform(df["fitness"].to_numpy(),
                                ds.get("transform", "log1p"))
    m = cfg["model"]
    model, record = train_ddp(
        X, y_cond, m["timesteps"], m["hidden"], m["lr"], m["epochs"],
        m["batch_size"], m["seed"],
        cfg["generate"].get("cond_drop", 0.15))
    if _env_int("RANK", 0) == 0:
        Path(out_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_model)
        Path(out_json).write_text(json.dumps({
            "git": git_commit(),
            "versions": {"torch": torch.__version__,
                         "numpy": np.__version__},
            "inputs": [{"path": in_parquet,
                        "sha256": sha256_file(in_parquet)}],
            "ddp": record}, indent=1))
        print(f"wrote {out_model} (world_size={record['world_size']})")


if __name__ == "__main__":
    main()
