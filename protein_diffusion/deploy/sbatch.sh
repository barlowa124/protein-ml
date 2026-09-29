#!/bin/bash
# Slurm batch script for the 2-process DDP train + eval.
# Mirrors what `snakemake train_ddp` runs locally under torchrun.
# Authored, not executed: no Slurm cluster was available at authoring
# time, so this file is evidence of the job spec, not a run record.
#
# Submit with: sbatch deploy/sbatch.sh
#SBATCH --job-name=protein-diffusion-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddp_%j.out
#SBATCH --error=logs/ddp_%j.err

set -euo pipefail
mkdir -p logs

# gloo backend: this model trains on CPU locally. On a GPU partition,
# switch to --export=ALL,NCCL_SOCKET_IFNAME=... + backend nccl.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29600
export PYTHONPATH=src

# Same invocation the Snakefile uses locally: positional args are the
# input parquet, output checkpoint, and run record.
.venv/bin/srun --ntasks=2 --cpu-bind=cores \
  .venv/bin/torchrun --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m protein_diffusion.train_ddp \
    data/processed/gb1.parquet \
    data/processed/ddpm_ddp.pt \
    results/train_ddp.json

# Checkpoint eval through the same oracle path as the local run.
.venv/bin/python -m protein_diffusion.eval_ddp \
  data/processed/gb1.parquet \
  data/processed/ddpm_ddp.pt \
  results/eval_ddp.json
