# Project Guidance

- ProteinMPNN and ESM-2 are upstream models; this repo is the
  orchestration and evaluation layer. Credit them explicitly and never
  present their outputs as our models' work.
- Generated sequences are computational candidates only — no claim of
  stability, folding, or function without experimental or stronger
  computational validation (e.g. structure prediction of designs).
- Never present a single sampling temperature or one backbone as a
  general result; report settings and the spread across seeds/temps.
- Raw model outputs and downloaded structures stay out of git; commit
  only compact derived artifacts under `results/`.
- `config/config.yaml` is the single source of truth for backbone,
  model versions, sampling parameters, and scoring settings.
- Run `python -m pytest tests/` and `snakemake -n` after changes.
