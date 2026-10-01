# Project Guidance

- ProteinMPNN, ESM-2, ESMFold, and Chai-1 are upstream models; this repo
  is the orchestration and evaluation layer. Credit them explicitly and
  never present their outputs as our models' work. Keep the fold backend
  labeled (fold_backend / provenance.structure_predictor) so ESMFold
  numbers are never reported as Chai-1 numbers or vice versa.
- Chai-1 runs in a separate environment (chai_lab requires torch<2.7)
  and is invoked as a subprocess via config.chai.command. Do not install
  chai_lab into this environment.
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
