# Project Guidance

- Use only public datasets (FLIP / eLife GB1 mirror). Never commit raw
  downloaded archives or model checkpoints; commit only compact metrics
  under `results/`.
- Generated-variant fitness claims must come from the measured oracle, not
  a surrogate model scoring its own outputs.
- Report memorization honestly: fraction of generated samples present in
  the training set is a headline metric, not a footnote.
- Unmeasured proposals (variants outside the landscape) count as zero
  fitness, not dropped rows.
- Keep `config/config.yaml` the single source of truth for schedule,
  model, threshold, and evaluation settings.
- Run `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` and
  `.venv/bin/snakemake -n` after changes. Snakemake does not track source
  edits — use `-F` to force reruns.
