# Project Guidance

- Use only public datasets (FLIP / eLife GB1 mirror). Never commit raw
  downloaded archives; commit only compact derived artifacts under
  `results/`.
- Report active-vs-random comparison honestly: random baseline is
  replicated over seeds; a single seeded AL trajectory is one draw — say so.
- Do not claim the simulation predicts wet-lab outcomes; the oracle is
  measured data, which removes assay noise a real campaign would face.
- Keep `config/config.yaml` the single source of truth for budget, batch,
  surrogate, acquisition, and evaluation settings.
- Run `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` and
  `.venv/bin/snakemake -n` after changes. Snakemake does not track source
  edits — use `-F` to force reruns of `run`/`report`.
