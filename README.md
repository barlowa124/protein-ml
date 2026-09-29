# protein-ml

[![ci](https://github.com/barlowa124/protein-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/barlowa124/protein-ml/actions/workflows/ci.yml)


Machine learning over measured protein fitness landscapes. Four related
projects merged into one repository, each a self-contained package with its
own tests, config, and commit history (imported via subtree merge).


## Where this sits in the portfolio

`protein-ml` is the **protein fitness ML** repo: supervised and generative models over measured fitness landscapes (GB1, AAV), active learning, ESM embeddings, and diffusion. Sibling repos:
[trust-tools](https://github.com/barlowa124/trust-tools) (agent security
and evals), [bio-qc](https://github.com/barlowa124/bio-qc) (lab-data QC
pipelines), [lab-informatics](https://github.com/barlowa124/lab-informatics)
(lab data plumbing and integrity),
[llm-posttraining](https://github.com/barlowa124/llm-posttraining)
(training-stage behavior work),
[protein-ml](https://github.com/barlowa124/protein-ml) (protein fitness
ML), and [mol-ml](https://github.com/barlowa124/mol-ml) (small-molecule
ML).

## Packages

| Directory | What it does |
|---|---|
| `protein_diffusion/` | Conditional DDPM over the measured GB1 fitness landscape. Reports memorization fraction and unmeasured-proposal handling as headline metrics. |
| `protein_stability_uncertainty/` | Sequence-to-melting-point regression with split-conformal intervals, Mondrian binning, and sparse-bin fallback. |
| `protein_design_ops/` | Orchestration/evaluation layer around ProteinMPNN + ESM-2: backbone parsing, sequence generation, spread-across-temperatures reporting. |
| `active_learning_loop/` | GP-UCB active learning over real fitness landscapes, with seeded random-baseline replication. |

## Running tests

Each package is independent. From its directory:

```bash
cd protein_diffusion && PYTHONPATH=src python -m pytest tests/ -q
```

Same pattern for the other three. Each subdirectory retains its own
`AGENTS.md` with project-specific rules (data policy, claim standards,
verification commands), which still apply.

## Why one repo

These four share datasets (FLIP / eLife GB1 mirrors), encoding utilities,
provenance manifests, and evaluation conventions. Merging them makes the
shared machinery visible in one place. A root-level parity test fails if
the vendored `esm_cache.py` copies drift apart.
