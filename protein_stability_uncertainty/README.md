# protein-stability-uncertainty

Sequence -> protein **melting point** regression with calibrated
uncertainty, evaluated on a homology-aware held-out split, the protein
analog of scaffold-split evaluation. Thermal stability is a food/protein
engineering trait (processing robustness, formulation shelf-life). The deliverable
is a prediction plus a calibrated interval and an applicability domain,
the part that transfers to decision support.

**Status: working demonstration.** Snakemake DAG: fetch -> parse ->
ESM-2 embed -> ridge -> split-conformal + applicability-domain eval.

## Design

- **Dataset**: FLIP meltome `mixed` split (Jarzab et al., Nature Methods
  2020 thermal proteome atlas, via the FLIP mirror): 27,951 proteins,
  target = melting point °C. The split is precomputed from MMSeqs2 20%
  sequence-identity clusters, so held-out clusters share <=20% identity
  with anything in training. No random-split optimism.
- **Encoder under test**: ESM-2 (`esm2_t6_8M`, mean-pooled last hidden
  state, sequences truncated at 1024 aa, count logged). Baseline:
  amino-acid composition + log length. If ESM-2 can't beat composition
  features, the LM earns nothing here.
- **Model**: ridge regression. What is under test is the evaluation
  wrapper, not the regressor.
- **Uncertainty**: split-conformal. Ridge is fit on train-minus-
  validation. Absolute residuals are calibrated on FLIP's `validation`
  subset (still train-side). 90% target coverage is measured on held-out
  clusters. Calibration never touches test.
- **Applicability domain**: Euclidean distance in embedding space to the
  training centroid. Bin edges are fixed on the calibration split (test
  quartiles would leak), so unequal test-bin sizes reflect real domain
  shift.
- **Conditional coverage**: Mondrian conformal, a separate residual
  quantile per AD bin. Marginal coverage can hide a domain gradient.
  per-bin calibration recovers flat coverage by widening intervals
  where extrapolation is real. Sparse bins (<30 calibration points)
  fall back to the global quantile.

## Result

(committed in `results/summary.json`; homology-separated test clusters)

| Features | RMSE °C | MAE °C | Spearman | Conformal coverage (target 0.90) | Mean width °C |
|---|---:|---:|---:|---:|---:|
| composition + length | 9.62 | 7.49 | 0.32 | 0.888 | 30.1 |
| ESM-2 mean-pooled | **7.78** | **5.93** | **0.50** | 0.900 | 25.3 |

Sequence LM embeddings carry real thermostability signal: +0.18
Spearman and -1.6 °C MAE over composition (RMSE -1.8 °C), with narrower intervals.

Marginal coverage lands at 0.89-0.90, but it is not uniform across the
applicability domain. With bins fixed on calibration-distance quartiles
(ESM-2 view):

| AD bin (near -> far) | n_test | Marginal coverage | MAE °C |
|---|---:|---:|---:|
| nearest | 743 | 0.965 | 4.7 |
| | 714 | 0.916 | 5.6 |
| | 866 | 0.879 | 6.3 |
| farthest | 811 | 0.851 | 6.9 |

Predictions degrade smoothly with distance from the training manifold
(MAE +47%, coverage -11.4pp nearest to farthest). **Mondrian conformal
recovers flat per-bin coverage**: a separate residual quantile per bin:

| AD bin | q (half-width °C) | Coverage | MAE °C |
|---|---:|---:|---:|
| nearest | 9.9 | 0.898 | 4.7 |
| | 11.3 | 0.870 | 5.6 |
| | 14.0 | 0.913 | 6.3 |
| farthest | 14.5 | 0.908 | 6.9 |

The mechanism is visible in the half-widths: intervals widen from 9.9 °C
near the manifold to 14.5 °C at the edge, and the near bin's *coverage
drops* (0.965 -> 0.898) because its interval correctly shrinks. That is
the trade: same marginal coverage (~0.90), but now the guarantee
holds per-bin instead of pooling easy and hard points.

Composition features stay flat (~0.88 across all bins) under both
schemes. Its distance measure is uninformative, which is itself the
finding: the embedding-space AD is doing real diagnostic work.

## Caveats

- Melting points are assay measurements with real noise. The conformal
  intervals bound *model* error, not experimental reproducibility.
- Mixed-species corpus. The test split is homology-separated, not
  species-stratified, so cross-species distribution shift is present and
  not isolated as a factor.
- AD quartiles are descriptive (measured coverage per bucket), not a
  calibrated guarantee conditioned on the domain flag.
- Ridge on mean-pooled embeddings loses positional information.
  Per-residue or attention-pooled features are the known upgrade path.
- 2,890 sequences >1024 aa are truncated (count logged). Their measured
  melting points may reflect C-terminal or multi-domain behavior the
  truncated embedding cannot see.
- Marginal coverage at/near target does not imply per-domain coverage.
  Mondrian bins flatten the gradient but per-bin coverage is still a
  finite-sample estimate, not a strict conditional guarantee.
- AD bins are coarse quartiles, not a learned domain boundary. A
  distance threshold for abstention would need its own calibration.

## Run

```bash
.venv/bin/snakemake -j1          # fetch -> parse -> embed -> evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`PROTSTAB_CONFIG` selects an alternate config. `encoder.kind:
composition` reproduces the no-LM baseline as a standalone run.
Embeddings cache to `data/processed/*_esm2.npy` keyed by encoder config
and exact sequence list. A changed model or context silently invalidates
the cache instead of reusing it.

## Deployable artifact + Space

`results/deploy_esm2.json` is the fitted model as plain JSON (scaler +
ridge coefficients + centroid + per-bin conformal quantiles), so it is
diffable and loads without pickle. `spaces/protein-stability/` holds a
Gradio app (sequence -> Tm + per-bin interval + domain flag) that
consumes it. `spaces/protein-stability/push_space.sh` copies app +
artifact into a cloned HuggingFace Space repo.

## Data

FLIP `splits/meltome/splits.zip` (CC BY 4.0; meltome atlas per Jarzab et
al., Nature Methods 2020). Downloaded zip and cached embeddings are
gitignored. The compact summary and provenance are committed.

## Related work

- [comp-tox-pipeline](https://github.com/barlowa124/mol-ml/tree/main/comp_tox_pipeline) applies the same conformal + applicability-domain discipline to Tox21 classification.
