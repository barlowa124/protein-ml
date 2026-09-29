# active-learning-loop

Simulated active-learning experiment selection over a **real, fully measured
protein fitness landscape**: does a surrogate-model-driven loop find
high-fitness variants with fewer experiments than random screening?

**Status: working demonstration, replicated on two landscapes.** Snakemake
DAG runs fetch -> parse -> AL-vs-random simulation -> summary report on:

- **GB1** four-site combinatorial landscape (Wu et al., eLife 2016, via the
  FLIP mirror): 149,361 variants at sites V39/D40/G41/V54 with
  experimentally measured enrichment fitness.
- **AAV2** capsid viability landscape (Ogden et al., Science 2019, FLIP
  mirror): 38,293 substitution variants over a 28-aa region, log-scale
  viability score, a larger partially epistatic landscape that
  includes stop-codon dead variants (`*`).

## The question

The loop experimental science wants closed: label a small
random screen, fit a surrogate, let an acquisition function pick the next
most informative batch, repeat under a fixed budget. The landscape is fully
measured, so the oracle is ground truth. Every acquisition decision is
scored against what the experiment would have returned.

## Method

- **Surrogate**: Gaussian process (RBF + white noise, fixed hyperparameters,
  normalized targets) over position-wise one-hot features. Alphabet and
  variant regex are per-dataset config (`GB1: 4x20`, `AAV: 28x21 incl. *`).
- **Target transform** per dataset: GB1 fits on `log1p(fitness)`,
  since enrichment is heavy-tailed (mean 0.08, max 8.76). AAV's log-viability
  score is already symmetric -> `identity`.
- **Acquisition**: UCB (`mean + kappa*std`, kappa=2). EI and greedy are
  implemented behind config.
- **Schedule**: 96-variant random initial screen, then batches of 24 up to a
  480-experiment budget.
- **Baseline**: random selection on the identical schedule, replicated over
  20 seeds. Same initial distribution, same budget. The matched counterfactual.
- **Metrics**: best-fitness-found curve, true top-100 discovery curve, AUBC
  (area under best curve, oracle-normalized), top-100 hit rate at budget.

## Result (committed in `results/summary.json`)

Active policy replicated over **8 seeds** against a 20-seed random baseline,
same budget, same schedule, both distributions reported:

| | Active (UCB-GP, 8 seeds) | Random (20 seeds) |
|---|---|---|
| AUBC | **0.672 ± 0.100** | 0.373 ± 0.107 |
| best fitness found, mean | **8.24** (oracle max 8.76) | 5.06 |
| true top-100 hits at budget, mean | **43.4** | 0.55 |
| acquired variants with fitness > 1.0 | 81% (seed-13 trajectory) | ~4% of landscape |

The policy **concentrates experiments on the functional
region**, with ~79x more true top-100 hits than random, and 6 of 8 trajectories
find the oracle-best variant within budget. The weakest seed still
beats the random mean on AUBC, but trajectory variance is real (0.465..0.772)
and the bands overlap at the low end. A single AL run is not a guarantee.
Per-trajectory metrics are in `results/summary.json` under
`per_trajectory`.

## Replication: AAV2 capsid viability (`results/summary_aav.json`)

Same code path, same experiment schedule, second landscape. The dataset
descriptor is the only thing that changes (`config/config_aav.yaml`).

| | Active (UCB-GP, 8 seeds) | Random (20 seeds) |
|---|---|---|
| AUBC | **0.604 ± 0.034** | 0.524 ± 0.076 |
| best fitness found, mean | **7.69** (oracle max 9.54) | 6.64 |
| true top-100 hits at budget, mean | **11.9** | 1.5 |

The advantage **shrinks** on AAV. Top-100 hit
enrichment stays strong (~8x), and every active trajectory's AUBC beats
the random *mean*, but the bands overlap. The worst active
trajectory (0.543) is below the best random one (0.646). The same
overlap exists on GB1 and is slightly wider there (0.465 vs 0.599,
gap 0.134). Expected reasons: AAV is 38k variants vs 149k with a
higher base rate of functional variants (~47% score > 0), so random
screening catches more. The 588-dim one-hot over a rougher landscape is
a harder GP regression than GB1's 80-dim near-orthogonal space. The
replication is the point. A portfolio AL demo that only works on one
friendly landscape isn't evidence of anything.

## Encoder ablation: ESM-2 vs one-hot on GB1 (`results/summary_gb1_esm2.json`)

Same landscape, same schedule. Only the feature space changes. Variants
are embedded by ESM-2 (`esm2_t6_8M`, mean-pooled) after substituting into
the WT GB1 sequence at sites 38/39/40/53, since a bare 4-AA string carries no
signal for a protein LM. Requires `.[esm]` extras.

| | one-hot + GP (8 seeds) | ESM-2 + GP (8 seeds) | Random (20 seeds) |
|---|---|---|---|
| AUBC | 0.672 ± 0.100 | 0.621 ± **0.040** | 0.373 ± 0.107 |
| best fitness found, mean | 8.24 | **8.31** | 5.06 |
| top-100 hits at budget, mean | **43.4** | 39.2 | 0.55 |

ESM-2 does **not** beat one-hot on GB1. A 4-site combinatorial library is already fully
specified by one-hot (every factor the GP needs is a measured coordinate),
while mean-pooled embeddings of sequences differing in 4 of 56 residues
are nearly isotropic (median pairwise distance 0.51; kernel scale was set
to 0.4 from that diagnostic, not tuned on results). ESM-2 does improve
cross-seed consistency: AUBC spread tightens 2.5x in std (0.040 vs
0.100) and best-found is marginally higher. Embeddings would be the right
encoder for landscapes spanning variable regions or requiring
generalization beyond measured combinations. Here they trade a little
peak-seeking for a lot of stability.

The same ablation on AAV is where embeddings *should* have an
edge. The 28-aa region is sequence-diverse, ESM-2 distance is
decorrelated from Hamming (~0 spearman), and median pairwise distance
is 1.19 (vs 0.51 on GB1):

| AAV | one-hot + GP | ESM-2 + GP | Random |
|---|---|---|---|
| AUBC | **0.604 ± 0.034** | 0.519 ± 0.029 | 0.524 ± 0.076 |
| best fitness, mean | 7.69 | 6.88 | 6.64 |
| top-100 hits, mean | **11.9** | 4.0 | 1.5 |

ESM-2 on AAV performs at the random baseline. The AL advantage
disappears entirely. Mechanism: GP-UCB works through metric structure,
and on these landscapes *Hamming distance is the informative metric*:
fitness correlates with mutation count/composition. Mean-pooled ESM-2
embeddings smooth over that structure by design (the same pooling that makes them generalize for
property prediction makes them metrically useless for nearest-neighbor-ish landscape exploitation).
One-hot is the right encoder for oracle-evaluated combinatorial AL.
Embeddings suit tasks needing transfer across
proteins or unmeasured regions, which an all-measured oracle cannot
test. Both ablations committed: `summary_gb1_esm2.json`,
`summary_aav_esm2.json`.

## Debugging trail

The first run reported active *worse* than random (AUBC 0.167, zero top-100
hits, best found frozen at the initial draw). Traced to two failures:

1. Per-round GP kernel optimization degenerated on the spiky landscape
   (bounds-hitting, matmul overflow) -> NaN acquisition scores -> argsort
   silently degenerated to index order, repeatedly picking the first
   unlabeled rows.
2. `length_scale=10` was mismatched to the one-hot metric (variant
   distances are sqrt(2)-sqrt(8)), making candidates nearly indistinguishable.

Fix: fixed hyperparameters matched to the feature metric + log1p target
transform. Silent-NaN acquisition produces *plausible-looking* wrong
results, and the only tell here was the frozen best-fitness curve.

## Caveats

- Oracle = noise-free measured values. Real experiments add assay noise the
  surrogate would have to absorb.
- One-hot encoding sees sites, not structure. Sequence-embedding encoders
  (e.g., ESM-2) are the natural upgrade behind the same interface.
- Greedy top-k batch selection ignores batch diversity.

## Run

```bash
.venv/bin/snakemake -j1          # full DAG
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`AL_CONFIG` env var selects an alternate config. `config/config_aav.yaml`
is the AAV descriptor used by the `*_aav` Snakefile rules. Results:
`results/summary{,_aav}.json`, `results/curves{,_aav}.png`,
`results/provenance{,_aav}.json`. Per-round records and the exact
acquisition order the policy chose are in `data/processed/` (regenerable,
gitignored).

## Data

FLIP `splits/gb1/four_mutations_full_data.csv.zip` (CC BY 4.0; extends Wu et
al., eLife 2016 supplement) and `splits/aav/full_data.csv.zip` (Ogden et
al., Science 2019). AAV parsing keeps the 28-aa substitution subset of the
28-aa mutated region including `*` stops. Indel/other-length rows are
dropped with a logged count (245,716 of 284,009; the dropped rows are
structural variants outside the fixed-width substitution landscape this
encoder covers). Downloaded zips are gitignored. Parsed parquets are
regenerable intermediates.

## Related work

- [protein-diffusion](https://github.com/barlowa124/protein-diffusion) is scored against the same GB1 measured oracle. The two results are directly comparable.
- [dti-fusion](https://github.com/barlowa124/dti-fusion) and [protein-design-ops](https://github.com/barlowa124/protein-design-ops) use the same ESM-2 embeddings that lose to one-hot here. The negative result is committed in `results/`.
