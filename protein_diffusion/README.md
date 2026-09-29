# protein-diffusion

A **conditional DDPM** trained on the full measured GB1 fitness landscape
and steered toward high fitness with classifier-free guidance. Generative
modeling where "did it work" is answerable from the measured oracle, not
vibes.

**Status: working demonstration on GB1, documented transfer failure on
AAV.** Snakemake DAG runs fetch -> train -> guided sampling -> oracle
evaluation on the GB1 four-site combinatorial landscape (Wu et al., eLife
2016, via the FLIP mirror): 149,361 variants at sites V39/D40/G41/V54 with
experimentally measured enrichment fitness. A second-landscape attempt on
AAV2 capsid viability (Ogden et al., Science 2019) fails for two
independent, measured reasons (see "Second landscape").

## Design

- **Data space**: continuous relaxation of the 4-site one-hot tensor
  (4 x 20 = 80 dims). Generation decodes by per-site argmax, so every sample
  is a valid variant string by construction.
- **Model**: epsilon-prediction DDPM: linear beta schedule (T=300), 2-layer
  MLP denoiser (hidden 256), ancestral sampling. The denoiser takes a
  fitness condition channel (log1p-scaled). 15% condition dropout during
  training enables classifier-free guidance:
  `eps = eps_uncond + w * (eps_cond - eps_uncond)`.
- **Training set**: the full 149k measured landscape, including the dead
  variants. The v1 fit-only training set is why v1 failed. A model
  that never sees dead variants cannot learn where the boundary is.
- **Evaluation is against the oracle**: generated variants are looked up
  in the *measured* landscape, real experimental fitness, not a surrogate
  scoring its own outputs.

## Result (8 sampling seeds per diffusion mode, 20 per random baseline, committed in `results/summary.json`)

| Sampling mode | mean fitness | >= 0.5 | >= 1.0 | top-100 hits | unique/512 |
|---|---|---|---|---|---|
| unconditioned | 0.061 | 2.7% | 1.7% | 0.3 | 510 |
| conditioned (w=0) | 0.163 | 7.5% | 5.0% | 1.3 | 510 |
| **guided w=4** | **1.04** | **47.2%** | 36.1% | 7.6 | 462 |
| **guided w=8** | **1.69** | **80.4%** | **63.2%** | 6.6 | 238 |
| mutate parent, mu=1 | 1.28 | 45.3% | 39.1% | **16.8** | **500** |
| mutate parent, mu=2 | 0.81 | 30.3% | 25.5% | 8.5 | 507 |
| random draws | 0.081 | 4.1% | 2.5% | 0.3 | — |

The guidance dose-response is the demonstration: conditioning alone shifts
the distribution modestly (7.5% fit), and cranking the CFG weight steers
hard into the functional region. 80% of proposals measure >= 0.5 vs 4%
at random, ~20x enrichment. The tradeoff is in the last columns:
strong guidance concentrates samples onto fewer modes (unique variants
510 -> 237, and top-100 hits dip from 7.6 to 6.6 as the sampler
collapses onto "good enough" modes instead of the highest-fitness ones). Diversity
vs fitness is the classic CFG tradeoff, measured here instead of assumed.

## The mutational baseline

`mutate_*` rows are the trivial experimental baseline a diffusion model
has to justify itself against: draw a parent uniformly from the >=3.0
fitness pool, apply `max(1, Poisson(mu))` random substitutions, score the
children. The verdict is mixed:

- The earlier deconstruction ("Hamming-1 neighbors of fit variants are
  dead, mean 0.087") held for neighbors of *marginally* fit rows (>=0.5).
  The **elite peak is locally smoother**: children of >=3.0 parents
  measure 45% >= 0.5 at mu=1, so mutation is not useless here.
- Guided diffusion still wins on **bulk enrichment**: 80% vs 45% fit rate,
  1.69 vs 1.28 mean fitness. CFG concentrates mass better than random
  mutagenesis.
- On **discovery metrics the trivial baseline wins**: 500 unique
  variants and ~17 top-100 hits per 512 samples vs the DDPM's 237 / 6.6.
  CFG buys hit-rate by spending diversity, and this landscape's elite
  neighborhood is connected enough that mutagenesis rides it.

The claim narrows accordingly: the DDPM demonstrates *conditional
steering at superior hit-rate*, not superiority over all baselines at
every objective.

## Why v1 failed, and what fixed it

(Values in this section were computed against the v1 run at the time; the
superseded checkpoint is not retained in git, so they stay prose. The AAV
diagnostics below are recomputed and committed in
`results/diagnostics_aav.json`.)

The first version trained an *unconditional* DDPM on only the ~5.8k
fitness >= 0.5 variants. It produced 36% fit samples, but a memorization
audit (after fixing an indexing bug that undercounted it as 8.8%) showed
**46% of samples were literal training rows, and novel variants scored at
landscape-random fitness (0.082 vs 0.081, 0% >= 1.0).**

The measured deconstruction:

- Samples were not bit-copies pre-decode (mean continuous distance to the
  nearest training point ~2.1 for both memorized and novel), and memorization
  happened at argmax: samples were blurs snapping to the nearest vertex.
- 90% of novel outputs were Hamming-1 neighbors of training rows. But
  GB1's functional region is an archipelago. Hamming-1 neighbors of fit
  variants measure 0.087 mean and **0% are >= 0.5**. Near-copy sampling is
  worthless on a landscape this sharp.
- Conclusion: an unconditional density model over a sparse fit set cannot
  generalize, since there is no smooth manifold to interpolate along. The fix
  is the *conditioning signal* (fitness, over the full landscape including
  dead variants) and the *mechanism* (guidance) to spend probability mass
  where it counts.

## Second landscape: AAV2, and why it fails twice

Same code, `config/config_aav.yaml`: 28-aa `mutated_region`, alphabet
extended with `*` stop variants (kept, since they are real dead variants),
`shift_log1p` conditioning for the negative log-viability scores
(`results/summary_aav.json`).

**Failure 1: the oracle is vacuous.** GB1 is 93% measured over its
4-site space, so decoded variants almost always have ground truth. AAV
is 38k designed variants inside a ~21^28 region: **100% of generated
samples are unmeasured**. No decoded string coincides with a measured
row, so oracle fitness is undefined for them. The GB1
"93% measured" caveat, inverted. AAV shows the caveat was
load-bearing.

**Failure 2: the model doesn't even learn the library.** Diagnostics
(`results/diagnostics_aav.json`): the library is dense (median pairwise
Hamming 7, nearest-member distance 2) and per-site conserved (median
site entropy 0.87 nats vs 3.0 uniform). Yet generated strings sit ~22
substitutions from every measured variant, near random-string distance,
and match the library's modal residue at only 9% of sites (7% guided;
library members: 83.5%). Scaling the denoiser (512 hidden,
40 epochs) improves the match to 29% and MSE 0.64→0.29, so undertraining is
part of it, but the samples remain far off-manifold. Where GB1's
memorization failure produced *plausible-looking* outputs, AAV produces
visible noise. Neither is a working generator, for different reasons.

What a working version would need: a decoder-aware sampler (projected /
discrete diffusion) or a learned-fitness proxy for eval, each with its
own circularity caveats. Documented, not implemented.

## Caveats

- Four-site combinatorial space is small and discrete, and 93% of it is
  measured, so "novel" is nearly unreachable by construction. The claim
  demonstrated is *conditional steering*, not de novo discovery. On a real
  protein the same machinery would need a novelty channel to be useful.
- Diversity collapse at high guidance is real and reported. A proposal
  engine would sweep w to trade hit-rate against diversity.
- DDPM in continuous one-hot space is a modeling convenience. Discrete
  diffusion (D3PM-style) over residues is the principled formulation.
- A simpler proposal distribution (sampling the empirical high-fitness
  pool directly) would trivially produce fit variants. The diffusion
  model is justified only when conditioning needs to generalize, which
  this landscape cannot test.

## Run

```bash
.venv/bin/snakemake -j1          # fetch -> train -> sample -> evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`DIFFUSION_CONFIG` env var selects an alternate config. Outputs:
`results/summary.json` and `results/provenance.json`. The model checkpoint and
intermediates live in `data/processed/` (regenerable, gitignored).

## Scale-out and serving

Distributed training runs through `train_ddp` under `torchrun`:

```bash
.venv/bin/snakemake train_ddp -j1   # torchrun --nproc_per_node=2
```

This is a real two-process run: gloo backend on CPU, gradient all-reduce
every step, deterministic per-epoch index sharding (rank `r` gets
`perm[r::world]`, so the union of shards is a full permutation), rank 0
saves the checkpoint and writes `results/train_ddp.json` (world size,
backend, input hash, loss tail). The committed run trained in 9 s and the
resulting checkpoint is oracle-evaluated through the identical scoring
path (`eval_ddp.py` -> `results/eval_ddp.json`, 8 seeds): guided w8 mean
fitness 1.94 vs single-process 1.69, >=0.5 rate 77.8% vs 80.4%, >=1.0
63.9% vs 63.2%, the same band. It is more diverse (318 vs 238
unique/512) and lands more top-100 hits (20.6 vs 6.6), consistent with
rank-sharded minibatching acting as a different sampler trajectory, not
a defect. Sharding differs from the single-process shuffle, so runs are
not bit-identical. The band, not the exact trajectory, is the claim.
`DIFFUSION_DDP_BACKEND=nccl` is the path
for GPU clusters. NCCL and multi-node are untested here (no GPU on the
authoring machine).

Sampling is served by `serve.py`:

```bash
MODEL_PATH=data/processed/ddpm.pt \
  .venv/bin/uvicorn protein_diffusion.serve:app --port 8000
```

`POST /sample` takes `n`, `seed`, `cond`, `guidance` and returns variants
flagged `measured` when they exist in the landscape parquet (oracle
fitness included) or `unmeasured` otherwise. `GET /metrics` exposes
Prometheus-format counters (requests, samples, measured/unmeasured
totals, measured fraction). `web/` is a React + TypeScript variant
browser against `/sample` (committed screenshot shows a real 32-variant
run at w=4: 100% measured, mean 1.221, best 4.629). `docker/Dockerfile` builds
the service image. `deploy/k8s.yaml` is a manifest skeleton
(Deployment + Service + probes) and `deploy/terraform/main.tf` applies
the same contract via the kubernetes provider
(image/replicas/namespace parameterized). Both are provided, not
deployed: no cluster was available at authoring time. `deploy/sbatch.sh`
is the matching Slurm job spec (2-task torchrun, gloo, checkpoint eval).
Also authored-not-run since no Slurm scheduler was available. A GitHub
Actions workflow (`.github/workflows/ci.yml`) runs the test suite and a
`snakemake -n` dry-run on push.

Trained weights are mirrored on HuggingFace at
[barlowa/protein-ddpm-landscapes](https://huggingface.co/barlowa/protein-ddpm-landscapes)
(`gb1/` is the fixed conditional model, `gb1-ddp/` is the 2-process DDP
checkpoint, `aav2/` is the documented underfit failure, kept on purpose).

## Cross-framework parity (JAX/Flax port)

`jax_model.py` reimplements the denoiser in Flax and loads the committed
torch checkpoint weight-for-weight. `eval_jax.py` then proves the port
two ways, both committed in `results/parity_jax.json`:

- **Numerical**: epsilon predictions match torch to 3.1e-6 max abs dev,
  and a full 300-step ancestral trajectory driven by *shared* noise ends
  2.1e-6 apart. That is fp32 reduction-order scale, not framework drift. The
  RNG streams differ by design, so the claim is trajectory-level parity
  under identical noise, not seed-identical sampling.
- **Oracle**: JAX-sampled guided-w8 variants score through the identical
  `_eval_batch` path: mean fitness 1.70 vs torch's 1.69, 80.7% vs 80.4%
  at >=0.5, 0% unmeasured. Same model, same landscape, same band.

## Data

FLIP `splits/gb1/four_mutations_full_data.csv.zip` (CC BY 4.0, extending Wu et
al., eLife 2016 supplement). Downloaded zip is gitignored. The parsed
parquet is a regenerable intermediate.

## Related work

- [active-learning-loop](https://github.com/barlowa124/active-learning-loop) uses the same GB1 measured landscape as its acquisition oracle, so the enrichment and generation-steering numbers are comparable across the two repos.
