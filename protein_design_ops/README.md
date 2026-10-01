# protein-design-ops

Backbone-conditioned sequence design with **independent rescoring**: given a
protein backbone, ProteinMPNN generates candidate sequences, then ESM-2
scores each candidate's zero-shot sequence fitness, a second model's
opinion on the first model's output. Candidates both models agree on are
the defensible shortlist. The report shows where the models disagree.

## Pipeline

```
backbone -> generate (ProteinMPNN) -> score (ESM-2)
    -> fold (ESMFold or Chai-1) -> selfconsistency -> report
```

- `backbone.py` - parse PDB, extract design chain + native sequence +
  CA coordinates
- `generate.py` - subprocess driver for upstream ProteinMPNN (external
  clone, not vendored, see `config.mpnn.repo_path`), parses FASTA headers
  carrying MPNN's own score and sequence recovery
- `score.py` - ESM-2 mean pseudo-log-likelihood per candidate (mask each
  position, log-prob of the residue at that position, average)
- `fold.py` - ESMFold structure screen: per-design pLDDT/PTM confidence
  plus the native for reference (~2.4 GB weights on first run, ~2 min/seq
  on CPU; skip with `snakemake report` after removing the fold input)
- `chai.py` - alternative fold backend. Drives Chai-1 as an external
  subprocess (`config.chai.command`), collects aggregate score,
  pTM/ipTM, pLDDT, and the predicted .cif per design
- `selfconsistency.py` - Kabsch CA RMSD of each predicted structure
  against the input backbone. A design that only scores well in
  sequence space but refolds somewhere else stays visible in the report
- `report.py` - consensus ranking, score correlation, diversity,
  identity-to-WT, fold-confidence + backbone-RMSD screens, scatter figure

### Fold backends

`config.fold.backend` selects the structure step. `esmfold` runs
in-process (the default). `chai1` invokes `chai-lab fold` in a separate
environment because chai_lab pins `torch<2.7`, which conflicts with the
torch this environment already runs. Point `config.chai.command` at that
env's binary and pass flags via `config.chai.extra_args`. Chai-1
downloads ~4 GB of weights on first use and is CPU-tolerable only at
reduced diffusion settings (e.g. `--num-diffn-timesteps 50
--num-diffn-samples 1`). Both backends write the same record schema, so
`selfconsistency` and `report` are backend-agnostic. The report's
`provenance.structure_predictor` records which one ran.

## Quickstart

```bash
# from the protein-ml repo root, so mpnn.repo_path resolves:
git clone https://github.com/dauparas/ProteinMPNN ../proteinmpnn-ext
cd protein_design_ops
uv venv --python 3.11 && uv pip install -e ".[dev]"
.venv/bin/python -m snakemake --cores 2
```

Demo backbone: trp-cage miniprotein (1L2Y, 20 aa). The whole DAG runs in
seconds on CPU, including ESM-2 scoring. A second config
(`config/config_1ubq.yaml`, ubiquitin, 76 aa natural fold) runs with
`DESIGN_CONFIG=config/config_1ubq.yaml snakemake`. Outputs are keyed by
`backbone.id` under `results/<id>/`.

## Observed results (16 candidates @ T=0.1 each)

| Backbone | Fold | Spearman(MPNN, ESM-2) | Mean recovery | Native ESM-2 | Designed ESM-2 mean |
|---|---|---:|---:|---:|---:|
| 1L2Y trp-cage (20 aa) | engineered miniprotein | **-0.632** | 0.38 | -2.87 | -2.59 |
| 1UBQ ubiquitin (76 aa) | natural globular | **-0.529** | 0.55 | -2.32 | -1.92 |

- On both backbones the two models **anticorrelate**, so the consensus shortlist is a
  compromise, not the best on either metric. This is reported as-is.
  Challenging single-model confidence is why the second scorer exists.
- Designed sequences out-score the native on ESM-2 for both backbones.
  For the natural fold the gap is wider (-1.92 vs -2.32). Plausible for
  fixed-backbone redesign. It is a sequence-fitness observation, not a
  folding or function claim.

## Structure screen (ESMFold, measured on both backbones)

All designs plus natives were folded with ESMFold v1. pLDDT is
ESMFold's own per-residue confidence, not an experimental structure.

| Backbone | native pLDDT | designed mean | confident (>=70) | designed pTM |
|---|---:|---:|---:|---:|
| 1L2Y trp-cage (20 aa) | 76.9 | 73.6 | 13/16 | 0.09-0.12 |
| 1UBQ ubiquitin (76 aa) | 77.4 | 80.6 | 16/16 | 0.83-0.87 |

The top-2 1L2Y consensus picks fold at 76.2 and 74.2, in the native
band (76.9) but not above it. Consensus #3 is the weakest folder in the batch (68.7),
so the three-way agreement is a real filter, not a formality. On 1UBQ
every design sits in the confident band and pTM separates cleanly
from the 1L2Y floor: 0.83-0.87 on the 76-mer vs 0.09-0.12 on the
20-mer. The pTM flat-line on trp-cage is a length artifact (pTM needs
longer chains to discriminate), now demonstrated on both lengths
instead of asserted.

`design_report.json` carries a `provenance` block: backbone id,
ProteinMPNN upstream commit + weights, sampling params/seed, and the
ESM-2 model id, enough to reproduce a run exactly.

**Reproducibility caveat:** ProteinMPNN's `--seed`
treats `0` as "pick a random seed" (the upstream script does
`if args.seed:`, and 0 is falsy). An earlier config set `seed: 0` believing
it pinned sampling. The runs above use `seed: 37` and are deterministic
(bit-identical candidates across repeated runs). The earlier ubiquitin
correlation near zero was a random-seed artifact, not a real
backbone-dependence signal. With sampling pinned, both backbones
anticorrelate.

## Scope

- Candidates are computational designs only. The ESMFold screen is a
  model-confidence signal, not a solved structure and not a function
  claim. It upgrades "scores well on two sequence models" to "also
  predicted to fold," and that is as far as the evidence goes.
- Two backbones, one temperature, one seed is a demo, not a study.
  `config/config.yaml` holds all settings for sweeps.
- ProteinMPNN and ESM-2 are upstream models (dauparas/ProteinMPNN,
  facebook/esm). This repo is orchestration + evaluation.

MIT licensed.

## Related work

- The pinned-seed provenance discipline here is what caught the upstream ProteinMPNN `--seed 0` bug, filed as [ProteinMPNN issue #154](https://github.com/dauparas/ProteinMPNN/issues/154).
- [dti-fusion](https://github.com/barlowa124/mol-ml/tree/main/dti_fusion) and [active-learning-loop](https://github.com/barlowa124/protein-ml/tree/main/active_learning_loop) use the same ESM-2 stack. The AL repo's committed negative result (one-hot beating ESM-2 on GB1) bounds what these embeddings can claim.
