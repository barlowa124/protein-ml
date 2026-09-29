# Methods and scope

## What each model contributes

| Stage | Model | What it sees | What its score means |
|---|---|---|---|
| generate | ProteinMPNN (v_48_020) | backbone atom features, sequence context | backbone-conditioned sequence likelihood (negative logP; lower = better) |
| score | ESM-2 (esm2_t6_8M) | sequence only | masked-marginal PLL: natural-sequence fitness prior (higher = better) |

The models are independent: MPNN never sees ESM's training distribution
directly and ESM never sees the backbone. Agreement between a
structure-conditioned likelihood and a sequence-prior fitness is
corroborating evidence. Anticorrelation (as observed on 1L2Y) means the
design sits where the two objectives trade off.

## Known limitations

- ESM-2 PLL is a sequence prior. It does not know the backbone, so a
  high-ESM sequence can still be a poor MPNN candidate and vice versa.
- MPNN `seq_recovery` is sequence identity to the input native sequence,
  not a folding metric.
- No structure validation of designs is performed. A stronger pipeline
  would fold top consensus candidates (ESMFold/Boltz) and filter by
  pLDDT/TM-score to backbone.
- Sampling temperature and seed are config'd. A study would sweep both
  and report the score distribution, not a single run.

## External dependencies

- `dauparas/ProteinMPNN` - cloned separately (`config.mpnn.repo_path`),
  invoked as a subprocess. Its weights ship with the repo. MIT license.
- `facebook/esm2_t6_8M_UR50D` via HuggingFace transformers, downloaded
  on first run.
