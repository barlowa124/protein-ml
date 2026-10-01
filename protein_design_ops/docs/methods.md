# Methods and scope

## What each model contributes

| Stage | Model | What it sees | What its score means |
|---|---|---|---|
| generate | ProteinMPNN (v_48_020) | backbone atom features, sequence context | backbone-conditioned sequence likelihood (negative logP; lower = better) |
| score | ESM-2 (esm2_t6_8M) | sequence only | masked-marginal PLL: natural-sequence fitness prior (higher = better) |
| fold | ESMFold (esmfold_v1) or Chai-1 (chai-lab CLI) | sequence only | predicted structure + model confidence (pLDDT, pTM; Chai-1 also aggregate score, ipTM, clash flag) |
| selfconsistency | none (Kabsch CA RMSD) | predicted structure vs input backbone | whether the design refolds onto the backbone it was designed for |

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
- The structure screen is model confidence (pLDDT/pTM/aggregate score)
  plus backbone self-consistency CA RMSD. Neither is an experimental
  structure, and pLDDT is not calibrated to a specific lab outcome.
- The Chai-1 backend runs in a separate environment (chai_lab requires
  torch<2.7) and is invoked as a subprocess. The driver is covered by
  fake-binary tests; a full Chai-1 screening run downloads ~6 GB of
  weights and is CPU-slow.
- Sampling temperature and seed are config'd. A study would sweep both
  and report the score distribution, not a single run.

## External dependencies

- `dauparas/ProteinMPNN` - cloned separately (`config.mpnn.repo_path`),
  invoked as a subprocess. Its weights ship with the repo. MIT license.
- `facebook/esm2_t6_8M_UR50D` via HuggingFace transformers, downloaded
  on first run.
- `facebook/esmfold_v1` via HuggingFace transformers, ~2.4 GB weights on
  first run. Apache-2.0 license.
- `chaidiscovery/chai-lab` (`pip install chai_lab`), invoked as a
  subprocess via `config.chai.command`. It requires torch<2.7 so it lives
  in its own environment. Weights download (~6 GB) on first run.
