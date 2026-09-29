---
title: Protein Melting Point + Conformal Interval
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Protein melting point with conformal interval

Paste a protein sequence and get a predicted melting point (Tm) with a
**90% conformal interval** and an **applicability-domain flag**.

- Model: ridge regression on mean-pooled ESM-2 (`esm2_t6_8M`, 320-d).
- Data: Meltome atlas via FLIP, 27,951 proteins, held-out split is
  homology-separated (<=20% identity).
- Interval: Mondrian conformal, a residual quantile per applicability-
  domain bin, so coverage stays flat (~0.87-0.91) as distance from the
  training manifold grows. Intervals widen 9.8 -> 14.5 °C from nearest
  to farthest bin.

**Scope**: intervals bound *model* error at the stated level, not assay
reproducibility. Research/education demo.

Source repo: https://github.com/barlowa124/protein-ml/tree/main/protein_stability_uncertainty
(pipeline, evaluation, and the `results/deploy_esm2.json` artifact this
app loads).
