"""Sequence -> melting point with a conformal interval and an AD flag.

Loads deploy_esm2.json (exported by `protstab.run`, committed in the
source repo under results/) and embeds with the same ESM-2 masked-mean
pooling used at training time. Research/education demo: intervals bound
model error at the stated level, not assay reproducibility.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

AA = "ACDEFGHIKLMNPQRSTVWY"
MODEL_ID = "facebook/esm2_t6_8M_UR50D"
DEP = json.loads(Path("deploy_esm2.json").read_text())
MAX_LEN = DEP["encoder"].get("max_len", 1024)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
esm = AutoModel.from_pretrained(MODEL_ID).eval()

MEAN = np.asarray(DEP["linear_model"]["scaler_mean"], dtype=np.float64)
SCALE = np.asarray(DEP["linear_model"]["scaler_scale"], dtype=np.float64)
COEF = np.asarray(DEP["linear_model"]["coef"], dtype=np.float64)
INTERCEPT = float(DEP["linear_model"]["intercept"])
CENTROID = np.asarray(DEP["applicability_domain"]["centroid"], dtype=np.float64)
EDGES = np.asarray(DEP["applicability_domain"]["bin_edges_inner"])
Q_BIN = np.asarray(DEP["applicability_domain"]["bin_q"])
Q_GLOBAL = float(DEP["applicability_domain"]["global_q"])
COVERAGE = 1.0 - float(DEP["conformal_alpha"])


def embed(seq):
    """Same pooling as training: mean over every non-pad position
    (BOS/EOS included) of the last hidden state."""
    enc = tok(seq, return_tensors="pt", truncation=True,
              max_length=MAX_LEN)
    with torch.no_grad():
        hidden = esm(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return (hidden * mask).sum(1).div(mask.sum(1)).numpy()[0]


def predict(sequence):
    seq = "".join(c for c in sequence.strip().upper() if c in AA)
    dropped = len(sequence.strip()) - len(seq)
    if not seq:
        return "Enter a protein sequence (letters ACDEFGHIKLMNPQRSTVWY)."
    note = ""
    if dropped:
        note += f" ({dropped} non-standard residue(s) ignored)"
    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]
        note += f" (truncated to {MAX_LEN} aa, as at training time)"

    x = embed(seq)
    dist = float(np.linalg.norm(x - CENTROID))
    # side="left" matches training's (lower < dist <= upper) convention:
    # a distance exactly on a bin edge lands in the lower bin
    bin_i = int(np.searchsorted(EDGES, dist, side="left"))
    z = (x - MEAN) / SCALE
    yhat = float(z @ COEF + INTERCEPT)
    q = float(Q_BIN[bin_i])

    caveat = ""
    if bin_i == len(Q_BIN) - 1:
        caveat = (" -- farthest applicability-domain bin: extrapolation is "
                  "strongest here and empirical coverage was lowest "
                  "(~0.85). Treat the interval as a floor.")
    return (
        f"**Predicted melting point: {yhat:.1f} °C**\n\n"
        f"{COVERAGE:.0%} conformal interval: **[{yhat - q:.1f}, "
        f"{yhat + q:.1f}] °C** (half-width {q:.1f} °C, calibrated per "
        f"domain bin)\n\n"
        f"Applicability domain: bin {bin_i + 1}/{len(Q_BIN)} "
        f"(embedding distance {dist:.2f} from training centroid){caveat}"
        f"{note}\n\n*Ridge on mean-pooled ESM-2 (8M), Meltome atlas "
        f"27,951 proteins, homology-separated eval. Model-error bounds "
        f"only; research/education use.*"
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Protein sequence",
                      placeholder="MKKVGMGIFN..."),
    outputs=gr.Markdown(),
    title="Protein melting point with conformal interval",
    description=("Sequence -> Tm + a conformal interval conditioned on an "
                 "applicability-domain bin. The interval widens as inputs "
                 "move away from the training manifold."),
    examples=[
        ["MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"],
    ],
)

if __name__ == "__main__":
    demo.launch()
