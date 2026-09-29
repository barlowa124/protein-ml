"""Train + calibrated evaluation on the meltome split.

Proper split-conformal: ridge is fit on train-minus-validation, absolute
residuals calibrated on the FLIP `validation` subset (still train-side),
and coverage measured on the held-out cluster split. The intervals are
valid only because calibration never sees the test set.

Applicability domain: Euclidean distance in embedding space to the
training centroid. Bin edges are fixed on the *calibration* set (test
quartiles would leak), so unequal test-bin sizes reflect real domain
shift. Marginal coverage can hide a domain gradient: Mondrian conformal
calibrates a separate residual quantile per bin and recovers per-bin
coverage at the cost of wider intervals where extrapolation is real.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protstab.config import load_config
from protstab.encode import build_features, composition
from protstab.provenance import write_manifest


def _quiet(fn, *args):
    """macOS Accelerate BLAS raises spurious FP warnings (divide/overflow/
    invalid) inside any large matmul. Results verified against float64
    reference (max diff ~1e-6); the flag state is the artifact, not the
    math."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return fn(*args)


def fit_ridge(X, y, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    _quiet(m.fit, X, y)
    return m


def predict(m, X):
    return _quiet(m.predict, X)


def metrics(y, yhat):
    from scipy.stats import spearmanr
    err = yhat - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "spearman": float(spearmanr(y, yhat)[0]),
    }


def _conformal_quantile(resid: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal level: the ceil((n+1)(1-alpha))-th
    smallest of n calibration residuals. The plain (1-alpha) quantile
    under-covers; the +1 correction is what makes the guarantee hold."""
    n = len(resid)
    level = min(1.0, (1 - alpha) * (n + 1) / n)
    return float(np.quantile(resid, level, method="higher"))


def conformal_eval(m, X_cal, y_cal, X_te, y_te, alpha):
    """Split conformal: q = corrected (1-alpha) quantile of |y - yhat|
    on cal."""
    q = _conformal_quantile(np.abs(predict(m, X_cal) - y_cal), alpha)
    yhat = predict(m, X_te)
    cov = np.abs(yhat - y_te) <= q
    return q, yhat, cov


def _descriptive_table(dist_te, cov, resid_abs, edges):
    """Marginal coverage + error per fixed AD bin (diagnostic, not calibrated)."""
    rows = []
    for b in range(len(edges) - 1):
        sel = (dist_te > edges[b]) & (dist_te <= edges[b + 1])
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(sel.sum()),
            "dist_range": [float(edges[b]),
                           float(min(edges[b + 1], dist_te.max()))],
            "coverage": float(cov[sel].mean()),
            "mae": float(resid_abs[sel].mean()),
        })
    return rows


def mondrian_eval(m, X_cal, y_cal, dist_cal, dist_te, yhat, y_te,
                  alpha, bins, min_cal=30):
    """Mondrian conformal: per-AD-bin residual quantiles, edges fixed on
    calibration distances. Sparse bins fall back to the global quantile."""
    resid_cal = np.abs(predict(m, X_cal) - y_cal)
    q_global = _conformal_quantile(resid_cal, alpha)
    edges = np.quantile(dist_cal, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, np.inf
    resid_te = np.abs(yhat - y_te)
    cov = np.zeros(len(y_te), dtype=bool)
    q_bins = np.full(bins, q_global)
    rows = []
    for b in range(bins):
        sel_c = (dist_cal > edges[b]) & (dist_cal <= edges[b + 1])
        sel_t = (dist_te > edges[b]) & (dist_te <= edges[b + 1])
        q_b = (_conformal_quantile(resid_cal[sel_c], alpha)
               if sel_c.sum() >= min_cal else q_global)
        q_bins[b] = q_b
        cov[sel_t] = resid_te[sel_t] <= q_b
        if sel_t.sum():
            rows.append({
                "bin": b,
                "n_cal": int(sel_c.sum()),
                "n_test": int(sel_t.sum()),
                "dist_range": [float(edges[b]),
                               float(min(edges[b + 1], dist_te.max()))],
                "q": float(q_b),
                "coverage": float(cov[sel_t].mean()),
                "mae": float(resid_te[sel_t].mean()),
                "fallback": bool(sel_c.sum() < min_cal),
            })
    return rows, cov, edges, q_bins


def export_deployable(m, name, centroid, edges, q_bins, q_global,
                      enc, alpha, out_path):
    """Write the fitted model + conformal parameters as plain JSON.

    Ridge + StandardScaler + conformal quantiles serialize losslessly;
    a JSON artifact is diffable and loads in any runtime (e.g. the
    HuggingFace Space) without pickle."""
    sc = m.named_steps["standardscaler"]
    ridge = m.named_steps["ridge"]
    doc = {
        "feature": name,
        "linear_model": {
            "coef": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
        },
        "applicability_domain": {
            "centroid": centroid.tolist(),
            "bin_edges_inner": [float(e) for e in edges[1:-1]],
            "global_q": float(q_global),
            "bin_q": q_bins.tolist(),
        },
        "encoder": enc,
        "conformal_alpha": alpha,
        "note": ("research/education only. Interval bounds model error at "
                 "the stated alpha, not assay reproducibility."),
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)


def main(in_parquet: str, out_json: str):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    ev = cfg["evaluation"]
    alpha = ev["conformal_alpha"]

    tr = df[(df.set == "train") & (~df.validation)].reset_index(drop=True)
    cal = df[(df.set == "train") & (df.validation)].reset_index(drop=True)
    te = df[df.set == "test"].reset_index(drop=True)
    max_len = cfg["dataset"]["max_len"]
    n_trunc = int((df.sequence.str.len() > max_len).sum())
    if n_trunc:
        print(f"truncating {n_trunc} sequences > {max_len} aa")

    enc = dict(cfg["encoder"])
    enc["max_len"] = max_len
    stem = str(Path(in_parquet).with_suffix(""))
    results = {}

    # Baseline first: composition features, same protocol
    for name, Xf in {
        "composition": composition(
            pd.concat([tr.sequence, cal.sequence, te.sequence])),
        "esm2": build_features(
            pd.concat([tr.sequence, cal.sequence, te.sequence]),
            enc, cache_stem=f"{stem}_all"),
    }.items():
        ntr, ncal = len(tr), len(cal)
        X_tr, X_cal, X_te = Xf[:ntr], Xf[ntr:ntr + ncal], Xf[ntr + ncal:]
        m = fit_ridge(X_tr, tr.target.to_numpy(), cfg["model"]["alpha"])
        q, yhat, cov = conformal_eval(
            m, X_cal, cal.target.to_numpy(), X_te, te.target.to_numpy(),
            alpha)
        centroid = X_tr.mean(axis=0)
        dist_cal = np.linalg.norm(X_cal - centroid, axis=1)
        dist_te = np.linalg.norm(X_te - centroid, axis=1)
        mon_rows, mon_cov, edges, q_bins = mondrian_eval(
            m, X_cal, cal.target.to_numpy(), dist_cal, dist_te,
            yhat, te.target.to_numpy(), alpha, ev["ad_bins"])
        export_deployable(
            m, name, centroid, edges, q_bins, q, enc, alpha,
            str(Path(out_json).parent / f"deploy_{name}.json"))
        res = {
            "conformal_q": float(q),
            "coverage": float(cov.mean()),
            "mean_interval_width": float(2 * q),
            "mondrian_coverage": float(mon_cov.mean()),
            "ad_table": _descriptive_table(dist_te, cov,
                                           np.abs(yhat - te.target.to_numpy()),
                                           edges),
            "mondrian_table": mon_rows,
            **metrics(te.target.to_numpy(), yhat),
        }
        results[name] = res
        print(f"{name}: rmse {res['rmse']:.2f} mae {res['mae']:.2f} "
              f"spearman {res['spearman']:.3f} | coverage "
              f"{res['coverage']:.3f} (mondrian {res['mondrian_coverage']:.3f}) "
              f"@ target {1 - alpha:.0%} "
              f"(width {res['mean_interval_width']:.1f} C)")

    out = {
        "config": {"encoder": enc, "model": cfg["model"],
                   "conformal_alpha": alpha, "max_len": max_len,
                   "n_truncated": n_trunc},
        "n_train": len(tr), "n_cal": len(cal), "n_test": len(te),
        "target_unit": "C",
        "results": results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    manifest = (Path(out_json).parent
                / ("provenance" + Path(out_json).stem.removeprefix("summary")
                   + ".json"))
    write_manifest(str(manifest), inputs=[in_parquet])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
