"""Structure prediction via Chai-1, driven as an external subprocess.

chai_lab pins torch<2.7 while the design environment runs a newer
torch, so Chai-1 runs under its own interpreter (config.chai.command)
the same way ProteinMPNN runs under config.mpnn.repo_path. Each
candidate sequence gets one `chai fold` call producing a .cif
structure and a .npz score file; we extract compact metrics and keep
the structure path so the self-consistency stage can align it back to
the input backbone.

Usage:
    python -m design_ops.chai scores.json fold.json folds_dir
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _npz_scores(npz_path: Path) -> dict:
    """Pull the scalar confidence metrics out of Chai's score npz."""
    z = np.load(npz_path)
    out = {}
    for key in ("aggregate_score", "ptm", "iptm"):
        if key in z:
            out[key] = round(float(np.asarray(z[key]).ravel()[0]), 4)
    if "has_clashes" in z:
        out["has_clashes"] = int(np.asarray(z["has_clashes"]).ravel()[0])
    if "plddt" in z:
        arr = np.asarray(z["plddt"], dtype=float)
        out["plddt_mean"] = round(float(arr.mean()), 2)
        out["plddt_min"] = round(float(arr.min()), 2)
    return out


def _cif_bfactor_plddt(cif_path: Path) -> tuple[float | None, float | None]:
    """Fallback pLDDT: Chai writes per-atom pLDDT in the B-factor column.

    Parses the atom_site loop with header-driven columns (same approach
    as selfconsistency.parse_ca_coords_cif).
    """
    lines = cif_path.read_text().splitlines()
    i, vals = 0, []
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            headers = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            if headers and all(h.startswith("_atom_site.") for h in headers):
                col = {h.split(".", 1)[1]: k for k, h in enumerate(headers)}
                atom_col = col.get("auth_atom_id", col.get("label_atom_id"))
                b_col = col.get("B_iso_or_equiv")
                while b_col is not None and j < len(lines) and lines[
                    j
                ].startswith(("ATOM", "HETATM")):
                    parts = lines[j].split()
                    if atom_col is None or parts[atom_col] == "CA":
                        vals.append(float(parts[b_col]))
                    j += 1
            i = j
        else:
            i += 1
    if not vals:
        return None, None
    return round(float(np.mean(vals)), 2), round(float(np.min(vals)), 2)


def fold_one(
    seq: str,
    command: str,
    work_dir: Path,
    extra_args: list[str] | None = None,
) -> dict:
    """Run `chai fold` on one sequence; return metrics + cif path."""
    # Chai parses the FASTA header for the entity type: it must be
    # "protein|<name>" or the run dies with "not a valid entity type".
    fasta = work_dir / "input.fasta"
    fasta.write_text(f">protein|candidate\n{seq}\n")
    cmd = [command, "fold", str(fasta), str(work_dir)] + list(extra_args or [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"chai fold exited {proc.returncode}: "
            f"{proc.stderr.strip()[-500:]}"
        )
    npz = sorted(work_dir.glob("scores.model_idx_*.npz"))
    cifs = sorted(work_dir.glob("pred.model_idx_*.cif"))
    if not npz or not cifs:
        raise RuntimeError(
            f"chai produced no scores/model files under {work_dir}"
        )
    metrics = _npz_scores(npz[0])
    metrics["structure_path"] = str(cifs[0])
    if "plddt_mean" not in metrics:
        mean, mn = _cif_bfactor_plddt(cifs[0])
        if mean is not None:
            metrics["plddt_mean"], metrics["plddt_min"] = mean, mn
    return metrics


def fold_records(
    records: list[dict],
    cfg: dict,
    folds_dir: Path,
    checkpoint_dir: str | None = None,
) -> list[dict]:
    chai = cfg.get("chai", {})
    command = chai.get("command", "chai")
    extra = chai.get("extra_args") or []
    if shutil.which(command) is None:
        raise RuntimeError(
            f"chai command {command!r} not found on PATH; point "
            "config.chai.command at the chai_lab environment's binary "
            "(chai_lab requires torch<2.7, keep it out of this env)"
        )
    folds_dir.mkdir(parents=True, exist_ok=True)
    parts = Path(checkpoint_dir) if checkpoint_dir else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)
    out = []
    for i, rec in enumerate(records):
        part = parts / f"{i:03d}.json" if parts else None
        if part and part.exists():
            print(f"  chai {i + 1}/{len(records)} (checkpoint)",
                  flush=True)
            out.append(json.loads(part.read_text()))
            continue
        tag = f"{i:03d}_{'native' if rec.get('is_native') else 'design'}"
        if not rec.get("seq"):
            res = {**rec, "fold": None, "fold_error": "empty seq"}
        else:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    f = fold_one(rec["seq"], command, Path(tmp), extra)
                    cif = Path(f.pop("structure_path"))
                    dest = folds_dir / f"{tag}.cif"
                    shutil.copy(cif, dest)
                print(
                    f"  chai {i + 1}/{len(records)} "
                    f"agg={f.get('aggregate_score')}",
                    flush=True,
                )
                res = {
                    **rec,
                    "fold": f,
                    "structure_path": str(dest),
                    "fold_backend": "chai1",
                }
            except Exception as e:  # fold failures are data, not crashes
                res = {**rec, "fold": None, "fold_error": str(e)}
        out.append(res)
        if part:
            part.write_text(json.dumps(res))
    return out


def main() -> None:
    scores_json, out_json, folds_dir = sys.argv[1:4]
    parts_dir = sys.argv[4] if len(sys.argv) > 4 else None
    from design_ops.config import load_config

    cfg = load_config()
    records = json.loads(Path(scores_json).read_text())
    folded = fold_records(records, cfg, Path(folds_dir), parts_dir)
    Path(out_json).write_text(json.dumps(folded, indent=1))
    n_ok = sum(1 for r in folded if r["fold"])
    print(f"chai folded {n_ok}/{len(folded)} sequences -> {out_json}")


if __name__ == "__main__":
    main()
