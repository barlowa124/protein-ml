"""Self-consistency screen: C-alpha RMSD of each folded design vs the
input backbone.

The sequence-space scores (MPNN likelihood, ESM-2 PLL) say the design
looks plausible to a language model. The structure screen closes the
loop differently: if a designed sequence refolds onto the backbone it
was designed for, that is evidence the sequence actually encodes the
fold. This stage computes that backbone RMSD for every folded record.

Structures come from whichever backend the fold stage ran:
`structure_path` on each record points to a PDB (ESMFold) or mmCIF
(Chai-1) file. Records without a usable structure keep their scores and
get bb_rmsd=None so a missing fold never hides a sequence.

Usage:
    python -m design_ops.selfconsistency fold.json backbone.json sc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def parse_ca_coords_pdb(text: str) -> np.ndarray:
    """CA coordinates from PDB ATOM records, first model only."""
    coords = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and len(line) >= 54:
            if line[12:16].strip() == "CA":
                coords.append(
                    (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
    return np.array(coords, dtype=float)


def parse_ca_coords_cif(text: str) -> np.ndarray:
    """CA coordinates from an mmCIF atom_site loop.

    Column order is read from the loop's own `_atom_site.<name>` headers
    so files with different column layouts still parse. Atom name comes
    from `auth_atom_id` when present (Chai-1 writes it), else
    `label_atom_id`.
    """
    lines = text.splitlines()
    coords = []
    i = 0
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
                try:
                    ix = col["Cartn_x"]
                    iy = col["Cartn_y"]
                    iz = col["Cartn_z"]
                except KeyError:
                    i = j
                    continue
                while j < len(lines) and lines[j].startswith(("ATOM", "HETATM")):
                    parts = lines[j].split()
                    if atom_col is not None and parts[atom_col] == "CA":
                        coords.append(
                            (
                                float(parts[ix]),
                                float(parts[iy]),
                                float(parts[iz]),
                            )
                        )
                    j += 1
            i = j
        else:
            i += 1
    return np.array(coords, dtype=float)


def parse_ca_coords(path: str) -> np.ndarray:
    """Dispatch on extension: .cif/.mmcif -> mmCIF parser, else PDB."""
    text = Path(path).read_text()
    if path.endswith((".cif", ".mmcif")):
        return parse_ca_coords_cif(text)
    return parse_ca_coords_pdb(text)


def kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """RMSD between two equal-length CA traces after optimal rotation."""
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            f"shape mismatch: {mobile.shape} vs {reference.shape}"
        )
    if len(mobile) == 0:
        raise ValueError("empty coordinate sets")
    m = mobile - mobile.mean(axis=0)
    r = reference - reference.mean(axis=0)
    h = m.T @ r
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = m @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - r) ** 2, axis=1))))


def add_selfconsistency(
    records: list[dict], backbone: dict
) -> list[dict]:
    """Attach bb_rmsd to every record whose fold produced a structure."""
    ref_coords = np.array(backbone["ca_coords"], dtype=float)

    out = []
    for rec in records:
        rec = dict(rec)
        if not rec.get("fold") or not rec.get("structure_path"):
            if rec.get("fold"):
                rec["fold"]["bb_rmsd"] = None
                rec["fold"]["bb_rmsd_note"] = "no structure path recorded"
            out.append(rec)
            continue
        try:
            coords = parse_ca_coords(rec["structure_path"])
            rec["fold"]["bb_rmsd"] = round(
                kabsch_rmsd(coords, ref_coords), 3
            )
        except (OSError, ValueError, IndexError) as e:
            rec["fold"]["bb_rmsd"] = None
            rec["fold"]["bb_rmsd_note"] = str(e)
        out.append(rec)
    return out


def main() -> None:
    fold_json, backbone_json, out_json = sys.argv[1:4]
    records = json.loads(Path(fold_json).read_text())
    backbone = json.loads(Path(backbone_json).read_text())
    out = add_selfconsistency(records, backbone)
    Path(out_json).write_text(json.dumps(out, indent=1))
    n = sum(
        1 for r in out if r.get("fold") and r["fold"].get("bb_rmsd") is not None
    )
    print(f"self-consistency: {n}/{len(out)} structures aligned -> {out_json}")


if __name__ == "__main__":
    main()
