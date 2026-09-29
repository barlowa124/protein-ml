"""Backbone ingestion: parse the configured PDB, extract the design chain.

Output (data/processed/backbone.json):
    pdb, chain, n_residues, native_seq, resseqs
"""

from __future__ import annotations

import json
import sys
from design_ops.config import load_config



AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_chain(pdb_path: str, chain: str) -> dict:
    """Residue sequence (1-letter) and resseq list for one chain."""
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break  # NMR ensembles: use MODEL 1 only
            # skip short/malformed records instead of slice-crashing
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            if line[21] != chain or line[12:16].strip() != "CA":
                continue
            resseq = int(line[22:26])
            resname = line[17:20].strip()
            if resname in AA3:
                residues.setdefault(resseq, AA3[resname])
    if not residues:
        raise ValueError(f"no CA atoms found for chain {chain!r} in {pdb_path}")
    resseqs = sorted(residues)
    return {
        "resseqs": resseqs,
        "native_seq": "".join(residues[r] for r in resseqs),
    }


def main() -> None:
    out_path = sys.argv[1]
    cfg = load_config()
    bb = cfg["backbone"]
    info = parse_chain(bb["pdb"], bb["chain"])
    info.update({"pdb": bb["pdb"], "chain": bb["chain"],
                 "n_residues": len(info["resseqs"])})
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"backbone: {bb['pdb']} chain {bb['chain']}: "
        f"{info['n_residues']} residues, native seq {info['native_seq']}"
    )


if __name__ == "__main__":
    main()
