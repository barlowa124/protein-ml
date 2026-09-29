"""Sequence generation: invoke upstream ProteinMPNN and parse its output.

ProteinMPNN is external (config.mpnn.repo_path); we call its
protein_mpnn_run.py in a subprocess and parse the emitted FASTA, which
carries the model's own score and seq_recovery in each header.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from design_ops.config import load_config


def parse_mpnn_fasta(path: str) -> list[dict]:
    """ProteinMPNN FASTA -> records; first entry is the native sequence."""
    def flush(header, seq_lines, records):
        if header is None:
            return
        if not seq_lines:
            return  # header-only lines (MPNN block markers) carry no record
        meta = dict(re.findall(r"(\w+)=(\S+?)(?:,|$)", header))
        if "score" not in meta:
            raise ValueError(
                f"malformed FASTA header (no score=): {header!r}")
        records.append(
            {
                "header": header,
                "seq": "".join(seq_lines),
                "mpnn_score": float(meta["score"]),
                "seq_recovery": float(meta.get("seq_recovery", 1.0)),
                "is_native": "designed_chains" in header
                    and "sample" not in meta,
            }
        )

    records = []
    header, seq_lines = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                flush(header, seq_lines, records)
                header, seq_lines = line[1:], []
            elif line and header is not None:
                seq_lines.append(line)  # FASTA wrapping: continuation
    flush(header, seq_lines, records)
    return records


def generate(cfg: dict, backbone: dict, workdir: Path) -> list[dict]:
    m = cfg["mpnn"]
    out_dir = workdir / "mpnn_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_abs = str(Path(backbone["pdb"]).resolve())
    cmd = [
        sys.executable,
        str(Path(m["repo_path"]) / m["run_script"]),
        "--pdb_path", pdb_abs,
        "--pdb_path_chains", backbone["chain"],
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(m["num_seq_per_target"]),
        "--sampling_temp", str(m["sampling_temp"]),
        "--batch_size", str(m["batch_size"]),
        "--seed", str(m["seed"]),
        "--model_name", m["model_name"],
        "--path_to_model_weights",
        str(Path(m["repo_path"]) / Path(m["weights"]).parent) + "/",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    fasta = next((out_dir / "seqs").glob("*.fa"), None)
    if fasta is None:
        raise RuntimeError(
            f"ProteinMPNN produced no .fa under {out_dir / 'seqs'}, "
            "check repo_path/run_script config"
        )
    return parse_mpnn_fasta(str(fasta))


def main() -> None:
    backbone_path, out_path = sys.argv[1], sys.argv[2]
    cfg = load_config()
    with open(backbone_path) as f:
        backbone = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        records = generate(cfg, backbone, Path(tmp))
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    n_native = sum(r["is_native"] for r in records)
    print(
        f"generate: {len(records)} sequences "
        f"({len(records) - n_native} designed + {n_native} native) "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
