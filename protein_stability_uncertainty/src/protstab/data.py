"""Fetch + parse the FLIP meltome `mixed` split.

The zip ships a ready-made CSV (sequence, target=melting point C, set,
validation flag) built from MMSeqs2 20%-identity clusters: held-out
clusters share <=20% identity with training: homology-aware evaluation
by construction. The `validation` column marks a train-side subset we
use as the conformal calibration set, so calibration never touches the
held-out clusters.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from protstab.config import load_config

DOWNLOAD_TIMEOUT = 300


def fetch_raw(url: str, out_zip: str) -> Path:
    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r:
            tmp.write_bytes(r.read())
        tmp.rename(out)  # atomic: a truncated zip is never reused silently
    return out


def parse_split(zip_path: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    for col in ("sequence", "target", "set"):
        if col not in df.columns:
            raise ValueError(f"missing column {col!r}: {list(df.columns)}")
    df = df.dropna(subset=["sequence", "target"])
    df["target"] = df["target"].astype(float)
    df["validation"] = df["validation"].fillna(False).astype(bool)
    return df.reset_index(drop=True)


def main(zip_path: str, out_parquet: str):
    cfg = load_config()
    fetch_raw(cfg["dataset"]["url"], zip_path)
    df = parse_split(zip_path, cfg["dataset"]["csv_name"])
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(
        f"meltome: {len(df)} proteins | "
        f"train {(df.set == 'train').sum()}, test {(df.set == 'test').sum()}, "
        f"cal {df.validation.sum()} | target mean {df.target.mean():.1f} C"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
