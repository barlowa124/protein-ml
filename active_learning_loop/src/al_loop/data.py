"""Fetch the GB1 landscape from the FLIP mirror and persist a clean parquet.

Source: J-SNACKKB/FLIP splits/gb1/four_mutations_full_data.csv.zip, which
extends the supplement of Wu et al., eLife 2016 (elife-16965) with derived
columns. The Fitness column is the measured enrichment value used here as
the ground-truth oracle for the selection simulation.

Not part of the default DAG for raw retention reasons: the zip is downloaded
to data/raw/ (gitignored), parsed, and the compact variant/fitness table is
what downstream stages consume.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from al_loop.config import load_config

DOWNLOAD_TIMEOUT = 300


def fetch_raw(url: str, out_zip: str) -> Path:
    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r:
            tmp.write_bytes(r.read())
        tmp.rename(out)  # atomic: no truncated zip reused silently
    return out


def parse_landscape(zip_path: str, variant_col: str, fitness_col: str,
                    variant_regex: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"expected one CSV in {zip_path}, got {names}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f)
    for col in (variant_col, fitness_col):
        if col not in df.columns:
            raise ValueError(
                f"missing column {col!r} in {names[0]}: {list(df.columns)}"
            )
    out = df[[variant_col, fitness_col]].dropna().drop_duplicates(variant_col)
    out.columns = ["variant", "fitness"]
    out["fitness"] = out["fitness"].astype(float)
    # variants outside the declared alphabet/length can't be one-hot
    # encoded (indels, ambiguous residues); drop them and say how many.
    canonical = out["variant"].str.fullmatch(variant_regex)
    n_dropped = int((~canonical).sum())
    if n_dropped:
        print(f"dropping {n_dropped} variants outside {variant_regex}")
    return out[canonical].reset_index(drop=True)


def main(zip_path: str, out_parquet: str):
    cfg = load_config()
    fetch_raw(cfg["dataset"]["url"], zip_path)
    df = parse_landscape(
        zip_path, cfg["dataset"]["variant_col"], cfg["dataset"]["fitness_col"],
        cfg["dataset"]["variant_regex"],
    )
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(
        f"landscape: {len(df)} variants, "
        f"fitness mean {df.fitness.mean():.4f}, max {df.fitness.max():.4f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
