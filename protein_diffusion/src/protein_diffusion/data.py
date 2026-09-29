"""Fetch the variant/fitness landscape (FLIP mirror) and persist a clean
parquet: the FULL measured landscape, dead variants included. (v2 trains
on all of it; v1's fit-only training set is why v1 memorized. See README.)
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from protein_diffusion.config import load_config

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
                    variant_regex: str = r"[ACDEFGHIKLMNPQRSTVWY]{4}") -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"expected one CSV in {zip_path}, got {names}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    for col in (variant_col, fitness_col):
        if col not in df.columns:
            raise ValueError(f"missing column {col!r}: {list(df.columns)}")
    out = df[[variant_col, fitness_col]].dropna().drop_duplicates(variant_col)
    out.columns = ["variant", "fitness"]
    out["fitness"] = out["fitness"].astype(float)
    canonical = out["variant"].str.fullmatch(variant_regex)
    n_dropped = int((~canonical).sum())
    if n_dropped:
        print(f"dropping {n_dropped} variants outside {variant_regex}")
    return out[canonical].reset_index(drop=True)


def main(zip_path: str, out_parquet: str):
    cfg = load_config()
    fetch_raw(cfg["dataset"]["url"], zip_path)
    ds = cfg["dataset"]
    df = parse_landscape(
        zip_path, ds["variant_col"], ds["fitness_col"],
        ds.get("variant_regex", r"[ACDEFGHIKLMNPQRSTVWY]{4}"),
    )
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(
        f"landscape: {len(df)} variants | fitness>={0.5} fraction: "
        f"{(df.fitness >= 0.5).mean():.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
