#!/usr/bin/env python3
"""Cache full training DataFrame to parquet.

Memory-efficient: processes one CSV file at a time, applies all per-row filters
and computes derived columns inline, then appends only the filtered subset to a
running list. Avoids a 2× peak from a big concat-then-filter pattern.

Usage:  python3 scripts/saturation/cache_training.py
Then:   train = pd.read_parquet("n_below_study/train_cache.parquet")
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_PARQUET_DIR = Path("n_below_study/hv_by_date_parquet")   # per-date parquet partitions
CACHE = Path("n_below_study/train_cache.parquet")

L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

DTYPE = {"date":"string","box":"category","met_sec":"int64","det":"int8",
         "L_cycles":"int32","PHO":"int32","Wide":"int32","Large":"int32",
         "Dt":"int32","Sci":"int32","Sci_ACD1":"int32","Sci_ACDN":"int32"}


def process_file(f, hv_dir):
    """Read one CSV, filter, compute derived columns. Returns filtered df or None.

    Per-date HV is loaded on demand from the partitioned parquet directory."""
    try:
        df = pd.read_csv(f, usecols=list(DTYPE), dtype=DTYPE)
    except Exception:
        return None
    if len(df) == 0: return None

    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH]
    if len(df) == 0: return None

    g = df.groupby(["box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()
    if len(df) == 0: return None

    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["Sci_ACD"]  = df["Sci_ACD1"] + df["Sci_ACDN"]
    df["ratio_local"] = (df["Sci_ACD"].astype("float32")
                         / df["Sci"].astype("float32").clip(lower=1)).clip(0, 1)
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd_rate","Sci_ACD"),("wide_rate","Wide"),
                    ("large_rate","Large"),("pho_rate","PHO")]:
        df[c] = (df[src] / df["length"]).astype("float32")
    df["group_rate"] = (df["sci_sec_total"] / df["length"]).astype("float32")
    df["dt_frac"]    = (df["Dt"].astype("float32") / df["L_cycles"]).astype("float32")
    # box is categorical — convert to plain Series before arithmetic
    df["det_global"] = (df["box"].astype(str).map(BOX_OFFSET).astype("int8")
                         + df["det"].astype("int8")).astype("int8")

    # HV lookup: load only this date's per-second HV from parquet
    date_str = df["date"].iloc[0].replace("-", "")
    hv_path = hv_dir / f"{date_str}.parquet"
    if not hv_path.exists():
        return None     # no HV for this date → skip whole file
    hv_day = pd.read_parquet(hv_path)
    hv_day = hv_day.drop_duplicates("met_sec", keep="first").set_index("met_sec").sort_index()
    hv_arr = hv_day.reindex(df["met_sec"].values).values
    df["hv"] = hv_arr[np.arange(len(df)), df["det_global"].values.astype(int)].astype("float32")
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    if len(df) == 0: return None

    keep = ["date","box","det","met_sec","L_cycles","length",
            "sci_rate","scipure_rate","acd_rate","wide_rate","large_rate","pho_rate",
            "group_rate","dt_frac","ratio_local","det_global","hv",
            "Sci","Sci_ACD","Sci_pure"]
    return df[keep].reset_index(drop=True)


def main():
    print(f"Using per-date HV parquet partitions in {HV_PARQUET_DIR}")
    if not HV_PARQUET_DIR.exists():
        raise FileNotFoundError(
            f"{HV_PARQUET_DIR} not found. Run scripts/saturation/build_hv_parquet.py first."
        )
    n_partitions = len(list(HV_PARQUET_DIR.glob("*.parquet")))
    print(f"  HV partitions: {n_partitions} dates")

    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"\nProcessing {len(files)} CSV files (per-file filter, low-memory)...")
    parts = []
    t0 = time.time()
    skipped = 0
    for i, f in enumerate(files):
        d = process_file(f, HV_PARQUET_DIR)
        if d is not None and len(d) > 0:
            parts.append(d)
        else:
            skipped += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            kept = sum(len(p) for p in parts)
            print(f"  {i+1}/{len(files)} | kept {len(parts)} files, "
                   f"{kept:,} rows | {elapsed:.0f}s elapsed | skipped {skipped}")

    print("\nConcat...")
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"final rows: {len(df):,}  memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    print("\nSaving parquet...")
    df.to_parquet(CACHE, compression="zstd")
    sz = CACHE.stat().st_size / 1e6
    print(f"Saved: {CACHE} ({sz:.1f} MB on disk)")


if __name__ == "__main__":
    main()
