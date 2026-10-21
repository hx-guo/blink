#!/usr/bin/env python3
"""Stream-partition train_cache.parquet → 18 per-(box, det) npz files + cross-det parquet.

Avoids the 26 GB peak from loading the whole parquet by streaming row groups.

Outputs:
    n_below_study/perdet_npz/{box}_{det}.npz   (18 files, ~280 MB each)
    n_below_study/box_totals.parquet            (cross-det sums per (date,box,sec))
"""
from pathlib import Path
import time
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CACHE = Path("n_below_study/train_cache.parquet")
PERDET_DIR = Path("n_below_study/perdet_npz")
PERDET_DIR.mkdir(exist_ok=True)
BOX_TOTALS = Path("n_below_study/box_totals.parquet")

COLS = ["date", "box", "det", "met_sec",
        "scipure_rate", "acd_rate", "wide_rate", "large_rate",
        "pho_rate", "dt_frac", "ratio_local", "sci_rate", "group_rate"]


def pass1_partition_perdet():
    """Pass 1: read row groups, split into per-(box, det) accumulators, save."""
    pf = pq.ParquetFile(CACHE)
    n_rgs = pf.num_row_groups
    print(f"Pass 1: streaming {n_rgs} row groups → 18 per-(box, det) npz files",
          flush=True)

    accs = {(b, d): {c: [] for c in COLS if c not in ("box", "det")}
            for b in "ABC" for d in range(6)}
    t0 = time.time()

    for rg in range(n_rgs):
        table = pf.read_row_group(rg, columns=COLS)
        df = table.to_pandas()
        for (box, det), sub in df.groupby(["box", "det"], observed=True):
            for c in accs[(box, det)]:
                accs[(box, det)][c].append(sub[c].values)
        if (rg + 1) % 16 == 0 or rg == n_rgs - 1:
            print(f"  RG {rg+1}/{n_rgs}  elapsed {time.time()-t0:.0f}s",
                  flush=True)
        del table, df
        gc.collect()

    print(f"\n  concatenating and saving 18 npz files...", flush=True)
    for (box, det), cols in accs.items():
        arrs = {}
        for c, lst in cols.items():
            if not lst: continue
            if c == "date":
                # Store date as int32 YYYYMMDD (e.g., 20170615) to avoid object dtype
                concat = np.concatenate(lst)
                arrs[c] = np.array([int(s.replace("-","")) for s in concat], dtype=np.int32)
            elif c == "met_sec":
                arrs[c] = np.concatenate(lst).astype(np.int64)
            else:
                arrs[c] = np.concatenate(lst).astype(np.float32)
        out = PERDET_DIR / f"{box}_{det}.npz"
        np.savez(out, **arrs)
        sz = out.stat().st_size / 1e6
        n = len(arrs.get("pho_rate", []))
        print(f"  {box}-{det}: {n:,} rows, {sz:.1f} MB", flush=True)
    accs.clear()
    gc.collect()


def pass2_box_totals():
    """Pass 2: compute per-(date, box, met_sec) sums of 4 channels.
       This stays in memory because the result is ~22M rows × 4 cols = small."""
    print(f"\nPass 2: computing cross-det box totals (groupby)...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(CACHE, columns=["date","box","met_sec",
                                          "scipure_rate","acd_rate",
                                          "wide_rate","large_rate"])
    print(f"  read {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)
    df = df.groupby(["date","box","met_sec"], observed=True, sort=False).sum().reset_index()
    print(f"  grouped to {len(df):,} (date,box,sec) totals "
          f"in {time.time()-t0:.0f}s", flush=True)
    df.rename(columns={"scipure_rate":"scipure_sum",
                       "acd_rate":"acd_sum",
                       "wide_rate":"wide_sum",
                       "large_rate":"large_sum"}, inplace=True)
    df.to_parquet(BOX_TOTALS, compression="zstd")
    print(f"  saved {BOX_TOTALS} ({BOX_TOTALS.stat().st_size/1e6:.1f} MB)",
          flush=True)


def main():
    t0 = time.time()
    pass1_partition_perdet()
    pass2_box_totals()
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
