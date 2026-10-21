#!/usr/bin/env python3
"""Downsample HV table to 1-row-per-minute granularity.

HV is stable within minutes (varies < 0.1 V/min), so per-second resolution is
wasteful. Compress 280M rows → ~4.7M rows by keeping one row per (date, minute).
Streamed chunked read keeps memory under control.

Input:  n_below_study/hv_table_full.csv.gz       (5.8 GB)
Output: n_below_study/hv_table_1min.csv.gz       (~100 MB)
"""
from pathlib import Path
import time
import pandas as pd

SRC = Path("n_below_study/hv_table_full.csv.gz")
OUT = Path("n_below_study/hv_table_1min.csv.gz")
DTYPE = {"date":"string","met_sec":"int64",
         **{f"hv{i}":"float32" for i in range(18)}}


def main():
    print(f"Reading {SRC} in chunks (5M rows each)...")
    t0 = time.time()
    kept_chunks = []
    n_in = 0
    for ci, chunk in enumerate(pd.read_csv(SRC, chunksize=5_000_000, dtype=DTYPE)):
        n_in += len(chunk)
        # Keep first row per (date, minute)
        chunk = chunk.assign(_min=chunk["met_sec"] // 60)
        chunk = chunk.drop_duplicates(["date", "_min"], keep="first").drop(columns=["_min"])
        kept_chunks.append(chunk)
        print(f"  chunk {ci+1}: in={n_in:,} kept_so_far={sum(len(c) for c in kept_chunks):,} "
              f"elapsed={time.time()-t0:.0f}s")

    print("\nMerging chunks + final dedupe at minute boundaries...")
    df = pd.concat(kept_chunks, ignore_index=True)
    del kept_chunks
    df = df.assign(_min=df["met_sec"] // 60)
    df = df.drop_duplicates(["date", "_min"], keep="first").drop(columns=["_min"])
    df = df.sort_values(["date","met_sec"]).reset_index(drop=True)
    print(f"final rows: {len(df):,}  memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB")

    print(f"\nWriting {OUT}...")
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"Saved: {OUT.stat().st_size/1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
