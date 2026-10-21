#!/usr/bin/env python3
"""Regression test: v2 on H1 strict cache (equatorial belt, low rate, ~no wraps).

H1 strict has Sci typically 50-1500 cnt/s. At these rates Large is usually < 1024
so wrapping should be rare. v1 and v2 should agree almost everywhere; any
disagreement signals v2 is "over-active" in the low-rate regime.

Reports:
- n_wraps distribution v1 vs v2
- per-row diff (v2 - v1)
- residual change in main range
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large import unwrap_large as unwrap_v1
from unwrap_large_v2 import unwrap_large_v2

CACHE = Path("n_below_study/clean_2020H1.parquet")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def main():
    print(f"Loading H1 strict {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")

    # v1
    large_v1 = np.zeros(len(df), dtype=np.float64)
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            large_v1[m] = unwrap_v1(df.loc[m, "PHO"].values, df.loc[m, "Large"].values)

    # v2
    large_v2 = unwrap_large_v2(
        df["PHO"].values, df["Large"].values, df["Wide"].values,
        df["Sci_1s"].values, df["L_cycles"].values, df["Dt"].values,
        C=150.0,
    )

    n_wraps_v1 = ((large_v1 - df["Large"].values) / 1024.0).round().astype(int)
    n_wraps_v2 = ((large_v2 - df["Large"].values) / 1024.0).round().astype(int)

    print("\nn_wraps distribution v1 vs v2 (H1 strict):")
    print(f"  {'k':<5}{'v1 count':>15}{'v2 count':>15}{'diff (v2-v1)':>18}")
    keys = sorted(set(n_wraps_v1) | set(n_wraps_v2))
    for k in keys:
        c1 = (n_wraps_v1 == k).sum()
        c2 = (n_wraps_v2 == k).sum()
        print(f"  {k:<5}{c1:>15,}{c2:>15,}{c2 - c1:>+18,}")

    diff = n_wraps_v2 - n_wraps_v1
    print(f"\nPer-row n_wraps shift (v2 vs v1):")
    for k in sorted(set(diff)):
        n = (diff == k).sum()
        print(f"  diff = {k:>+3}:  {n:>10,}  ({n / len(diff) * 100:.4f}%)")

    # Where do disagreements occur?
    disagree = diff != 0
    print(f"\nDisagreement rows: {disagree.sum():,} ({disagree.sum() / len(diff) * 100:.4f}%)")
    if disagree.sum() > 0:
        dd = df.loc[disagree].copy()
        dd["n_v1"] = n_wraps_v1[disagree]
        dd["n_v2"] = n_wraps_v2[disagree]
        dd["L_v1"] = large_v1[disagree]
        dd["L_v2"] = large_v2[disagree]
        print(f"  PHO stats: median={dd['PHO'].median():.0f}, max={dd['PHO'].max():.0f}")
        print(f"  Wide stats: median={dd['Wide'].median():.0f}, max={dd['Wide'].max():.0f}")
        print(f"  Sci stats: median={dd['Sci_1s'].median():.0f}, max={dd['Sci_1s'].max():.0f}")
        print("\n  10 disagreement examples:")
        cols = ["box", "det", "PHO", "Large", "Wide", "Sci_1s", "n_v1", "n_v2", "L_v1", "L_v2"]
        print(dd.head(10)[cols].to_string())

    # base + residual in main range
    base_v1 = ((df["PHO"].astype("float64") - large_v1) * lf - df["Wide"].astype("float64")) / L
    base_v2 = ((df["PHO"].astype("float64") - large_v2) * lf - df["Wide"].astype("float64")) / L
    sci = df["Sci_1s"].astype("float64").values
    main_v1 = (base_v1.values > 0) & (sci > 50) & (sci < 1500) & (base_v1.values < 2000)
    main_v2 = (base_v2.values > 0) & (sci > 50) & (sci < 1500) & (base_v2.values < 2000)
    rv1 = base_v1.values[main_v1] - sci[main_v1]
    rv2 = base_v2.values[main_v2] - sci[main_v2]
    print(f"\nMain-range (Sci 50-1500) residual:")
    print(f"  v1: median={np.median(rv1):+.0f}, Q25={np.quantile(rv1, 0.25):+.0f}, Q75={np.quantile(rv1, 0.75):+.0f}")
    print(f"  v2: median={np.median(rv2):+.0f}, Q25={np.quantile(rv2, 0.25):+.0f}, Q75={np.quantile(rv2, 0.75):+.0f}")


if __name__ == "__main__":
    main()
