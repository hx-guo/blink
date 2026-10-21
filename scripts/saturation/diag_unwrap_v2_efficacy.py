#!/usr/bin/env python3
"""Compare unwrap_large (v1, r·PHO predictor) vs unwrap_large_v2 (conservation predictor).

Reports for both algorithms:
- n_wraps distribution
- stubborn upper-cloud rows (base_corr > 1000 AND Sci_obs < 300)
- physical violations (Large_corr > PHO - Wide)
- post-correction median residual

Then verifies v2 on the specific stubborn examples from v1.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large import unwrap_large as unwrap_v1
from unwrap_large_v2 import unwrap_large_v2

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def report(name, large_corr, df, base_raw):
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    base_corr = ((df["PHO"].astype("float64") - large_corr) * lf - df["Wide"].astype("float64")) / L
    sci = df["Sci_1s"].astype("float64").values
    n_wraps = ((large_corr - df["Large"].astype("float64").values) / 1024.0).round().astype(int)

    print(f"\n=== {name} ===")
    print("  n_wraps distribution:")
    for k in sorted(set(n_wraps)):
        if k >= 0:
            print(f"    n_wraps = {k:>2}:  {(n_wraps == k).sum():>12,}  ({(n_wraps == k).sum() / len(n_wraps) * 100:>6.3f}%)")

    upper_raw = (base_raw.values > 1000) & (sci < 300) & (sci > 0)
    stubborn = upper_raw & (base_corr.values > 1000)
    print(f"  original upper cloud:    {upper_raw.sum():,} rows")
    print(f"  stubborn after correct:  {stubborn.sum():,} rows ({stubborn.sum() / max(upper_raw.sum(), 1) * 100:.2f}%)")

    # Physical violations
    over = (large_corr > (df["PHO"].astype("float64").values - df["Wide"].astype("float64").values))
    print(f"  physical violations (Large_corr > PHO - Wide): {over.sum():,}")

    # Median residual in valid main cloud (Sci 300-2000, base 0-3000)
    main = (base_corr.values > 0) & (sci > 300) & (sci < 2000) & (base_corr.values < 3000)
    resid = base_corr.values[main] - sci[main]
    print(f"  median residual in main range (Sci 300-2000): {np.median(resid):+.0f} cnt/s")
    print(f"  Q25/Q75 of main-range residual: {np.quantile(resid, 0.25):+.0f} / {np.quantile(resid, 0.75):+.0f}")
    return base_corr, stubborn


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    base_raw = ((df["PHO"].astype("float64") - df["Large"].astype("float64")) * lf - df["Wide"].astype("float64")) / L

    # v1: r·PHO predictor, per-det
    print("\nRunning v1 (r·PHO predictor) per-(box, det)...")
    large_v1 = np.zeros(len(df), dtype=np.float64)
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            large_v1[m] = unwrap_v1(df.loc[m, "PHO"].values, df.loc[m, "Large"].values)

    # v2: conservation predictor
    print("Running v2 (conservation predictor) globally (no per-det needed)...")
    large_v2 = unwrap_large_v2(
        df["PHO"].values, df["Large"].values, df["Wide"].values,
        df["Sci_1s"].values, df["L_cycles"].values, df["Dt"].values,
        C=150.0,
    )

    base_v1, stub_v1 = report("v1 (r·PHO)", large_v1, df, base_raw)
    base_v2, stub_v2 = report("v2 (conservation)", large_v2, df, base_raw)

    # Show v2 result on the rows that v1 left stubborn
    print("\n=== Rows v1 left stubborn — did v2 fix them? ===")
    v1_only = stub_v1 & ~stub_v2
    v2_only = stub_v2 & ~stub_v1
    both = stub_v1 & stub_v2
    print(f"  v1 stubborn, v2 fixed:  {v1_only.sum():,}")
    print(f"  v2 stubborn, v1 fixed:  {v2_only.sum():,}")
    print(f"  both still stubborn:    {both.sum():,}")

    # On v1-only-stubborn rows, show base before/after v2
    if v1_only.sum() > 0:
        sub = df.loc[v1_only].copy()
        sub["base_v1"] = base_v1.values[v1_only]
        sub["base_v2"] = base_v2.values[v1_only]
        sub["large_v1"] = large_v1[v1_only]
        sub["large_v2"] = large_v2[v1_only]
        print(f"\n  on v1-stubborn rows that v2 fixed: base_v1 median={sub['base_v1'].median():.0f}, base_v2 median={sub['base_v2'].median():.0f}")
        print(f"  large_v1 median={sub['large_v1'].median():.0f}, large_v2 median={sub['large_v2'].median():.0f}")
        print("\n  10 examples:")
        cols = ["box", "det", "PHO", "Large", "Wide", "Sci_1s", "large_v1", "large_v2", "base_v1", "base_v2"]
        print(sub.sort_values("base_v1", ascending=False).head(10)[cols].to_string())


if __name__ == "__main__":
    main()
