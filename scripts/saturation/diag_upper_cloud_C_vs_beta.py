#!/usr/bin/env python3
"""Diagnose the upper cloud in C_vs_beta_diagnostic.png:
    rows where Sci_pred_base is high (> 1000 cnt/s) but Sci_obs is low (< 300 cnt/s).

User hypothesis: Large is unusually small in these rows.
Test by comparing Large, PHO, Wide, Large/PHO and Wide/PHO distributions between
upper-cloud rows and main-cloud rows.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "OOC", "Wide", "Large", "Sci_1s"]


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    base = (((df["PHO"] - df["Large"]) * lf - df["Wide"]) / L).values
    sci = df["Sci_1s"].astype("float64").values

    upper_mask = (base > 1000) & (sci < 300) & (sci > 0)
    main_mask = (base > 0) & (sci > 0) & (sci >= 300) & (sci < 2000) & (base < 3000) & ~upper_mask

    print(f"\nUpper cloud (base > 1000 AND Sci < 300): {upper_mask.sum():,} rows")
    print(f"Main cloud  (Sci 300-2000, base 0-3000):  {main_mask.sum():,} rows")
    print(f"Upper / total: {upper_mask.sum() / len(df) * 100:.2f}%")

    pho_rate = df["PHO"].values / L
    large_rate = df["Large"].values / L
    wide_rate = df["Wide"].values / L
    ooc_rate = df["OOC"].values / L
    dt_frac = df["Dt"].values / df["L_cycles"].values

    def stats(name, arr, mask):
        a = arr[mask]
        if len(a) == 0:
            return
        print(f"    {name:<14}  median={np.median(a):>10.1f}  mean={np.mean(a):>10.1f}  "
              f"Q05={np.quantile(a, 0.05):>10.1f}  Q95={np.quantile(a, 0.95):>10.1f}")

    for label, mask in [("UPPER CLOUD", upper_mask), ("MAIN CLOUD", main_mask)]:
        print(f"\n--- {label} ({mask.sum():,} rows) ---")
        stats("PHO  cnt/s", pho_rate, mask)
        stats("Large cnt/s", large_rate, mask)
        stats("Wide  cnt/s", wide_rate, mask)
        stats("OOC   cnt/s", ooc_rate, mask)
        stats("Sci_1s cnt/s", sci, mask)
        stats("base  cnt/s", base, mask)
        stats("dt_frac", dt_frac, mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_lp = np.where(pho_rate > 0, large_rate / pho_rate, np.nan)
            ratio_wp = np.where(pho_rate > 0, wide_rate / pho_rate, np.nan)
            ratio_sp = np.where(pho_rate > 0, sci / pho_rate, np.nan)
        stats("Large/PHO", ratio_lp, mask)
        stats("Wide/PHO", ratio_wp, mask)
        stats("Sci/PHO", ratio_sp, mask)

    print("\n--- UPPER CLOUD: where do they live? ---")
    upper_df = df.loc[upper_mask].copy()
    print(f"  date range: {upper_df['date'].min()} → {upper_df['date'].max()}")
    print(f"  unique dates: {upper_df['date'].nunique()}")
    print("  top 5 dates by row count:")
    for d, n in upper_df["date"].value_counts().head(5).items():
        print(f"    {d}: {n:,} rows")
    print("  (box, det) distribution:")
    print(upper_df.groupby(["box", "det"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
