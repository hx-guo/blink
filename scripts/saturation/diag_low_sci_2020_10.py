#!/usr/bin/env python3
"""Drill into the 2020-10 low-Sci cluster: which dates, same Wide-spike signature?

If 2020-10 is a continuous PSD-threshold anomaly window (like 2020-05), it should:
- have consecutive dates
- have elevated Wide/PHO ratio
- have low Sci_1s

If it's just sporadic noise (not a true anomaly), dates will be scattered.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6


def main():
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df):,} rows")

    pho = df["PHO"].astype("float64").values
    large_raw = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc

    large_corr = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0)
    base = ((pho - large_corr) * lf - wide) / L

    # 2020-10 low-Sci outliers
    low_sci_oct = (sci < 200) & (sci > 0) & (df["date"].str.startswith("2020-10"))
    print(f"\n2020-10 low-Sci rows: {low_sci_oct.sum():,}")

    # Date breakdown within 2020-10
    oct_df = df.loc[low_sci_oct].copy()
    print(f"\nDates in 2020-10 with low-Sci rows:")
    date_counts = oct_df["date"].value_counts().sort_index()
    for d, n in date_counts.items():
        print(f"  {d}: {n:>5,} rows")

    # All 2020-10 rows (low-Sci AND main) for comparison
    all_oct = df["date"].str.startswith("2020-10")
    print(f"\n2020-10 ALL rows: {all_oct.sum():,}")
    print(f"2020-10 low-Sci fraction: {low_sci_oct.sum() / max(all_oct.sum(), 1) * 100:.2f}%")

    # Wide/PHO signature
    print(f"\nWide/PHO ratio in 2020-10 outliers:")
    wp = wide[low_sci_oct] / np.maximum(pho[low_sci_oct], 1)
    print(f"  median={np.median(wp):.3f}, Q05={np.quantile(wp, 0.05):.3f}, Q95={np.quantile(wp, 0.95):.3f}")
    print(f"  rows with Wide/PHO > 0.3: {(wp > 0.3).sum():,} ({(wp > 0.3).mean()*100:.1f}%)")

    # Compare 2020-10 LOW-Sci vs 2020-10 MAIN-CLOUD to check if it's PSD anomaly or sparse low-rate
    main_oct = (sci >= 450) & (sci < 2000) & all_oct
    print(f"\n2020-10 main cloud rows: {main_oct.sum():,}")
    if main_oct.sum() > 0:
        wp_main = wide[main_oct] / np.maximum(pho[main_oct], 1)
        print(f"  Wide/PHO median in 2020-10 MAIN: {np.median(wp_main):.3f}")
        sci_main = sci[main_oct]
        print(f"  Sci_1s median in 2020-10 MAIN:   {np.median(sci_main):.0f}")
        residual_main_oct = (base[main_oct] - sci[main_oct])
        print(f"  Conservation residual in 2020-10 MAIN: median={np.median(residual_main_oct):+.0f}")

    # Same for ALL 2020-10
    residual_oct = base[low_sci_oct] - sci[low_sci_oct]
    print(f"\nConservation residual in 2020-10 LOW-Sci: median={np.median(residual_oct):+.0f}, Q25={np.quantile(residual_oct, 0.25):+.0f}, Q75={np.quantile(residual_oct, 0.75):+.0f}")

    # Check: are the 2020-10 low-Sci rows clustered around specific MET seconds (orbit-related)?
    print(f"\nMET range of 2020-10 low-Sci rows:")
    met = oct_df["met_sec"].values
    print(f"  range: {met.min()} - {met.max()}")
    print(f"  total span: {met.max() - met.min()} sec = {(met.max() - met.min()) / 86400:.1f} days")

    # Box/det
    print(f"\n(box, det) distribution in 2020-10 low-Sci:")
    print(oct_df.groupby(["box", "det"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
