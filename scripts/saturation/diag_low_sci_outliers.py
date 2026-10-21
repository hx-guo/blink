#!/usr/bin/env python3
"""Diagnose the low-Sci outlier population (Sci < 200) in the 2020 relaxed sample.

These rows are statistically isolated from the main cloud (gap between Sci=165 and Sci=448).
If they're a real low-rate physical population, residual should follow whatever model fits.
If they're an artifact (threshold adjustment, blind detector, mode change), residual is
unreliable — and our prior "additive C ≈ 150" conclusion needs revisiting.

Hypotheses to test:
A) Specific date window (PSD threshold adjustment beyond 2020-05?)
B) Specific (box, det) (Box C blind detector?)
C) Elevated Wide/PHO ratio (CsI counter saturating or threshold-shifted)
D) Specific HV state
E) Specific orbital position
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
PSD_START, PSD_END = "2020-04-30", "2020-05-31"


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  rows: {len(df):,}, cols: {len(df.columns)}")
    print(f"  Available cols: {df.columns.tolist()}")

    # NOTE: we INCLUDE 2020-05 here on purpose — to see if known anomaly month shows same pattern
    full_df = df.copy()

    # Define outlier vs main using unwrapped base
    pho = full_df["PHO"].astype("float64").values
    large_raw = full_df["Large"].astype("float64").values
    wide = full_df["Wide"].astype("float64").values
    sci = full_df["Sci_1s"].astype("float64").values
    lc = full_df["L_cycles"].astype("float64").values
    dtv = full_df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc

    large_corr = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0)
    base = ((pho - large_corr) * lf - wide) / L

    # Outlier criterion: low Sci AND base > 0 (valid)
    low_sci = (sci > 0) & (sci < 200) & (base > 0)
    main_cloud = (sci >= 450) & (sci < 2000) & (base > 0) & (base < 3000)

    print(f"\n=== Group sizes ===")
    print(f"  Low-Sci outliers (Sci < 200):       {low_sci.sum():>12,}  ({low_sci.sum()/len(full_df)*100:.3f}%)")
    print(f"  Main cloud (Sci 450-2000):           {main_cloud.sum():>12,}  ({main_cloud.sum()/len(full_df)*100:.3f}%)")
    print(f"  Between-band (Sci 200-450):          {((sci >= 200) & (sci < 450)).sum():>12,}")

    print(f"\n=== Date distribution of low-Sci outliers ===")
    low_df = full_df.loc[low_sci].copy()
    print(f"  Unique dates: {low_df['date'].nunique()}")
    monthly = low_df.groupby(low_df["date"].str[:7]).size().sort_values(ascending=False).head(20)
    print(f"  Top 20 months by row count:")
    for m, n in monthly.items():
        print(f"    {m}: {n:>6,} rows")

    print(f"\n  PSD anomaly month (2020-05) explicit count:")
    psd_anom = ((low_df["date"] >= "2020-04-30") & (low_df["date"] <= "2020-05-31")).sum()
    print(f"    2020-05 PSD anomaly: {psd_anom:,} rows ({psd_anom/low_sci.sum()*100:.1f}% of outliers)")

    print(f"\n=== (box, det) distribution ===")
    print("  Low-Sci outliers:")
    print(low_df.groupby(["box", "det"]).size().unstack(fill_value=0).to_string())

    # Per-(box, det) main cloud count for comparison
    main_df = full_df.loc[main_cloud]
    print("\n  Main cloud (for normalization):")
    print(main_df.groupby(["box", "det"]).size().unstack(fill_value=0).to_string())

    # Outlier rate (per detector): outliers / main_cloud
    print("\n  Outlier fraction per (box, det) (low_sci / main_cloud):")
    out_pd = low_df.groupby(["box", "det"]).size().unstack(fill_value=0)
    main_pd = main_df.groupby(["box", "det"]).size().unstack(fill_value=0)
    ratio = (out_pd / main_pd * 100).round(3)
    print(ratio.to_string())

    print(f"\n=== Wide/PHO ratio in outliers vs main ===")
    wide_pho_low = (full_df["Wide"].astype(float).values / pho)[low_sci & (pho > 0)]
    wide_pho_main = (full_df["Wide"].astype(float).values / pho)[main_cloud & (pho > 0)]
    print(f"  Low-Sci:    median={np.median(wide_pho_low):.4f}, Q05={np.quantile(wide_pho_low, 0.05):.4f}, Q95={np.quantile(wide_pho_low, 0.95):.4f}")
    print(f"  Main cloud: median={np.median(wide_pho_main):.4f}, Q05={np.quantile(wide_pho_main, 0.05):.4f}, Q95={np.quantile(wide_pho_main, 0.95):.4f}")

    if (full_df["Wide"].astype(float).values[low_sci] > 1000).any():
        n_high_wide = (full_df["Wide"].astype(float).values[low_sci] > 1000).sum()
        print(f"\n  Outliers with Wide > 1000 (extreme CsI activity): {n_high_wide:,} ({n_high_wide/low_sci.sum()*100:.1f}%)")

    print(f"\n=== HV in outliers vs main ===")
    if "HV" in full_df.columns:
        hv_low = full_df["HV"].values[low_sci]
        hv_main = full_df["HV"].values[main_cloud]
        print(f"  Low-Sci:    median={np.median(hv_low):.0f}, Q05={np.quantile(hv_low, 0.05):.0f}, Q95={np.quantile(hv_low, 0.95):.0f}")
        print(f"  Main cloud: median={np.median(hv_main):.0f}, Q05={np.quantile(hv_main, 0.05):.0f}, Q95={np.quantile(hv_main, 0.95):.0f}")

    if "Lat" in full_df.columns:
        print(f"\n=== Latitude distribution ===")
        lat_low = np.abs(full_df["Lat"].values[low_sci])
        lat_main = np.abs(full_df["Lat"].values[main_cloud])
        print(f"  |Lat| Low-Sci:    median={np.median(lat_low):.1f}, Q05={np.quantile(lat_low, 0.05):.1f}, Q95={np.quantile(lat_low, 0.95):.1f}")
        print(f"  |Lat| Main:       median={np.median(lat_main):.1f}, Q05={np.quantile(lat_main, 0.05):.1f}, Q95={np.quantile(lat_main, 0.95):.1f}")

    print(f"\n=== Counter rates summary ===")
    pho_rate = pho / L
    large_rate = large_corr / L  # use unwrapped Large
    wide_rate = wide / L
    sci_rate = sci  # already cnt/s

    for label, m in [("Low-Sci", low_sci), ("Main cloud", main_cloud)]:
        print(f"\n  {label} ({m.sum():,} rows):")
        for name, arr in [("PHO rate", pho_rate), ("Large rate (unwrap)", large_rate),
                           ("Wide rate", wide_rate), ("Sci_1s rate", sci_rate)]:
            v = arr[m]
            print(f"    {name:<22}  median={np.median(v):>8.0f}  Q05={np.quantile(v, 0.05):>8.0f}  Q95={np.quantile(v, 0.95):>8.0f}")

    # Check: is the conservation residual in outliers significantly different from main?
    residual = base - sci
    print(f"\n=== Conservation residual ===")
    for label, m in [("Low-Sci", low_sci), ("Main cloud", main_cloud)]:
        r = residual[m]
        print(f"  {label}: median={np.median(r):+.0f}, Q25={np.quantile(r, 0.25):+.0f}, Q75={np.quantile(r, 0.75):+.0f}")


if __name__ == "__main__":
    main()
