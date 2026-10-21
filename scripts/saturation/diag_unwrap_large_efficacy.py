#!/usr/bin/env python3
"""Test if scripts/unwrap_large.py correctly recovers wrapped Large counts on the
2020 relaxed sample, focusing on the "upper cloud" rows (base_raw>1000, Sci_obs<300).

Reports per-detector:
- r_cal: calibrated Large/PHO ratio
- n_wraps distribution for upper-cloud rows
- base BEFORE vs AFTER unwrap
- "still anomalous" count (base_corrected still > 1000)
- physical sanity violations (real_Large > PHO - Wide)
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Wide", "Large", "Sci_1s"]


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")

    # Apply unwrap per-(box, det)
    large_corrected = np.zeros(len(df), dtype=np.float64)
    r_cals = {}
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            pho_d = df.loc[m, "PHO"].values
            large_d = df.loc[m, "Large"].values
            # Replicate calibration to log r
            low = (pho_d > 200) & (pho_d < 2500) & (large_d < 900)
            r = float(np.median(large_d[low] / pho_d[low])) if low.sum() >= 20 else 0.3
            r_cals[(box, det)] = r
            large_corrected[m] = unwrap_large(pho_d, large_d)

    df["Large_raw"] = df["Large"].astype("float64")
    df["Large_corr"] = large_corrected
    df["n_wraps"] = ((df["Large_corr"] - df["Large_raw"]) / 1024).round().astype(int)

    base_raw = ((df["PHO"] - df["Large_raw"]) * lf - df["Wide"]) / L
    base_corr = ((df["PHO"] - df["Large_corr"]) * lf - df["Wide"]) / L
    df["base_raw"] = base_raw.values
    df["base_corr"] = base_corr.values
    df["sci_obs"] = df["Sci_1s"].astype("float64")

    print("\n=== r_cal per detector ===")
    for box in "ABC":
        line = "  " + "  ".join(f"{box}-{d}: {r_cals[(box, d)]:.3f}" for d in range(6))
        print(line)
    r_vals = list(r_cals.values())
    print(f"  range: [{min(r_vals):.3f}, {max(r_vals):.3f}], mean: {np.mean(r_vals):.3f}")

    # Global n_wraps distribution
    print("\n=== n_wraps distribution (ALL rows) ===")
    nw = df["n_wraps"].values
    for k in sorted(set(nw)):
        print(f"  n_wraps = {k:>2}:  {(nw == k).sum():>12,}  ({(nw == k).sum() / len(nw) * 100:>6.3f}%)")

    # Upper cloud
    upper = (df["base_raw"] > 1000) & (df["sci_obs"] < 300) & (df["sci_obs"] > 0)
    upper_df = df.loc[upper].copy()
    print(f"\n=== UPPER CLOUD ({upper.sum():,} rows) ===")
    print(f"  n_wraps distribution:")
    nw_u = upper_df["n_wraps"].values
    for k in sorted(set(nw_u)):
        print(f"    n_wraps = {k:>2}:  {(nw_u == k).sum():>6,}  ({(nw_u == k).sum() / len(nw_u) * 100:>6.2f}%)")
    print(f"\n  base BEFORE unwrap: median={np.median(upper_df['base_raw']):.0f}  Q25={np.quantile(upper_df['base_raw'], 0.25):.0f}  Q75={np.quantile(upper_df['base_raw'], 0.75):.0f}")
    print(f"  base AFTER  unwrap: median={np.median(upper_df['base_corr']):.0f}  Q25={np.quantile(upper_df['base_corr'], 0.25):.0f}  Q75={np.quantile(upper_df['base_corr'], 0.75):.0f}")
    print(f"  Sci_obs in upper cloud: median={np.median(upper_df['sci_obs']):.0f}")

    # How many upper-cloud rows return to "reasonable" base after correction?
    fixed = (upper_df["base_corr"] < 1000).sum()
    still_anomalous = (upper_df["base_corr"] > 1000).sum()
    print(f"\n  After unwrap:")
    print(f"    base_corr < 1000 (likely fixed):    {fixed:>5,}  ({fixed / len(upper_df) * 100:.1f}%)")
    print(f"    base_corr > 1000 (still anomalous): {still_anomalous:>5,}  ({still_anomalous / len(upper_df) * 100:.1f}%)")

    # Distance from Sci (perfect fix = base_corr ≈ sci + ~150)
    residual_corr = upper_df["base_corr"] - upper_df["sci_obs"]
    print(f"  residual after unwrap (base_corr - Sci): median={np.median(residual_corr):+.0f}, Q25={np.quantile(residual_corr, 0.25):+.0f}, Q75={np.quantile(residual_corr, 0.75):+.0f}")
    print(f"  (expected ~+150 from H1 strict C)")

    # Physical sanity: real_Large should not exceed PHO - Wide
    over_limit = (df["Large_corr"] > df["PHO"] - df["Wide"]).sum()
    print(f"\n=== Physical sanity check (Large_corr > PHO − Wide is unphysical) ===")
    print(f"  rows violating: {over_limit:>10,} / {len(df):,}  ({over_limit / len(df) * 100:.4f}%)")
    if over_limit > 0:
        violators = df.loc[df["Large_corr"] > df["PHO"] - df["Wide"]].copy()
        violators["overshoot"] = violators["Large_corr"] - (violators["PHO"] - violators["Wide"])
        print(f"  overshoot: median={violators['overshoot'].median():.0f}, max={violators['overshoot'].max():.0f}")

    # Also: did unwrap introduce a NEW upper cloud? (rows that were fine but got over-corrected)
    main_before = (df["base_raw"] > 300) & (df["base_raw"] < 2000) & (df["sci_obs"] > 300) & (df["sci_obs"] < 2000)
    bad_after_main = main_before & (df["base_corr"] < 0)  # over-corrected to negative
    print(f"\n=== Over-correction check (rows that were 'main' but became base_corr < 0) ===")
    print(f"  count: {bad_after_main.sum():>10,}  ({bad_after_main.sum() / main_before.sum() * 100:.4f}% of main)")

    # Show how upper cloud n_wraps correlates with PHO (high PHO → more wraps expected)
    print(f"\n=== n_wraps vs PHO in upper cloud ===")
    print(f"  {'n_wraps':<10}{'count':>8}{'PHO median':>14}{'PHO Q90':>10}{'Wide median':>14}")
    for k in sorted(set(nw_u)):
        if k == 0:
            continue
        m = nw_u == k
        sub = upper_df.loc[m]
        print(f"  {k:<10}{len(sub):>8}{sub['PHO'].median():>14.0f}{sub['PHO'].quantile(0.9):>10.0f}{sub['Wide'].median():>14.0f}")


if __name__ == "__main__":
    main()
