#!/usr/bin/env python3
"""Deep-dive: understand all remaining failure modes of unwrap_large_v2, then test
iteration (v3): re-fit per-det C using v2-corrected data, re-unwrap with per-det C.

Reports:
A) Conservation residual after v2 — how well does predictor match corrected Large?
B) The 15 stubborn upper-cloud rows: characterize
C) The 133 physical violations: characterize
D) Iteration: v3 = v2 with per-det C
E) Compare v3 vs v2
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

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def compute_base(df, large):
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    return ((df["PHO"].astype("float64") - large) * lf - df["Wide"].astype("float64")) / L


def fit_per_det_C(df, large, base_col_name=None):
    """Per (box, det) C = mean(base - Sci_obs) on rows where formula is valid."""
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    base = ((df["PHO"].astype("float64") - large) * lf - df["Wide"].astype("float64")) / L
    sci = df["Sci_1s"].astype("float64")
    residual_C0 = (base - sci).values
    valid = np.isfinite(residual_C0) & (sci.values > 0) & (base.values > 0)
    consts = {}
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & valid
            consts[(box, det)] = float(np.mean(residual_C0[m]))
    return consts


def unwrap_v2_per_det_C(df, consts):
    """Same as unwrap_large_v2 but uses per-det C."""
    C_arr = np.zeros(len(df))
    for (b, d), v in consts.items():
        m = ((df["box"] == b) & (df["det"] == d)).values
        C_arr[m] = v
    return unwrap_large_v2(
        df["PHO"].values, df["Large"].values, df["Wide"].values,
        df["Sci_1s"].values, df["L_cycles"].values, df["Dt"].values,
        C=C_arr,
    )


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    # ============================================================
    # A. v2 baseline (global C=150)
    # ============================================================
    print("\n" + "="*70)
    print("A. v2 baseline (global C=150)")
    print("="*70)
    large_v2 = unwrap_large_v2(
        df["PHO"].values, df["Large"].values, df["Wide"].values,
        df["Sci_1s"].values, df["L_cycles"].values, df["Dt"].values,
        C=150.0,
    )
    n_wraps_v2 = ((large_v2 - df["Large"].values) / 1024.0).round().astype(int)
    base_v2 = compute_base(df, large_v2).values
    sci = df["Sci_1s"].astype("float64").values

    # Predicted Large by conservation
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    predicted_L = (df["PHO"].astype("float64") - (df["Wide"].astype("float64") + (df["Sci_1s"].astype("float64") + 150.0) * L) / lf).values

    # Conservation residual: predicted - corrected (should be small, well within ±512)
    cons_resid = predicted_L - large_v2
    print(f"\nConservation residual (predicted_L - large_v2):")
    print(f"  median = {np.median(cons_resid):+.0f}, mean = {np.mean(cons_resid):+.0f}")
    print(f"  Q05 = {np.quantile(cons_resid, 0.05):+.0f}, Q95 = {np.quantile(cons_resid, 0.95):+.0f}")
    print(f"  abs > 512: {(np.abs(cons_resid) > 512).sum():,} rows ({(np.abs(cons_resid) > 512).mean()*100:.3f}%)")
    print(f"  abs > 1024: {(np.abs(cons_resid) > 1024).sum():,} rows  ← these mean predictor is off by >1 wrap, true error")

    # n_wraps distribution
    print("\nn_wraps distribution:")
    for k in sorted(set(n_wraps_v2)):
        if k >= 0:
            print(f"  n_wraps = {k:>2}:  {(n_wraps_v2 == k).sum():>12,}  ({(n_wraps_v2 == k).sum() / len(n_wraps_v2) * 100:>6.4f}%)")

    # ============================================================
    # B. Stubborn upper-cloud rows post-v2
    # ============================================================
    print("\n" + "="*70)
    print("B. Stubborn upper-cloud rows (base_corr > 1000 AND Sci < 300)")
    print("="*70)
    upper_orig_mask = (compute_base(df, df["Large"].astype("float64")).values > 1000) & (sci < 300) & (sci > 0)
    stubborn = upper_orig_mask & (base_v2 > 1000)
    sd = df.loc[stubborn].copy()
    sd["Large_v2"] = large_v2[stubborn]
    sd["n_wraps_v2"] = n_wraps_v2[stubborn]
    sd["base_v2"] = base_v2[stubborn]
    sd["wide_rate"] = (sd["Wide"] / (sd["L_cycles"] * L_CYCLES_TO_SEC)).astype(float)
    sd["pho_rate"] = (sd["PHO"] / (sd["L_cycles"] * L_CYCLES_TO_SEC)).astype(float)
    sd["wide_pho_ratio"] = sd["Wide"].astype(float) / sd["PHO"].astype(float)
    sd["residual_v2"] = sd["base_v2"] - sd["Sci_1s"].astype(float)
    print(f"\nStubborn count: {stubborn.sum()}")
    print(f"\nClassification:")
    flag_partile = (sd["wide_pho_ratio"] > 0.3).values
    flag_sci_zero = (sd["Sci_1s"] < 30).values
    flag_phys_viol = (sd["Large_v2"] > (sd["PHO"] - sd["Wide"])).values
    print(f"  particle event (Wide/PHO > 0.3):     {flag_partile.sum()}")
    print(f"  near-zero Sci (Sci < 30):            {flag_sci_zero.sum()}")
    print(f"  physical viol (Large > PHO-Wide):    {flag_phys_viol.sum()}")
    print(f"  classified (any flag):               {(flag_partile | flag_sci_zero | flag_phys_viol).sum()}")
    print(f"  unclassified:                        {(~(flag_partile | flag_sci_zero | flag_phys_viol)).sum()}")
    print("\n10 examples (sorted by base_v2 desc):")
    cols = ["box", "det", "PHO", "Large", "Large_v2", "n_wraps_v2", "Wide", "wide_pho_ratio", "Sci_1s", "base_v2", "residual_v2"]
    print(sd.sort_values("base_v2", ascending=False).head(10)[cols].to_string())

    # ============================================================
    # C. Physical violations (Large_v2 > PHO - Wide)
    # ============================================================
    print("\n" + "="*70)
    print("C. Physical violations (Large_v2 > PHO - Wide)")
    print("="*70)
    viol = large_v2 > (df["PHO"].astype("float64").values - df["Wide"].astype("float64").values)
    vd = df.loc[viol].copy()
    vd["Large_v2"] = large_v2[viol]
    vd["n_wraps_v2"] = n_wraps_v2[viol]
    vd["overshoot"] = vd["Large_v2"] - (vd["PHO"] - vd["Wide"])
    vd["wide_pho"] = vd["Wide"].astype(float) / vd["PHO"].astype(float)
    print(f"\nCount: {viol.sum()} ({viol.sum()/len(df)*100:.4f}% of all rows)")
    print(f"\nDistribution of n_wraps_v2 in violators:")
    nw_v = vd["n_wraps_v2"].values
    for k in sorted(set(nw_v)):
        print(f"  n_wraps = {k:>2}:  {(nw_v == k).sum():>4}")
    print(f"\nOvershoot stats (Large_v2 - (PHO - Wide)):")
    print(f"  median = {vd['overshoot'].median():.0f}, max = {vd['overshoot'].max():.0f}")
    print(f"\nWide/PHO ratio in violators (high = particle event):")
    print(f"  median = {vd['wide_pho'].median():.3f}, Q95 = {vd['wide_pho'].quantile(0.95):.3f}")
    print("\n10 examples:")
    cols = ["box", "det", "PHO", "Large", "Large_v2", "Wide", "wide_pho", "Sci_1s", "overshoot"]
    print(vd.sort_values("overshoot", ascending=False).head(10)[cols].to_string())

    # ============================================================
    # D. Iteration: re-fit per-det C from v2-corrected, then re-unwrap
    # ============================================================
    print("\n" + "="*70)
    print("D. Iteration (v3): re-fit per-det C using v2 data, re-unwrap")
    print("="*70)
    # Only fit C on rows that are clean: not particle events, not stubborn
    is_clean = (~stubborn) & (df["Wide"].astype(float) / df["PHO"].astype(float) < 0.2).values
    df_clean = df.loc[is_clean].copy()
    large_v2_clean = large_v2[is_clean]
    consts_v3 = fit_per_det_C(df_clean, large_v2_clean)
    print("\nPer-det C (from v2-corrected clean rows):")
    for box in "ABC":
        line = "  " + "  ".join(f"{box}-{d}: {consts_v3[(box, d)]:+6.1f}" for d in range(6))
        print(line)
    c_vals = list(consts_v3.values())
    print(f"  range: [{min(c_vals):+.1f}, {max(c_vals):+.1f}], mean: {np.mean(c_vals):+.1f}")

    large_v3 = unwrap_v2_per_det_C(df, consts_v3)
    n_wraps_v3 = ((large_v3 - df["Large"].values) / 1024.0).round().astype(int)
    base_v3 = compute_base(df, large_v3).values

    upper_orig = upper_orig_mask
    stubborn_v3 = upper_orig & (base_v3 > 1000)
    viol_v3 = large_v3 > (df["PHO"].astype("float64").values - df["Wide"].astype("float64").values)
    print(f"\nv3 results:")
    print(f"  Stubborn (was 15 in v2):    {stubborn_v3.sum()}")
    print(f"  Physical viol (was 133):    {viol_v3.sum()}")
    print(f"\nn_wraps distribution v3:")
    for k in sorted(set(n_wraps_v3)):
        if k >= 0:
            print(f"  n_wraps = {k:>2}:  {(n_wraps_v3 == k).sum():>12,}  ({(n_wraps_v3 == k).sum() / len(n_wraps_v3) * 100:>6.4f}%)")

    # Same-row diff between v2 and v3
    diff = n_wraps_v3 - n_wraps_v2
    print(f"\nv3 vs v2 n_wraps shift:")
    for k in sorted(set(diff)):
        if (diff == k).sum() > 0:
            print(f"  diff = {k:>+3}:  {(diff == k).sum():>10,}")

    # Residual in main range
    main = (base_v3 > 0) & (sci > 300) & (sci < 2000) & (base_v3 < 3000)
    resid_v3 = base_v3[main] - sci[main]
    main2 = (base_v2 > 0) & (sci > 300) & (sci < 2000) & (base_v2 < 3000)
    resid_v2 = base_v2[main2] - sci[main2]
    print(f"\nMain-range (Sci 300-2000) residual:")
    print(f"  v2: median={np.median(resid_v2):+.0f}, Q25={np.quantile(resid_v2, 0.25):+.0f}, Q75={np.quantile(resid_v2, 0.75):+.0f}")
    print(f"  v3: median={np.median(resid_v3):+.0f}, Q25={np.quantile(resid_v3, 0.25):+.0f}, Q75={np.quantile(resid_v3, 0.75):+.0f}")


if __name__ == "__main__":
    main()
