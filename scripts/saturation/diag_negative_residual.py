#!/usr/bin/env python3
"""Diagnose the negative-residual cloud in the AFTER-model plot.

User observation: there's a sharp 'cut' of rows below the main cloud at residual
~-300 to -500. Symmetric to upper wrap cloud — looks like over-correction.

For each negative-residual row:
- What did v2 assign as n_wraps?
- What did the predictor predict vs observed Large?
- Would n_wraps-1 give residual closer to 0?
"""
from __future__ import annotations

import sys
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import aacgmv2

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]


def main():
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"Loaded {len(df):,} rows")

    # Compute mlat via interp grid
    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                      bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)

    pho = df["PHO"].values; large_raw = df["Large"].values
    wide = df["Wide"].values; sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values; dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    large_corr, conf = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0,
                                        return_confidence=True)
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L

    # C_det per-det (from |mlat|<5°)
    is_clean = ((conf > CONF_LOW) & (wide / np.maximum(pho, 1) < 0.3) & (sci > 100)
                & np.isfinite((base - sci).values if hasattr(base, "values") else (base - sci))
                & ~np.isnan(mlat))
    is_eq = is_clean & (abs_mlat < 5)
    residual_raw = (base - sci).values if hasattr(base, "values") else (base - sci)
    C_det_per_row = np.zeros(len(df))
    for box in "ABC":
        for det in range(6):
            m_dt = ((df["box"] == box) & (df["det"] == det)).values
            m = m_dt & is_eq
            cval = float(np.mean(residual_raw[m])) if m.sum() > 100 else 120.0
            C_det_per_row[m_dt] = cval

    # B(|mlat|)
    B_per_row = 0.26 * np.maximum(0, abs_mlat - 20)**2
    B_per_row[np.isnan(B_per_row)] = 0.0

    residual_clean = residual_raw - C_det_per_row - B_per_row

    # The predictor used in unwrap_v2: predicted_Large = PHO - (Wide + (Sci+C)*L)/(1-dt)
    predicted_large = pho.astype("float64") - (wide.astype("float64") + (sci + 150.0) * L) / lf

    # Negative residual cloud: residual_clean < -200
    is_negative = is_clean & (residual_clean < -200) & (residual_clean > -800)
    print(f"\nNegative-residual rows ({-800} < resid_clean < {-200}): {is_negative.sum():,}")

    if is_negative.sum() == 0:
        print("None found — no over-correction visible at this threshold.")
        return

    nd = df.loc[is_negative].copy()
    nd["residual_clean"] = residual_clean[is_negative]
    nd["large_corr"] = large_corr[is_negative]
    nd["n_wraps"] = n_wraps[is_negative]
    nd["predicted_large"] = predicted_large[is_negative]
    nd["abs_mlat"] = abs_mlat[is_negative]
    nd["wide_pho"] = wide[is_negative] / np.maximum(pho[is_negative], 1)
    nd["base"] = base.values[is_negative] if hasattr(base, "values") else base[is_negative]

    print(f"\n=== Summary of negative-residual rows ===")
    print(f"  n_wraps distribution:")
    for k in sorted(set(nd['n_wraps'])):
        cnt = (nd['n_wraps'] == k).sum()
        print(f"    n_wraps={k}: {cnt:>6,} ({cnt/len(nd)*100:.1f}%)")
    print(f"\n  Stats:")
    print(f"    residual_clean: median={nd['residual_clean'].median():.1f}, range [{nd['residual_clean'].min():.0f}, {nd['residual_clean'].max():.0f}]")
    print(f"    Sci_1s: median={nd['Sci_1s'].median():.0f}")
    print(f"    PHO: median={nd['PHO'].median():.0f}")
    print(f"    Large raw: median={nd['Large'].median():.0f}, large_corr: median={nd['large_corr'].median():.0f}")
    print(f"    Wide: median={nd['Wide'].median():.0f}, Wide/PHO: median={nd['wide_pho'].median():.3f}")
    print(f"    |mlat|: median={nd['abs_mlat'].median():.1f}")
    print(f"    predicted_large: median={nd['predicted_large'].median():.0f}")
    print(f"    Diff predicted-large_raw: median={(nd['predicted_large']-nd['Large']).median():.0f}")
    print(f"    Diff predicted-large_corr (post-unwrap): median={(nd['predicted_large']-nd['large_corr']).median():.0f}")

    # If we removed 1 wrap from large_corr, what would residual be?
    nd["large_corr_minus1"] = np.maximum(nd["large_corr"] - 1024, nd["Large"])
    base_minus1 = ((nd["PHO"].astype("float64") - nd["large_corr_minus1"]) *
                   (1 - nd["Dt"].astype("float64") / nd["L_cycles"].astype("float64"))
                   - nd["Wide"].astype("float64")) / (nd["L_cycles"] * L_CYCLES_TO_SEC)
    C_det_neg = C_det_per_row[is_negative]
    B_neg = B_per_row[is_negative]
    residual_clean_minus1 = base_minus1 - nd["Sci_1s"] - C_det_neg - B_neg
    print(f"\n  If we REMOVE 1 wrap from these rows (test under-correction hypothesis):")
    print(f"    residual_clean_minus1: median={residual_clean_minus1.median():.0f}, range [{residual_clean_minus1.min():.0f}, {residual_clean_minus1.max():.0f}]")
    print(f"    Closer to 0? {abs(residual_clean_minus1.median()) < abs(nd['residual_clean'].median())}")

    # 10 example rows
    print(f"\n=== 10 example rows (sorted by most negative residual) ===")
    cols = ["box", "det", "PHO", "Large", "large_corr", "n_wraps", "Wide", "Sci_1s", "predicted_large", "residual_clean"]
    print(nd.sort_values("residual_clean").head(10)[cols].to_string())


if __name__ == "__main__":
    main()
