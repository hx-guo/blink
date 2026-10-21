#!/usr/bin/env python3
"""v6: Decompose C_det into electronic floor R_det and CR sensitivity s_det.

Model:
    C(det, |mlat|) = R_det + s_det · [1 + k · max(0, |mlat|-20)²]

Per-detector OLS fit of y = R + s·u, where
    u_i = 1 + k · max(0, |mlat_i| - 20)²
    y_i = resid_v2_i  (= base_v2 - Sci_1s, what C is supposed to model)

Strategy:
  1. Reuse v5's v2 unwrap to get resid_v2 = base_v2 - Sci_obs
  2. Fix k = k_v5 (from v5 unified fit). Per-det OLS for (R_det, s_det).
  3. Report R, s, σR, σs, plus implied C_eq = R + s for comparison to v5's s_det.
  4. Optional: iterate k once given new R/s. Check whether k changes.

Expected: 17 dets with R≈0, B-2 with R≈58, all dets with s ≈ similar magnitude.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0


def fit_R_s_per_det(y, u, box_arr, det_arr, weight_mask):
    """Per-det OLS: y = R + s·u.

    Returns dict[(box,det)] -> (R, s, sigma_R, sigma_s, n)
    """
    out = {}
    for box in "ABC":
        for det in range(6):
            m = (box_arr == box) & (det_arr == det) & weight_mask
            n = int(m.sum())
            if n < 200:
                out[(box, det)] = (np.nan, np.nan, np.nan, np.nan, n)
                continue
            xi = u[m]
            yi = y[m]
            # OLS y = R + s·x. Closed-form.
            xbar = xi.mean()
            ybar = yi.mean()
            Sxx = ((xi - xbar) ** 2).sum()
            Sxy = ((xi - xbar) * (yi - ybar)).sum()
            s = Sxy / Sxx
            R = ybar - s * xbar
            # residual variance
            yhat = R + s * xi
            sse = ((yi - yhat) ** 2).sum()
            var = sse / (n - 2)
            sigma_s = np.sqrt(var / Sxx)
            sigma_R = np.sqrt(var * (1.0 / n + xbar ** 2 / Sxx))
            out[(box, det)] = (float(R), float(s), float(sigma_R), float(sigma_s), n)
    return out


def fit_k_global(y, R_per_row, s_per_row, abs_mlat, weight_mask):
    """Fix R, s per row. Fit k via OLS on:
        y - R = s · (1 + k · w²),  w = max(0, |mlat|-20)
    →   (y - R)/s - 1 = k · w²
    Robust median over a robust mask (|mlat| >= 25, w² >= 25).
    """
    w2 = np.maximum(0.0, abs_mlat - B_THRESHOLD) ** 2
    m = weight_mask & (abs_mlat >= 25) & (s_per_row > 1e-3)
    target = (y[m] - R_per_row[m]) / s_per_row[m] - 1.0
    w2m = w2[m]
    # k = sum(w² · target) / sum(w⁴)   (least squares through origin)
    k = float((w2m * target).sum() / (w2m * w2m).sum())
    # median-based estimate (robust check)
    k_med = float(np.median(target / np.maximum(w2m, 1.0)))
    return k, k_med


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  rows: {len(df):,}")

    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]),
        grid["mlat"],
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)
    abs_mlat_safe = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)

    pho = df["PHO"].astype("float64").values
    large_raw = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc
    box_arr = df["box"].values
    det_arr = df["det"].values

    # --- v2 unwrap with C=150 to get resid_v2 (same as v5 step 0) ---
    print("\n[1/4] v2 unwrap with C=150...")
    large_v2, _ = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=150.0, return_confidence=True
    )
    base_v2 = (pho - large_v2) * lf / L - wide / L
    resid_v2 = base_v2 - sci

    # Clean mask
    is_clean = (
        (wide / np.maximum(pho, 1) < 0.3)
        & (sci > 100)
        & np.isfinite(resid_v2)
        & ~np.isnan(mlat)
        & (np.abs(resid_v2) < 2000)  # cap obvious outliers
    )
    print(f"  clean rows: {is_clean.sum():,}")

    # --- Initial k from v5 unified fit ---
    # v5 reports k ≈ 0.00188 — use that. Will refit iteratively.
    k = 0.00188
    print(f"\n[2/4] Fix initial k = {k:.5f} (from v5). Per-det fit (R, s)...")

    w2 = np.maximum(0.0, abs_mlat_safe - B_THRESHOLD) ** 2
    u = 1.0 + k * w2

    fit1 = fit_R_s_per_det(resid_v2, u, box_arr, det_arr, is_clean)

    print("\n  --- Iteration 1: k fixed at v5 value ---")
    print(f"  {'det':>5}  {'R':>7} ± {'σR':<5}  {'s':>6} ± {'σs':<4}  {'R+s (=v5 s_det)':>16}  {'N':>9}")
    R_per_row = np.zeros(len(df))
    s_per_row = np.zeros(len(df))
    for box in "ABC":
        for det in range(6):
            R, s, sR, sS, n = fit1[(box, det)]
            tag = f"{box}-{det}"
            print(
                f"  {tag:>5}  {R:>7.1f} ± {sR:<5.1f}  {s:>6.1f} ± {sS:<4.1f}  {R+s:>16.1f}  {n:>9,}"
            )
            m = (box_arr == box) & (det_arr == det)
            R_per_row[m] = R
            s_per_row[m] = s

    # --- Iterate k given new R, s ---
    print("\n[3/4] Refit global k given new (R_det, s_det)...")
    k_ols, k_med = fit_k_global(resid_v2, R_per_row, s_per_row, abs_mlat_safe, is_clean)
    print(f"  k_OLS    = {k_ols:.5f}")
    print(f"  k_median = {k_med:.5f}")

    # --- Iteration 2 with updated k ---
    k2 = k_ols
    print(f"\n[4/4] Iteration 2 with k = {k2:.5f}...")
    u2 = 1.0 + k2 * w2
    fit2 = fit_R_s_per_det(resid_v2, u2, box_arr, det_arr, is_clean)

    print(f"\n  --- Iteration 2: k = {k2:.5f} ---")
    print(f"  {'det':>5}  {'R':>7} ± {'σR':<5}  {'s':>6} ± {'σs':<4}  {'R+s':>7}  {'N':>9}")
    R2_per_row = np.zeros(len(df))
    s2_per_row = np.zeros(len(df))
    for box in "ABC":
        for det in range(6):
            R, s, sR, sS, n = fit2[(box, det)]
            tag = f"{box}-{det}"
            flag = "  *" if abs(R) > 3 * max(sR, 1) else ""
            print(
                f"  {tag:>5}  {R:>7.1f} ± {sR:<5.1f}  {s:>6.1f} ± {sS:<4.1f}  {R+s:>7.1f}  {n:>9,}{flag}"
            )
            m = (box_arr == box) & (det_arr == det)
            R2_per_row[m] = R
            s2_per_row[m] = s

    # --- Summary: which dets have R significantly nonzero? ---
    print("\n  Detectors with |R| > 3σR (significant electronic floor):")
    sig_dets = []
    for box in "ABC":
        for det in range(6):
            R, s, sR, sS, n = fit2[(box, det)]
            if not np.isfinite(R):
                continue
            if abs(R) > 3 * max(sR, 1):
                sig_dets.append((box, det, R, sR, s))
                print(f"    {box}-{det}: R = {R:+.1f} ± {sR:.1f}  ({abs(R)/max(sR,1):.1f}σ),  s = {s:.1f}")
    if not sig_dets:
        print("    (none)")

    # --- Model evaluation: C_v6_per_row, residual_v6 ---
    C_v6 = R2_per_row + s2_per_row * u2
    # v5 equivalent for comparison: s_v5 = R+s (treated as single C_eq), no R, k same
    s_v5_per_row = R2_per_row + s2_per_row  # what v5 fit would have given (mean at eq)
    C_v5 = s_v5_per_row * u2  # WRONG model: factor of u over (R+s)

    print("\n  --- Model comparison on clean rows ---")
    is_eval = is_clean & (abs_mlat_safe >= 30)
    if is_eval.sum() > 1000:
        resid_v6 = (resid_v2 - C_v6)[is_eval]
        resid_v5 = (resid_v2 - C_v5)[is_eval]
        print(f"  High |mlat| (>=30°), N = {is_eval.sum():,}:")
        print(f"    v5 model:  median = {np.median(resid_v5):+.2f}, MAD = {np.median(np.abs(resid_v5-np.median(resid_v5))):.2f}")
        print(f"    v6 model:  median = {np.median(resid_v6):+.2f}, MAD = {np.median(np.abs(resid_v6-np.median(resid_v6))):.2f}")

    is_b2 = is_clean & (box_arr == "B") & (det_arr == 2)
    is_b2_eq = is_b2 & (abs_mlat_safe < 10)
    is_b2_hi = is_b2 & (abs_mlat_safe >= 30)
    print(f"\n  B-2 only:")
    print(f"    equatorial (|mlat|<10): N={is_b2_eq.sum():,}, mean resid_v2 = {resid_v2[is_b2_eq].mean():.1f}")
    print(f"    high |mlat| (>=30):     N={is_b2_hi.sum():,}, mean resid_v2 = {resid_v2[is_b2_hi].mean():.1f}")
    R_b2, s_b2, _, _, _ = fit2[("B", 2)]
    print(f"    v6 fit:  R = {R_b2:.1f},  s = {s_b2:.1f},  s/(R+s) = {s_b2/(R_b2+s_b2)*100:.0f}% (CR fraction at eq)")


if __name__ == "__main__":
    main()
