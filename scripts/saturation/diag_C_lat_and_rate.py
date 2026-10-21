#!/usr/bin/env python3
"""Verify two pending claims:

1. C(Lat) — does C increase with |Lat|?  (Tests cosmic ray secondary hypothesis
   that explains the H1 strict +120 vs relaxed +195 gap of ~75 cnt/s)

2. Residual rate dependence — after iterated unwrap + per-det C subtraction,
   is the residual flat at zero (pure additive) or has a k·Sci slope?

Workflow:
- Load relaxed cache
- Iteratively converge unwrap + per-det C (we've done this; C ≈ +195 mean)
- Filter Wide/PHO < 0.3 (drop magnetar mode) AND HIGH confidence only
- Test 1: bin by |Lat|, fit C per band
- Test 2: per (box, det), compute residual = base - Sci - C(box,det), then fit residual = a + k·Sci
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6

C_INIT = 150.0
MAX_ITER = 6
TOL = 0.5

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat"]


def iterate_unwrap(df):
    """Run iterated unwrap + per-det C estimation. Returns large_corr, conf, C_per_det."""
    pho = df["PHO"].values
    large_raw = df["Large"].values
    wide = df["Wide"].values
    sci = df["Sci_1s"].values
    lc = df["L_cycles"].values
    dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    C_per_row = np.full(len(df), C_INIT)
    last_C_per_det = None
    for it in range(MAX_ITER):
        large_corr, conf = unwrap_large_v2(
            pho, large_raw, wide, sci, lc, dtv,
            C=C_per_row, return_confidence=True,
        )
        base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
        residual = (base - sci.astype("float64")).values if hasattr(base, "values") else (base - sci.astype("float64"))
        wide_pho = wide / np.maximum(pho, 1)
        is_clean = (conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100) & np.isfinite(residual)
        consts = {}
        for box in "ABC":
            for det in range(6):
                m = ((df["box"] == box) & (df["det"] == det)).values & is_clean
                consts[(box, det)] = float(np.mean(residual[m])) if m.sum() > 100 else 150.0
        if last_C_per_det is not None:
            max_change = max(abs(consts[k] - last_C_per_det[k]) for k in consts)
            if max_change < TOL:
                break
        last_C_per_det = consts
        C_per_row = np.full(len(df), 150.0)
        for (b, d), v in consts.items():
            m = ((df["box"] == b) & (df["det"] == d)).values
            C_per_row[m] = v
    return large_corr, conf, consts, residual, base


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    print("Running iterated unwrap...")
    large_corr, conf, C_per_det, residual, base = iterate_unwrap(df)
    print(f"  Converged per-det C: mean={np.mean(list(C_per_det.values())):.1f}, "
          f"range=[{min(C_per_det.values()):.0f}, {max(C_per_det.values()):.0f}]")

    pho = df["PHO"].values
    wide = df["Wide"].values
    sci = df["Sci_1s"].values.astype("float64")
    wide_pho = wide / np.maximum(pho, 1)
    is_clean = (conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100) & np.isfinite(residual)
    print(f"  Clean rows for analysis: {is_clean.sum():,}")

    abs_lat = np.abs(df["Lat"].values)

    # ============================================================
    # Test 1: C vs |Lat|
    # ============================================================
    print("\n" + "="*70)
    print("Test 1: per-det C vs |Lat| bands")
    print("="*70)

    lat_bands = [(0, 3), (3, 10), (10, 20), (20, 30), (30, 45)]
    print(f"\n  {'Lat band':<14}{'rows':>12}{'mean C':>10}{'std C':>10}{'range C':>20}")
    band_results = []
    for lat_lo, lat_hi in lat_bands:
        m_band = (abs_lat >= lat_lo) & (abs_lat < lat_hi) & is_clean
        # Compute per-det C in this band
        consts_band = {}
        for box in "ABC":
            for det in range(6):
                m = ((df["box"] == box) & (df["det"] == det)).values & m_band
                if m.sum() < 100:
                    continue
                consts_band[(box, det)] = float(np.mean(residual[m]))
        if not consts_band:
            continue
        vals = list(consts_band.values())
        print(f"  |Lat|={lat_lo:2d}-{lat_hi:2d}°  {m_band.sum():>11,}  "
              f"{np.mean(vals):>+8.1f}  {np.std(vals):>9.1f}  "
              f"[{min(vals):>+6.0f}, {max(vals):>+6.0f}]")
        band_results.append((lat_lo, lat_hi, np.mean(vals), consts_band))

    # Per-det breakdown across bands (to see if same per-det relative pattern)
    print(f"\n  Per-det C across |Lat| bands (looking for consistency):")
    print(f"    {'(box,det)':<10}" + "  ".join(f"|Lat|{lo}-{hi}°" for lo, hi, _, _ in band_results))
    for box in "ABC":
        for det in range(6):
            row = [f"{box}-{det}".ljust(10)]
            for lo, hi, _, consts_band in band_results:
                if (box, det) in consts_band:
                    row.append(f"{consts_band[(box, det)]:+8.0f} ")
                else:
                    row.append("    -    ")
            print("    " + "".join(row))

    # ============================================================
    # Test 2: residual rate dependence
    # ============================================================
    print("\n" + "="*70)
    print("Test 2: residual rate dependence after per-det C subtraction")
    print("="*70)

    # residual_corrected = residual - C(box, det)
    C_per_row = np.zeros(len(df))
    for (b, d), v in C_per_det.items():
        m = ((df["box"] == b) & (df["det"] == d)).values
        C_per_row[m] = v
    residual_corrected = residual - C_per_row

    print(f"\n  Per-det LSQ fit: residual_corrected = a + k·Sci  on clean rows")
    print(f"  {'(box,det)':<10}{'a (cnt/s)':>14}{'k':>14}{'k·1000 cnt/s':>16}{'N':>10}")
    fit_results = []
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_clean
            if m.sum() < 1000:
                continue
            x = sci[m]
            y = residual_corrected[m]
            X = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, k = coef
            fit_results.append((box, det, a, k, m.sum()))
            print(f"  {box}-{det}      {a:>+12.2f}  {k:>+12.5f}    {k*1000:>+12.1f}      {m.sum():>10,}")

    a_vals = [r[2] for r in fit_results]
    k_vals = [r[3] for r in fit_results]
    print(f"\n  Across 18 detectors:")
    print(f"    intercept (a): mean={np.mean(a_vals):+.2f}, std={np.std(a_vals):.2f}")
    print(f"    slope    (k): mean={np.mean(k_vals):+.5f}, std={np.std(k_vals):.5f}")
    print(f"    k range: [{min(k_vals):+.4f}, {max(k_vals):+.4f}]")
    print(f"    k_avg · Sci_at_1000 = {np.mean(k_vals) * 1000:+.1f} cnt/s")
    print(f"    k_avg · Sci_at_2000 = {np.mean(k_vals) * 2000:+.1f} cnt/s")


if __name__ == "__main__":
    main()
