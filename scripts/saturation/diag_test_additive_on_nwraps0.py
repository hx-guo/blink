#!/usr/bin/env python3
"""Critical test: on n_wraps=0 rows only (no Large wrap involved), is the residual
truly flat (additive C) or sloped (mixed/multiplicative)?

If pure additive: residual slope k ≈ 0 on n_wraps=0 rows
If mixed/multiplicative: k > 0 even on n_wraps=0 rows

Also check n_wraps=1 rows: after v2 unwrap, their residual should match
n_wraps=0 if v2 is correct. If n_wraps=1 residual is offset by ~+1024,
v2 missed those wraps (and the apparent slope is unwrap artifact).
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

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def fit_per_det(df, residual, sci, valid_mask, label="Group"):
    print(f"\n{label}: per-(box, det) fit residual = a + k·Sci")
    print(f"  {'(box,det)':<10}{'a':>12}{'k':>12}{'N':>10}")
    a_vals, k_vals = [], []
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & valid_mask
            if m.sum() < 1000:
                print(f"  {box}-{det}      (insufficient {m.sum()})")
                continue
            x = sci[m]
            y = residual[m]
            X = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, k = coef
            a_vals.append(a); k_vals.append(k)
            print(f"  {box}-{det}     {a:>+10.1f}  {k:>+11.5f}  {m.sum():>10,}")
    print(f"\n  Across {len(k_vals)} detectors:")
    print(f"    a mean ± std: {np.mean(a_vals):+.1f} ± {np.std(a_vals):.1f}")
    print(f"    k mean ± std: {np.mean(k_vals):+.5f} ± {np.std(k_vals):.5f}")
    return np.mean(a_vals), np.mean(k_vals)


def main():
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"Loaded {len(df):,} rows")

    pho = df["PHO"].values
    large_raw = df["Large"].values
    wide = df["Wide"].values
    sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values
    dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    # One-pass v2 unwrap with global C=195 (use the converged value)
    large_corr, conf = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=195.0, return_confidence=True,
    )
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean_basic = (conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100) & np.isfinite(residual)
    print(f"\nClean rows (HIGH conf, non-magnetar, Sci>100): {is_clean_basic.sum():,}")

    # =====================================================
    # Test 1: only n_wraps=0 rows (no wrap at all → pure physics test)
    # =====================================================
    print("\n" + "="*70)
    print("Test 1: n_wraps=0 rows ONLY (no unwrap involved)")
    print("="*70)
    is_n0 = is_clean_basic & (n_wraps == 0)
    print(f"  Count: {is_n0.sum():,}")
    a0, k0 = fit_per_det(df, residual, sci, is_n0, label="n_wraps=0")

    # =====================================================
    # Test 2: only n_wraps=1 rows (v2 corrected with 1 wrap)
    # =====================================================
    print("\n" + "="*70)
    print("Test 2: n_wraps=1 rows ONLY (v2 added 1 wrap)")
    print("="*70)
    is_n1 = is_clean_basic & (n_wraps == 1)
    print(f"  Count: {is_n1.sum():,}")
    a1, k1 = fit_per_det(df, residual, sci, is_n1, label="n_wraps=1")

    # =====================================================
    # Verification: if n=1 residual matches n=0 residual, v2 unwrap is correct.
    # If n=1 is shifted by ~+1024, v2 missed wraps.
    # =====================================================
    print("\n" + "="*70)
    print("Comparison: n=0 vs n=1 — should match if v2 unwrap is correct")
    print("="*70)
    print(f"  n_wraps=0: a = {a0:+.1f},  k = {k0:+.5f}")
    print(f"  n_wraps=1: a = {a1:+.1f},  k = {k1:+.5f}")
    print(f"  Δa = a(n=1) - a(n=0) = {a1-a0:+.1f}  (should be ≈0 if unwrap correct)")
    print(f"  Δk = k(n=1) - k(n=0) = {k1-k0:+.5f}  (should be ≈0)")

    if abs(a1 - a0) > 200:
        print(f"\n  ⚠ Δa is large ({a1-a0:+.0f}) — v2 unwrap may have systematic bias")
    else:
        print(f"\n  ✓ n=0 and n=1 residuals agree → v2 unwrap is consistent")


if __name__ == "__main__":
    main()
