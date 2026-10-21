#!/usr/bin/env python3
"""Hard-evidence tests for what causes the 2020-05 and 2020-10 anomaly clusters.

No prior assumption about cause. Just data signatures.

Tests:
A) OOC (^241Am on-board calibration) counter: stable across anomaly?
   - If stable → detector hardware itself is fine, only classification layer changes
   - If unstable → broader detector/DAQ issue
B) PHO total rate: does total NaI trigger rate change vs normal days?
   - If unchanged → events are conserved, only redistributed (classification logic change)
   - If changed → real event rate changed (e.g., external source / hardware failure)
C) Transition sharpness: how fast does Wide/PHO jump from 0.02 to 0.48?
   - Cycle-level (seconds) → operational/firmware change
   - Hour/day-level → progressive hardware degradation or thermal effect
D) Comparison: 2020-05 vs 2020-10 — same fingerprint or different?
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6


def stats(name, x, q05=0.05, q95=0.95):
    if len(x) == 0:
        return f"{name}: empty"
    return (f"{name}: n={len(x):>10,}  median={np.median(x):>10.2f}  "
            f"mean={np.mean(x):>10.2f}  Q05={np.quantile(x, q05):>10.2f}  "
            f"Q95={np.quantile(x, q95):>10.2f}")


def main():
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df):,} rows")

    pho = df["PHO"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    ooc = df["OOC"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    pho_rate = pho / L
    wide_rate = wide / L
    ooc_rate = ooc / L
    wide_pho = np.where(pho > 0, wide / pho, np.nan)

    # Define anomaly periods
    is_2020_05 = df["date"].str.startswith("2020-05").values
    is_2020_10_anom = ((df["date"] >= "2020-10-10") & (df["date"] <= "2020-10-16")).values
    is_normal = ~is_2020_05 & ~is_2020_10_anom & \
        (df["date"] >= "2020-01-01").values & (df["date"] <= "2020-12-31").values

    # Anomalous rows specifically (within anomaly periods, high Wide/PHO)
    anom_signature = wide_pho > 0.3

    # ===========================
    # Test A: OOC stable?
    # ===========================
    print("\n" + "="*70)
    print("A. OOC (^241Am calibration source) rate — should be stable hardware-wise")
    print("="*70)
    print(stats("Normal days", ooc_rate[is_normal]))
    print(stats("2020-05 (all)", ooc_rate[is_2020_05]))
    print(stats("2020-05 (anom signature, W/P>0.3)", ooc_rate[is_2020_05 & anom_signature]))
    print(stats("2020-10-10..16 (all)", ooc_rate[is_2020_10_anom]))
    print(stats("2020-10-10..16 (anom signature)", ooc_rate[is_2020_10_anom & anom_signature]))

    # ===========================
    # Test B: PHO total rate
    # ===========================
    print("\n" + "="*70)
    print("B. PHO total trigger rate — should be conserved if only classification changes")
    print("="*70)
    print(stats("Normal days", pho_rate[is_normal]))
    print(stats("2020-05 (all)", pho_rate[is_2020_05]))
    print(stats("2020-05 (anom signature)", pho_rate[is_2020_05 & anom_signature]))
    print(stats("2020-10-10..16 (all)", pho_rate[is_2020_10_anom]))
    print(stats("2020-10-10..16 (anom signature)", pho_rate[is_2020_10_anom & anom_signature]))

    # ===========================
    # Test C: Wide rate side-by-side
    # ===========================
    print("\n" + "="*70)
    print("C. Wide rate")
    print("="*70)
    print(stats("Normal days", wide_rate[is_normal]))
    print(stats("2020-05 (all)", wide_rate[is_2020_05]))
    print(stats("2020-05 (anom signature)", wide_rate[is_2020_05 & anom_signature]))
    print(stats("2020-10-10..16 (all)", wide_rate[is_2020_10_anom]))
    print(stats("2020-10-10..16 (anom signature)", wide_rate[is_2020_10_anom & anom_signature]))

    # ===========================
    # Test D: transition sharpness for 2020-10
    # ===========================
    print("\n" + "="*70)
    print("D. Transition sharpness (Wide/PHO ratio per day around 2020-10 anomaly)")
    print("="*70)
    bracket_dates = [f"2020-10-{d:02d}" for d in range(1, 25)]
    for d in bracket_dates:
        m = (df["date"] == d).values
        if m.sum() == 0:
            continue
        wp = wide_pho[m & np.isfinite(wide_pho)]
        if len(wp) == 0:
            continue
        wp_med = np.median(wp)
        n_anom = (wp > 0.3).sum()
        pct = n_anom / len(wp) * 100
        print(f"  {d}: rows={m.sum():>5}, W/P median={wp_med:.3f}, anom-rows={n_anom:>5} ({pct:>5.2f}%)")

    print("\nSame for 2020-04-25..05-05 (2020-05 onset):")
    bracket_dates = [f"2020-04-{d:02d}" for d in range(25, 31)] + [f"2020-05-{d:02d}" for d in range(1, 6)]
    for d in bracket_dates:
        m = (df["date"] == d).values
        if m.sum() == 0:
            continue
        wp = wide_pho[m & np.isfinite(wide_pho)]
        if len(wp) == 0:
            continue
        wp_med = np.median(wp)
        n_anom = (wp > 0.3).sum()
        pct = n_anom / len(wp) * 100
        print(f"  {d}: rows={m.sum():>5}, W/P median={wp_med:.3f}, anom-rows={n_anom:>5} ({pct:>5.2f}%)")

    # ===========================
    # Test E: per-(box, det) signature uniformity
    # ===========================
    print("\n" + "="*70)
    print("E. Wide/PHO median per (box, det) on anomaly vs normal days (2020-10-12 vs normal)")
    print("="*70)
    print("\n  Normal days median W/P per (box, det):")
    for box in "ABC":
        cells = []
        for det in range(6):
            m = is_normal & (df["box"] == box).values & (df["det"] == det).values
            wp = wide_pho[m & np.isfinite(wide_pho)]
            cells.append(f"{np.median(wp):.3f}" if len(wp) else "n/a")
        print(f"    {box}: {' '.join(cells)}")

    print("\n  2020-10-12 median W/P per (box, det):")
    is_1012 = (df["date"] == "2020-10-12").values
    for box in "ABC":
        cells = []
        for det in range(6):
            m = is_1012 & (df["box"] == box).values & (df["det"] == det).values
            wp = wide_pho[m & np.isfinite(wide_pho)]
            cells.append(f"{np.median(wp):.3f}" if len(wp) else "n/a")
        print(f"    {box}: {' '.join(cells)}")


if __name__ == "__main__":
    main()
