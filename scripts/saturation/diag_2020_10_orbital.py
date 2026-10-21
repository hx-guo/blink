#!/usr/bin/env python3
"""Drill into 2020-10-10..16 anomaly: is it orbit-tied or globally uniform?

If orbit-tied (Lat/Lon clustering in anomaly seconds) → external particle source
(SPE, cosmic ray storm, or trapped particles).
If globally uniform → some other mechanism.

Tests within just the 2020-10-10..16 anomaly week:
A) Lat/Lon/Alt distribution of anom rows vs non-anom rows (same days)
B) Time-of-day pattern (orbital phase)
C) Per-second co-occurrence: do all 18 (box, det) elevate at the SAME second?
D) PHO rate distribution
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6


def stats(name, x):
    if len(x) == 0:
        return f"{name}: empty"
    return (f"{name}: n={len(x):>10,}  median={np.median(x):>10.2f}  "
            f"mean={np.mean(x):>10.2f}  Q05={np.quantile(x, 0.05):>10.2f}  "
            f"Q95={np.quantile(x, 0.95):>10.2f}")


def main():
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df):,} rows")

    pho = df["PHO"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    wide_pho = np.where(pho > 0, wide / pho, np.nan)

    # Restrict to 2020-10-10..16 week
    in_week = ((df["date"] >= "2020-10-10") & (df["date"] <= "2020-10-16")).values
    print(f"\n2020-10-10..16 week: {in_week.sum():,} rows")

    # Anomaly: W/P > 0.3
    anom = in_week & (wide_pho > 0.3) & np.isfinite(wide_pho)
    nonanom = in_week & (wide_pho <= 0.3) & np.isfinite(wide_pho)
    print(f"  Anom (W/P>0.3):    {anom.sum():>8,}  ({anom.sum() / in_week.sum() * 100:.2f}%)")
    print(f"  Non-anom (W/P≤0.3): {nonanom.sum():>8,}  ({nonanom.sum() / in_week.sum() * 100:.2f}%)")

    # ===========================
    # A. Lat/Lon/Alt distribution
    # ===========================
    print("\n" + "="*70)
    print("A. Orbital position of anom vs non-anom (same week)")
    print("="*70)
    if "Lat" in df.columns:
        print(stats("  |Lat| Anom",       np.abs(df["Lat"].values[anom])))
        print(stats("  |Lat| Non-anom",   np.abs(df["Lat"].values[nonanom])))
    if "Lon" in df.columns:
        # Lon in [0, 360) per HXMT convention
        print(stats("  Lon Anom",         df["Lon"].values[anom]))
        print(stats("  Lon Non-anom",     df["Lon"].values[nonanom]))
    if "Alt" in df.columns:
        print(stats("  Alt Anom",         df["Alt"].values[anom]))
        print(stats("  Alt Non-anom",     df["Alt"].values[nonanom]))

    # Lat in signed
    if "Lat" in df.columns:
        print(stats("  Lat (signed) Anom",     df["Lat"].values[anom]))
        print(stats("  Lat (signed) Non-anom", df["Lat"].values[nonanom]))

    # ===========================
    # B. Time-of-day / MET pattern
    # ===========================
    print("\n" + "="*70)
    print("B. Anomaly time series within the week")
    print("="*70)
    print("Hourly anom-fraction over the week (using MET truncated to hours):")
    if "met_sec" in df.columns:
        met = df["met_sec"].values
        # Compute hour bins from MET (MET starts of week)
        met_week = met[in_week]
        anom_in_week = (wide_pho[in_week] > 0.3) & np.isfinite(wide_pho[in_week])
        met_anom = met_week[anom_in_week]
        met_normal = met_week[~anom_in_week]

        # Hour bins from min MET to max MET in week
        met_min = met_week.min()
        met_max = met_week.max()
        # Output anomaly count per hour
        hours = np.arange(met_min, met_max + 3600, 3600)
        anom_per_hour, _ = np.histogram(met_anom, bins=hours)
        total_per_hour, _ = np.histogram(met_week, bins=hours)
        anom_frac = np.where(total_per_hour > 0, anom_per_hour / total_per_hour, 0)
        # Show first 12 hours of anomaly time + interesting parts
        print(f"\n  Total hours: {len(hours) - 1}")
        # Find hours where anom_frac > 0.05 (significant)
        sig = np.where(anom_frac > 0.05)[0]
        print(f"  Hours with anom_frac > 5%: {len(sig)} out of {len(hours) - 1}")
        # Look at clustering
        if len(sig) > 0:
            diffs = np.diff(sig)
            print(f"  Gap between sig hours: median={np.median(diffs):.1f} hr, max={diffs.max()} hr")
        # Print a sample of hour bins
        print("\n  Sample of 30 random anom hours:")
        sample = np.random.RandomState(0).choice(np.arange(len(hours) - 1), min(30, len(hours) - 1), replace=False)
        sample.sort()
        for i in sample:
            day = pd.to_datetime(hours[i], unit="s", origin="2012-01-01 00:00:00").strftime("%Y-%m-%d %H")
            print(f"    {day} UTC offset: anom={anom_per_hour[i]:>4} / total={total_per_hour[i]:>4} ({anom_frac[i]*100:>5.1f}%)")

    # ===========================
    # C. PHO rate during anom vs non-anom
    # ===========================
    print("\n" + "="*70)
    print("C. PHO rate distribution within the week")
    print("="*70)
    print(stats("  PHO Anom",     pho[anom] / L[anom]))
    print(stats("  PHO Non-anom", pho[nonanom] / L[nonanom]))

    # ===========================
    # D. Per-second uniformity
    # Are all 18 (box, det) anomalous at the SAME second?
    # ===========================
    print("\n" + "="*70)
    print("D. Per-second uniformity test")
    print("="*70)
    # Group by met_sec, count how many of the 18 detectors are anomalous
    if "met_sec" in df.columns:
        week_df = df.loc[in_week].copy()
        week_df["anom"] = wide_pho[in_week] > 0.3
        # Per (met_sec), how many of 18 detectors are flagged anom?
        anom_per_sec = week_df.groupby("met_sec")["anom"].sum()
        total_per_sec = week_df.groupby("met_sec").size()
        # Only seconds where all 18 dets present
        full_sec = total_per_sec == 18
        full_sec_anom = anom_per_sec[full_sec]
        print(f"  Total seconds with all 18 dets present: {full_sec.sum():,}")
        if full_sec.sum() > 0:
            print(f"  Anom-dets-per-second distribution:")
            for k in [0, 1, 5, 10, 15, 17, 18]:
                n = (full_sec_anom == k).sum()
                print(f"    {k:>2}/18 dets anom: {n:>8,}  ({n / full_sec.sum() * 100:.2f}%)")
            # Bimodality test
            partial = (full_sec_anom > 0) & (full_sec_anom < 18)
            all_or_none = (full_sec_anom == 0) | (full_sec_anom == 18)
            print(f"  All-or-none (0 or 18 dets anom):    {all_or_none.sum():,} ({all_or_none.mean() * 100:.2f}%)")
            print(f"  Partial (1-17 dets anom):           {partial.sum():,} ({partial.mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
