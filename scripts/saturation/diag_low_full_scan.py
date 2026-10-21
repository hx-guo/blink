#!/usr/bin/env python3
"""Full-scale LOW-mode investigation across all 9000+ files.

Memory-efficient: file-by-file processing, keep only LOW rows + per-file stats.
Fixed MET epoch (2012-01-01 UTC) for correct date labels.

Three analyses:
  1. Per-file LOW% ranking (which dates are anomalous)
  2. Per-month / per-year aggregation
  3. For top-anomalous dates: time-series within 00:xx hour, check if PHO/Wide
     also abnormal during LOW segments
"""
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

CSV_DIR = Path("n_below_study/per_sec_csvs")
L_THRESH = 50_000
SCI_LO, SCI_HI = 400.0, 1000.0
LOW_THRESH = 0.40
HEPOCH = datetime(2012, 1, 1, 0, 0, 0)


def main():
    files = sorted(CSV_DIR.glob("*.csv"))
    files = [f for f in files if f.stat().st_size > 1000]
    print(f"Scanning {len(files):,} non-empty files...")

    # Per-file summary + all LOW rows
    per_file_stats = []   # (filename, date, box, n_clean, n_low)
    low_rows = []         # (box, det, met_sec, sci, large, pho, wide) for LOW rows

    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, usecols=["box","det","met_sec","L_cycles",
                                          "Sci","Large","PHO","Wide"])
        except Exception:
            continue
        if len(d) == 0: continue

        d = d[d["L_cycles"] > L_THRESH]
        d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
        if len(d) == 0: continue

        d["ratio"] = d["Large"].astype("float32") / d["Sci"].clip(lower=1)
        low = d[d["ratio"] < LOW_THRESH]
        per_file_stats.append({
            "file": f.name,
            "n_clean": len(d),
            "n_low": len(low),
            "low_pct": 100*len(low)/len(d),
        })
        if len(low) > 0:
            low_rows.append(low[["box","det","met_sec","Sci","Large","PHO","Wide"]].copy())

        if (i+1) % 1000 == 0:
            print(f"  ...processed {i+1}/{len(files)} files, accumulated "
                  f"{sum(len(p) for p in low_rows):,} LOW rows")

    stats = pd.DataFrame(per_file_stats)
    low_df = pd.concat(low_rows, ignore_index=True) if low_rows else pd.DataFrame()
    low_df["dt"] = [HEPOCH + timedelta(seconds=int(t)) for t in low_df["met_sec"].values]
    low_df["date"] = [d.strftime("%Y-%m-%d") for d in low_df["dt"]]
    low_df["yr_mo"] = [d.strftime("%Y-%m") for d in low_df["dt"]]
    low_df["yr"] = [d.year for d in low_df["dt"]]
    low_df["mo"] = [d.month for d in low_df["dt"]]
    low_df["dom"] = [d.day for d in low_df["dt"]]
    low_df["hour"] = [d.hour for d in low_df["dt"]]

    print(f"\nTotals: {stats['n_clean'].sum():,} CLEAN rows, "
          f"{len(low_df):,} LOW rows ({100*len(low_df)/stats['n_clean'].sum():.3f}%)")

    # ============= 1. Per-file LOW% ranking (top 20 anomalous days) =============
    print(f"\n{'='*70}")
    print("1. Top 20 files by LOW% (anomalous dates)")
    print(f"{'='*70}")
    print(f"  {'file':>22s}  {'CLEAN':>8s}  {'LOW':>6s}  {'LOW%':>7s}")
    top = stats.sort_values("low_pct", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  {r['file']:>22s}  {int(r['n_clean']):>8d}  "
              f"{int(r['n_low']):>6d}  {r['low_pct']:>6.2f}%")

    # Files with LOW% > 5%
    print(f"\n  Files with LOW% > 5%:  {(stats['low_pct'] > 5).sum()}/{len(stats)}")
    print(f"  Files with LOW% > 1%:  {(stats['low_pct'] > 1).sum()}/{len(stats)}")
    print(f"  Files with LOW% > 0.1%: {(stats['low_pct'] > 0.1).sum()}/{len(stats)}")
    print(f"  Files with LOW% = 0%:   {(stats['low_pct'] == 0).sum()}/{len(stats)}")

    # ============= 2. Per-year, per-month aggregation =============
    print(f"\n{'='*70}")
    print("2. Per-year LOW distribution")
    print(f"{'='*70}")
    by_yr = low_df.groupby("yr").size().rename("n_low").reset_index()
    by_yr = by_yr.sort_values("yr")
    print(f"  {'year':>6s}  {'n_LOW':>8s}")
    for _, r in by_yr.iterrows():
        print(f"  {int(r['yr']):>6d}  {int(r['n_low']):>8d}")

    print(f"\n{'='*70}")
    print("3. Per-month LOW distribution (Jan-Dec aggregated across years)")
    print(f"{'='*70}")
    by_mo = low_df.groupby("mo").size().rename("n_low").reset_index()
    by_mo = by_mo.sort_values("mo")
    print(f"  {'month':>6s}  {'n_LOW':>8s}")
    for _, r in by_mo.iterrows():
        print(f"  {int(r['mo']):>6d}  {int(r['n_low']):>8d}")

    print(f"\n{'='*70}")
    print("4. Per-DOM LOW distribution (1-31 aggregated)")
    print(f"{'='*70}")
    by_dom = low_df.groupby("dom").size().rename("n_low").reset_index()
    by_dom = by_dom.sort_values("dom")
    print(f"  {'dom':>4s}  {'n_LOW':>8s}")
    for _, r in by_dom.iterrows():
        print(f"  {int(r['dom']):>4d}  {int(r['n_low']):>8d}")

    # ============= 5. For top anomalous date: check PHO/Wide during LOW =============
    print(f"\n{'='*70}")
    print("5. For top 3 anomalous files: do PHO/Wide also look unusual during LOW?")
    print(f"{'='*70}")
    for _, top_r in top.head(3).iterrows():
        fname = top_r["file"]
        fpath = CSV_DIR / fname
        d = pd.read_csv(fpath, usecols=["box","det","met_sec","L_cycles",
                                          "Sci","Large","PHO","Wide"])
        d = d[d["L_cycles"] > L_THRESH]
        d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
        d["ratio"] = d["Large"] / d["Sci"].clip(lower=1)
        d["mode"] = np.where(d["ratio"] > 0.5, "HIGH",
                     np.where(d["ratio"] < 0.4, "LOW", "AMBIG"))
        n_high = (d["mode"]=="HIGH").sum()
        n_low = (d["mode"]=="LOW").sum()
        if n_low < 10 or n_high < 10: continue
        h = d[d["mode"]=="HIGH"]
        l = d[d["mode"]=="LOW"]
        print(f"\n  File: {fname}  (LOW%={top_r['low_pct']:.2f}, N_HIGH={n_high}, N_LOW={n_low})")
        print(f"    {'metric':>18s}  {'HIGH median':>13s}  {'LOW median':>12s}  "
              f"{'ratio LOW/HIGH':>16s}")
        for metric in ["Sci","PHO","Wide","Large","L_cycles"]:
            hm = h[metric].median()
            lm = l[metric].median()
            ratio = lm/hm if hm > 0 else float('nan')
            mark = "  ← 显著异常" if (ratio < 0.5 or ratio > 2.0) and metric != "L_cycles" else ""
            print(f"    {metric:>18s}  {hm:>13.1f}  {lm:>12.1f}  {ratio:>16.3f}{mark}")


if __name__ == "__main__":
    main()
