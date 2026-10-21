#!/usr/bin/env python3
"""Memory-efficient: list ALL LOW-mode dates + segment start times.
Only retains LOW-mode rows (~30k) instead of loading all 3M CLEAN rows.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CSV_DIR = Path("n_below_study/per_sec_csvs")
L_THRESH = 50_000
SCI_LO, SCI_HI = 400.0, 1000.0
LOW_THRESH = 0.40
HEPOCH = datetime(2012, 12, 31, 0, 0, 0)


def main():
    # File-by-file: keep only LOW rows
    low_parts = []
    n_files = 0
    for f in sorted(CSV_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(f, usecols=["box","det","met_sec","L_cycles","Sci","Large"])
            if len(d) == 0: continue
            n_files += 1
            d = d[d["L_cycles"] > L_THRESH]
            d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
            if len(d) == 0: continue
            d["ratio"] = d["Large"].astype("float32") / d["Sci"].clip(lower=1)
            low = d[d["ratio"] < LOW_THRESH]
            if len(low) > 0:
                low_parts.append(low[["box","det","met_sec"]].copy())
        except Exception:
            pass
    low = pd.concat(low_parts, ignore_index=True)
    print(f"Files processed: {n_files}, total LOW rows: {len(low):,}")

    # Add date column from MET
    low["dt"] = [HEPOCH + timedelta(seconds=int(t)) for t in low["met_sec"].values]
    low["date"] = [d.strftime("%Y-%m-%d") for d in low["dt"]]
    low["dom"]  = [d.day for d in low["dt"]]
    low["hour"] = [d.hour for d in low["dt"]]

    print(f"\n{'='*78}")
    print("ALL distinct dates with at least one LOW row (sorted by count)")
    print(f"{'='*78}")
    by_date = low.groupby("date").size().rename("n_low").reset_index()
    by_date = by_date.sort_values("n_low", ascending=False)
    print(f"  Total distinct dates: {len(by_date)}")
    print(f"  {'date':>12s}  {'day-of-month':>13s}  {'n_LOW':>7s}")
    for _, r in by_date.iterrows():
        dom = pd.to_datetime(r["date"]).day
        marker = " ← 月初" if dom == 1 else ""
        print(f"  {r['date']:>12s}  {dom:>13d}  {int(r['n_low']):>7d}{marker}")

    # ----- All continuous LOW segments ≥30s -----
    print(f"\n{'='*78}")
    print("ALL continuous LOW segments ≥30s (per (box, det)), sorted by duration")
    print(f"{'='*78}")
    low_sorted = low.sort_values(["box","det","met_sec"])
    segs = []
    for (box, det), g in low_sorted.groupby(["box","det"]):
        met = g["met_sec"].values
        if len(met) == 0: continue
        breaks = np.where(np.diff(met) > 2)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends   = np.concatenate([breaks, [len(met)-1]])
        for s, e in zip(starts, ends):
            seg_len = e - s + 1
            if seg_len >= 30:
                segs.append({
                    "box": box, "det": det,
                    "met_start": int(met[s]), "met_end": int(met[e]),
                    "duration": int(met[e] - met[s] + 1),
                    "n_low": seg_len,
                    "ts_start": HEPOCH + timedelta(seconds=int(met[s])),
                })
    seg_df = pd.DataFrame(segs).sort_values("duration", ascending=False)
    print(f"  Total segments ≥30s: {len(seg_df)}")
    print(f"\n  {'date_time_UTC':>20s}  {'dom':>3s}  {'hh:mm':>6s}  "
          f"{'box-det':>8s}  {'dur_s':>6s}  {'n_LOW':>6s}")
    for _, r in seg_df.iterrows():
        ts = r["ts_start"]
        dom = ts.day
        hhmm = ts.strftime("%H:%M")
        marker = " ← 月初" if dom == 1 else ""
        print(f"  {ts.strftime('%Y-%m-%d %H:%M:%S')}  {dom:>3d}  {hhmm:>6s}  "
              f"{r['box']}-{r['det']:>1d}      {r['duration']:>6d}  {r['n_low']:>6d}{marker}")

    # Pattern summary
    print(f"\n{'='*78}")
    print("Pattern summary")
    print(f"{'='*78}")
    n_dom_1 = (seg_df["ts_start"].apply(lambda x: x.day) == 1).sum()
    n_hour_0 = (seg_df["ts_start"].apply(lambda x: x.hour) == 0).sum()
    n_both = ((seg_df["ts_start"].apply(lambda x: x.day) == 1) &
              (seg_df["ts_start"].apply(lambda x: x.hour) == 0)).sum()
    print(f"  Total ≥30s segments: {len(seg_df)}")
    print(f"  At day-of-month = 1:  {n_dom_1}  ({100*n_dom_1/len(seg_df):.1f}%)")
    print(f"  At hour = 0 (UTC):    {n_hour_0}  ({100*n_hour_0/len(seg_df):.1f}%)")
    print(f"  Both 1st AND 00:xx:   {n_both}  ({100*n_both/len(seg_df):.1f}%)")


if __name__ == "__main__":
    main()
