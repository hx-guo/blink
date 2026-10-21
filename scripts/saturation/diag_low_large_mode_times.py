#!/usr/bin/env python3
"""Find time/date patterns of LOW-Large mode (upper-cloud) data.

Classification:
  Large/Sci > 0.50  → HIGH mode (main band)
  Large/Sci < 0.40  → LOW mode (upper cloud)
  in between        → ambiguous (skipped)

Then look for patterns:
  - Date distribution (which days have most LOW rows)
  - MET-mod-day / MET-mod-orbit time-of-day patterns
  - Per-(box, det) prevalence
  - Per-HV bin
  - Longest continuous LOW segments per (box, det)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
L_THRESH = 50_000
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

# CLEAN band (same as V8)
SCI_LO, SCI_HI, BOX_RATE_CAP = 400.0, 1000.0, 6000.0

# LOW/HIGH thresholds on Large/Sci ratio
LOW_THRESH = 0.40
HIGH_THRESH = 0.50

# Orbital period of HXMT (~ 96.5 minutes, ~ 5790 seconds)
ORBIT_PERIOD = 5790

# MET epoch: 2017-06-15 ≈ MET 0 (approximate)
# 86400 s/day for daily modulation


def load():
    dtype = {"date":"string","box":"category","met_sec":"int64","det":"int8",
             "L_cycles":"int32","PHO":"int32","Wide":"int32","Large":"int32",
             "Dt":"int32","Sci":"int32","Sci_ACD1":"int32","Sci_ACDN":"int32"}
    parts = []
    for f in sorted(CSV_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(f, usecols=list(dtype), dtype=dtype)
            if len(d) > 0:
                parts.append(d)
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH]
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > 100].copy()
    df["sci_rate"] = df["Sci"] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    hv = pd.read_csv(HV_TABLE, dtype={"date":"string","met_sec":"int64",
        **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    return df


def main():
    print("Loading training data...")
    df = load()
    print(f"  total rows: {len(df):,}")

    # CLEAN band
    clean = ((df["sci_rate"] >= SCI_LO) & (df["sci_rate"] < SCI_HI)
              & (df["group_rate"] < BOX_RATE_CAP))
    df = df[clean].copy()
    print(f"  CLEAN-band rows: {len(df):,}")

    # Classification by Large/Sci ratio
    df["large_over_sci"] = df["Large"].astype("float32") / df["Sci"].clip(lower=1)
    df["mode"] = np.where(df["large_over_sci"] > HIGH_THRESH, "HIGH",
                  np.where(df["large_over_sci"] < LOW_THRESH, "LOW", "AMBIG"))

    counts = df["mode"].value_counts()
    print(f"\nMode classification ({LOW_THRESH} ≤ Large/Sci < {HIGH_THRESH} marked ambiguous):")
    for m, n in counts.items():
        print(f"  {m:>6s}: {n:>9,d}  ({100*n/len(df):5.2f}%)")

    low = df[df["mode"] == "LOW"]
    print(f"\nLOW-mode summary: {len(low):,} rows ({100*len(low)/len(df):.2f}%)")

    # ============= Date distribution =============
    print("\n" + "="*70)
    print("Per-date LOW-mode prevalence (top 30 worst days)")
    print("="*70)
    per_date = df.groupby("date").agg(
        total=("mode", "size"),
        low_n=("mode", lambda x: (x=="LOW").sum()),
    )
    per_date["low_frac"] = per_date["low_n"] / per_date["total"]
    per_date_sorted = per_date.sort_values("low_frac", ascending=False)
    print(f"  {'date':>12s}  {'total':>7s}  {'LOW_n':>7s}  {'LOW%':>7s}")
    for d, r in per_date_sorted.head(30).iterrows():
        print(f"  {d:>12s}  {int(r['total']):>7d}  {int(r['low_n']):>7d}  {r['low_frac']*100:>6.2f}%")

    print(f"\n  ...and {len(per_date_sorted)-30} more dates...")
    print(f"\n  Date distribution: {(per_date['low_frac'] > 0.10).sum()} dates with >10% LOW, "
          f"{(per_date['low_frac'] > 0.30).sum()} with >30%, {(per_date['low_frac'] > 0.50).sum()} with >50%")

    # ============= Per-(box, det) prevalence =============
    print("\n" + "="*70)
    print("Per-(box, det) LOW-mode prevalence")
    print("="*70)
    pd_grp = df.groupby(["box", "det"]).agg(
        total=("mode", "size"),
        low_n=("mode", lambda x: (x=="LOW").sum()),
    ).reset_index()
    pd_grp["low_frac"] = pd_grp["low_n"] / pd_grp["total"]
    print(f"  {'box-det':>8s}  {'total':>7s}  {'LOW_n':>7s}  {'LOW%':>7s}")
    for _, r in pd_grp.iterrows():
        print(f"  {r['box']}-{r['det']:>1d}      {int(r['total']):>7d}  {int(r['low_n']):>7d}  {r['low_frac']*100:>6.2f}%")

    # ============= Per-HV bin =============
    print("\n" + "="*70)
    print("Per-HV-bin LOW-mode prevalence")
    print("="*70)
    df["hv_bin"] = (df["hv"] / 10).round() * 10
    hv_grp = df.groupby("hv_bin").agg(
        total=("mode", "size"),
        low_n=("mode", lambda x: (x=="LOW").sum()),
    ).reset_index()
    hv_grp["low_frac"] = hv_grp["low_n"] / hv_grp["total"]
    hv_grp = hv_grp.sort_values("hv_bin")
    print(f"  {'HV [V]':>8s}  {'total':>9s}  {'LOW_n':>7s}  {'LOW%':>7s}")
    for _, r in hv_grp.iterrows():
        print(f"  {r['hv_bin']:>+8.0f}  {int(r['total']):>9d}  {int(r['low_n']):>7d}  "
              f"{r['low_frac']*100:>6.2f}%")

    # ============= Continuous LOW segments =============
    print("\n" + "="*70)
    print("Continuous LOW segments per (box, det) — top 20 longest")
    print("="*70)
    # Build per-(box, det) sorted series of met_sec for LOW rows
    df_low = df[df["mode"] == "LOW"].sort_values(["box","det","met_sec"]).copy()
    segs = []
    for (box, det), g in df_low.groupby(["box", "det"]):
        met = g["met_sec"].values
        if len(met) == 0: continue
        # Find consecutive runs (gap <= 1 sec)
        breaks = np.where(np.diff(met) > 2)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends   = np.concatenate([breaks, [len(met)-1]])
        for s, e in zip(starts, ends):
            seg_len = e - s + 1
            if seg_len >= 30:   # only seg ≥30s
                segs.append({
                    "box": box, "det": det,
                    "met_start": int(met[s]), "met_end": int(met[e]),
                    "duration": int(met[e] - met[s] + 1),
                    "n_low": seg_len,
                    "date_start": pd.to_datetime(int(met[s]), unit='s', origin='2012-12-31'),
                })
    seg_df = pd.DataFrame(segs).sort_values("duration", ascending=False)
    if len(seg_df) > 0:
        print(f"  Found {len(seg_df)} LOW segments of length ≥30 seconds across all (box,det)")
        print(f"\n  {'date':>20s}  {'box-det':>8s}  {'met_start':>11s}  "
              f"{'duration_s':>11s}  {'n_LOW':>6s}")
        for _, r in seg_df.head(20).iterrows():
            print(f"  {str(r['date_start']):>20s}  {r['box']}-{r['det']:>1d}      "
                  f"{r['met_start']:>11d}  {r['duration']:>11d}  {r['n_low']:>6d}")
        # Total LOW time vs scattered
        print(f"\n  Total LOW time in ≥30s segments: {seg_df['duration'].sum():,} sec")
        print(f"  Total LOW rows in dataset:        {len(df_low):,}")
        print(f"  → {100*seg_df['duration'].sum()/len(df_low):.1f}% of LOW rows are in long segments")

    # ============= Time of day / orbital pattern =============
    # MET is seconds since some epoch (HXMT epoch ≈ 2012-12-31)
    print("\n" + "="*70)
    print("Time-of-day pattern (MET mod 86400, in hours)")
    print("="*70)
    df["hour_of_day"] = (df["met_sec"] % 86400) / 3600.0
    hod_grp = df.groupby(pd.cut(df["hour_of_day"], bins=24)).agg(
        total=("mode", "size"),
        low_n=("mode", lambda x: (x=="LOW").sum()),
    ).reset_index()
    hod_grp["low_frac"] = hod_grp["low_n"] / hod_grp["total"]
    print(f"  {'hour':>6s}  {'total':>9s}  {'LOW%':>7s}")
    for _, r in hod_grp.iterrows():
        print(f"  {str(r['hour_of_day']):>14s}  {int(r['total']):>9d}  {r['low_frac']*100:>6.2f}%")

    print("\n" + "="*70)
    print("Orbital phase pattern (MET mod 5790 s = ~96.5 min orbit, in 10 bins)")
    print("="*70)
    df["orbit_phase"] = (df["met_sec"] % ORBIT_PERIOD) / ORBIT_PERIOD
    op_grp = df.groupby(pd.cut(df["orbit_phase"], bins=10)).agg(
        total=("mode", "size"),
        low_n=("mode", lambda x: (x=="LOW").sum()),
    ).reset_index()
    op_grp["low_frac"] = op_grp["low_n"] / op_grp["total"]
    print(f"  {'phase':>14s}  {'total':>9s}  {'LOW%':>7s}")
    for _, r in op_grp.iterrows():
        print(f"  {str(r['orbit_phase']):>14s}  {int(r['total']):>9d}  {r['low_frac']*100:>6.2f}%")


if __name__ == "__main__":
    main()
