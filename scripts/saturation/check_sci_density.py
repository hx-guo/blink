#!/usr/bin/env python3
"""Quick density check: how many per-det-sec bins fall in each Sci decade?
Pooled across the same 14M-point dataset used by plot_n_below_beta_gamma_full.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path("n_below_study/per_sec_csvs")
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100

dtype = {"date": "string", "box": "category", "met_sec": "int64",
         "det": "int8", "L_cycles": "int32",
         "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
use = list(dtype)

files = sorted(CSV_DIR.glob("*.csv"))
parts = []
for i, f in enumerate(files):
    try:
        parts.append(pd.read_csv(f, usecols=use, dtype=dtype))
    except Exception:
        pass
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(files)}")
df = pd.concat(parts, ignore_index=True)
df["length"] = df["L_cycles"].astype("float32") * 16e-6
df = df[df["L_cycles"] > L_THRESH].copy()

g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
g.name = "sci_sec_total"
df = df.merge(g, on=["date", "box", "met_sec"])
df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]
df["sci_rate"] = df["Sci"] / df["length"]

print(f"\nTotal: {len(df):,} per-det-sec rows\n")

# Density per Sci decade, per box
edges = [40, 60, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000]

print(f"{'Sci range [cnt/s/det]':>22s} | {'A':>10s} {'B':>10s} {'C':>10s} | {'total':>10s}  fraction")
print("-" * 80)
total_all = len(df)
for lo, hi in zip(edges[:-1], edges[1:]):
    row = []
    for box in "ABC":
        sub = df[df["box"] == box]
        n = ((sub["sci_rate"] >= lo) & (sub["sci_rate"] < hi)).sum()
        row.append(n)
    tot = sum(row)
    frac = tot / total_all * 100
    print(f"  {lo:>5d} – {hi:>5d}        | {row[0]:>10,d} {row[1]:>10,d} {row[2]:>10,d} | {tot:>10,d}  {frac:>5.2f}%")

# Above 10000
row = []
for box in "ABC":
    sub = df[df["box"] == box]
    n = (sub["sci_rate"] >= 10000).sum()
    row.append(n)
tot = sum(row); frac = tot / total_all * 100
print(f"  >= 10000             | {row[0]:>10,d} {row[1]:>10,d} {row[2]:>10,d} | {tot:>10,d}  {frac:>5.2f}%")

# Specifically: ratio of (100-200) vs (900-1100)
for box in "ABC":
    sub = df[df["box"] == box]
    n_low = ((sub["sci_rate"] >= 100) & (sub["sci_rate"] < 200)).sum()
    n_mid = ((sub["sci_rate"] >= 900) & (sub["sci_rate"] < 1100)).sum()
    n_peak = ((sub["sci_rate"] >= 1500) & (sub["sci_rate"] < 3000)).sum()
    print(f"Box {box}: 100-200 = {n_low:>9,d},  900-1100 = {n_mid:>9,d},  "
          f"1500-3000 = {n_peak:>9,d}   (ratios {n_low/max(n_mid,1):.2f}, {n_low/max(n_peak,1):.2f})")
