#!/usr/bin/env python3
"""Classify each per-(date,box) observation as 'high_γ' or 'low_γ' based on
the free fit, then look for what discriminates them.

Use the bimodal γ distribution as the labeller:
  γ < 0.4  → low_γ  state
  γ > 0.8  → high_γ state
  (in-between is ambiguous — exclude or label as 'mixed')

Then for each observation, summarize:
  - date (epoch)
  - Wide/PHO ratio mean
  - Large/PHO ratio mean
  - dt_frac
  - acd1_frac, acdn_frac
  - sci_rate distribution
  - mean HV (within normal HV mode already)

Tests: which observable variable shows a step at the γ-boundary?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PER_OBS_FIT = Path("plots/per_obs_free_gamma_fits.csv")
CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE_PARTIAL = Path("n_below_study/hv_table_partial.csv.gz")
HV_TABLE_FULL = Path("n_below_study/hv_table.csv.gz")
OUT_DIR = Path("plots")

L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def main():
    fits = pd.read_csv(PER_OBS_FIT)
    print(f"Loaded {len(fits)} per-obs fits")
    fits["state"] = pd.cut(fits["gamma"], bins=[-1, 0.4, 0.8, 2], labels=["low_γ","mixed","high_γ"])
    print(fits["state"].value_counts())

    # Load per-sec data, join with HV, filter to normal HV mode
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH]
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_frac"] = df["Wide"] / df["PHO"].clip(lower=1)
    df["large_frac"] = df["Large"] / df["PHO"].clip(lower=1)
    df["dt_frac"] = df["Dt"] / df["PHO"].clip(lower=1)
    df["acd1_frac"] = df["Sci_ACD1"] / df["Sci"].clip(lower=1)
    df["acdn_frac"] = df["Sci_ACDN"] / df["Sci"].clip(lower=1)
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["year"] = df["date"].str.slice(0, 4).astype("int16")
    df["month"] = df["date"].str.slice(5, 7).astype("int8")

    # Join HV
    hv_path = HV_TABLE_PARTIAL if HV_TABLE_PARTIAL.exists() else HV_TABLE_FULL
    print(f"Loading HV table {hv_path}...")
    hv = pd.read_csv(hv_path, dtype={"date":"string","met_sec":"int64",
                                     **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"normal-mode rows: {len(df):,}")

    # Add state label by joining on (date, box)
    df = df.merge(fits[["date","box","state","gamma","alpha","b","beta"]],
                  on=["date","box"], how="left")
    df = df[df["state"].isin(["low_γ","high_γ"])].copy()
    print(f"after state filter: {len(df):,}")

    # Compare ratios between states
    print(f"\n=== Per-state ratio stats ===")
    for col in ["wide_frac","large_frac","dt_frac","acd1_frac","acdn_frac",
                "year","sci_rate"]:
        for state in ["low_γ","high_γ"]:
            sub = df[df["state"] == state][col]
            print(f"  {state:>7s} {col:>12s}: n={len(sub):>9,d}  "
                  f"median={sub.median():>8.4f}  mean={sub.mean():>8.4f}")

    # Year breakdown
    print(f"\n=== Year breakdown by state ===")
    yearstate = df.groupby(["year","state"], observed=True).size().unstack(fill_value=0)
    print(yearstate)

    # Quick: are low_γ states clustered at certain dates? Print first 20 of each
    print(f"\n=== Sample dates in each state (first 15) ===")
    for state in ["low_γ","high_γ"]:
        dates = sorted(set(fits[fits["state"] == state]["date"].astype(str)))
        print(f"  {state}: {len(dates)} unique dates")
        print(f"    first 15: {dates[:15]}")

    # Plot: histograms of ratios, two colors per state
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols = ["wide_frac","large_frac","dt_frac","acd1_frac","acdn_frac","sci_rate"]
    for ax, c in zip(axes.flat, cols):
        for state, color in zip(["low_γ","high_γ"], ["C3","C0"]):
            d = df[df["state"] == state][c]
            ax.hist(d, bins=120, alpha=0.5, color=color, label=f"{state} (N={len(d):,})")
        ax.set_xlabel(c)
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)
        if c == "sci_rate":
            ax.set_xscale("log")
            ax.set_yscale("log")
    fig.suptitle("Per-state ratio distributions (low_γ vs high_γ within normal HV mode)")
    fig.tight_layout()
    out = OUT_DIR / "gamma_state_ratios.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
