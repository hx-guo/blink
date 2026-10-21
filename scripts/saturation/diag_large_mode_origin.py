#!/usr/bin/env python3
"""Investigate what discriminates 'low Large/Sci' vs 'high Large/Sci' date modes.

Each date has 18 dets, with median Large/Sci in main band Sci=1000-1500.
That median varies from 0.11 to 0.97 across dates — bimodal distribution.

Hypothesis candidates:
  A. HV sub-mode within [-1100, -900]
  B. Time / mission epoch
  C. Per-detector (some dets are low-Large)
  D. Box-level effect

Plot:
  - Large/Sci median per date vs date
  - Large/Sci median per date vs median HV per date
  - Large/Sci median per det (across all dates)
  - Per-Box, per-det heatmap
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32"}
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

    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["large_frac"]  = df["large_rate"] / df["sci_rate"]
    df["det_global"]  = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

    hv = pd.read_csv(HV_TABLE,
                     dtype={"date":"string","met_sec":"int64",
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
    return df


def main():
    df = load()

    # Restrict to main band where the bimodality is clearest
    main = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    print(f"\nMain-band rows (Sci 1000-1500): {len(main):,}")

    # ========== 1. Large/Sci median per date+box+det ==========
    grp = main.groupby(["date", "box", "det"], observed=True).agg(
        large_frac=("large_frac", "median"),
        hv=("hv", "median"),
        met=("met_sec", "median"),
        N=("large_frac", "count"),
    ).reset_index()
    grp = grp[grp["N"] > 30]
    print(f"(date, box, det) groups with N>30: {len(grp):,}")

    # ========== 2. Per-det median across all dates ==========
    by_det = grp.groupby(["box", "det"])["large_frac"].agg(["median", "std", "count"])
    print(f"\n=== Per-det Large/Sci median (across dates) ===")
    print(by_det.to_string())

    # ========== 3. Per-date median across all dets ==========
    by_date = grp.groupby("date").agg(
        large_frac=("large_frac", "median"),
        hv=("hv", "median"),
        met=("met", "median"),
        N=("N", "sum"),
    )
    by_date = by_date.sort_values("date")
    print(f"\n=== Date stats: {len(by_date)} dates ===")
    print(f"  Large/Sci median: min={by_date['large_frac'].min():.3f}, "
          f"max={by_date['large_frac'].max():.3f}")

    # ========== 4. Plot: 4-panel diagnosis ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: per-det medians (box-colored)
    ax = axes[0, 0]
    det_means = grp.groupby(["box", "det"])["large_frac"].median().reset_index()
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = det_means[det_means["box"] == box]
        ax.scatter(sub["det"].astype(int) + (0 if box=="A" else (0.3 if box=="B" else 0.6)),
                   sub["large_frac"], s=80, color=color, label=f"Box {box}",
                   edgecolor="black", lw=0.5)
    ax.set_xlabel("det (within box)")
    ax.set_ylabel("median Large/Sci")
    ax.set_title("Per-detector Large/Sci (in Sci 1000-1500 band)\n"
                 "If certain dets are 'low Large', this would explain bimodality")
    ax.set_xticks(range(6))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0.4, color="gray", ls=":", lw=1)

    # Panel 2: per-date median Large/Sci vs date order
    ax = axes[0, 1]
    by_date_sorted = by_date.reset_index().sort_values("date")
    colors = ["red" if v > 0.55 else "blue" if v < 0.4 else "gray"
              for v in by_date_sorted["large_frac"]]
    ax.scatter(range(len(by_date_sorted)), by_date_sorted["large_frac"],
               c=colors, s=30, alpha=0.7)
    ax.set_xlabel("date index (chronological)")
    ax.set_ylabel("median Large/Sci (date-level)")
    ax.set_title("Large/Sci by date (chronological)\n"
                 "If temporal: should show smooth trend or sudden epoch shift")
    ax.axhline(0.4, color="blue", ls=":", lw=1, alpha=0.5)
    ax.axhline(0.55, color="red", ls=":", lw=1, alpha=0.5)
    ax.grid(alpha=0.3)

    # Panel 3: Large/Sci vs median HV (per date)
    ax = axes[1, 0]
    ax.scatter(by_date["hv"], by_date["large_frac"], s=30, alpha=0.7, c="C2")
    ax.set_xlabel("median HV [V]")
    ax.set_ylabel("Large/Sci median per date")
    ax.set_title("Large/Sci vs HV (date-level)\n"
                 "If HV-dependent: should show monotonic trend")
    ax.grid(alpha=0.3)

    # Panel 4: Heatmap: date × det
    ax = axes[1, 1]
    grp["det_global"] = grp["box"].map(BOX_OFFSET) + grp["det"].astype(int)
    pivot = grp.pivot_table(index="date", columns="det_global", values="large_frac", aggfunc="median")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xlabel("det_global (0-5: Box A, 6-11: Box B, 12-17: Box C)")
    ax.set_ylabel("date index")
    ax.set_title("Heatmap: Large/Sci by date × det")
    fig.colorbar(im, ax=ax, label="Large/Sci")
    # Mark blind det 16
    ax.axvline(16, color="black", ls="--", lw=1, alpha=0.7)
    ax.text(16.2, 5, "det 16\n(blind)", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "diag_large_mode_origin.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ========== 5. Quantify: variance from date vs det ==========
    overall_med = grp["large_frac"].median()
    var_total = grp["large_frac"].var()
    var_within_date = grp.groupby("date")["large_frac"].var().mean()
    var_within_det = grp.groupby(["box", "det"])["large_frac"].var().mean()
    print(f"\n=== Variance decomposition ===")
    print(f"  Overall Large/Sci variance:           {var_total:.4f}")
    print(f"  Mean variance WITHIN each date:        {var_within_date:.4f}")
    print(f"  Mean variance WITHIN each (box,det):   {var_within_det:.4f}")
    print(f"  → If date dominates: var_within_date < var_total")
    print(f"  → If det dominates:  var_within_det < var_total")


if __name__ == "__main__":
    main()
