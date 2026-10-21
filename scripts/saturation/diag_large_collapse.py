#!/usr/bin/env python3
"""Diagnose the Large channel collapse around Sci=1000-1500 and verify
the cleanest fit region.

Two questions:
  1. Is Sci=2000 already in saturation? PHO/Sci peaked at Sci=1500 then drops.
     → test by refitting M7 in tighter region [400, 1500] and [400, 1200].
  2. What causes the bump in additive/multiplicative residuals at Sci=1000?
     Hypothesis: Large/Sci ratio drops from 0.74 to 0.35 between Sci=1000 and
     Sci=1500. Is this:
       (a) Continuous physics (pile-up shifting Large up to overflow)?
       (b) Bimodal population (two operating modes mixed)?
       (c) Some date-specific anomaly?

Plot:
  - histogram of Large/Sci within Sci bins → bimodality test
  - Large/Sci median by date → temporal stability test
  - Box × Sci × Large/Sci → cross-check
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

    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["large_frac"]  = df["large_rate"] / df["sci_rate"]
    df["wide_frac"]   = df["wide_rate"]  / df["sci_rate"]
    df["pho_frac"]    = df["pho_rate"]   / df["sci_rate"]
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

    # ========== Test 1: bimodality of Large/Sci within Sci bins ==========
    print(f"\n=== Test 1: histogram of Large/Sci within Sci bins ===")
    sci_bins_for_hist = [(400, 600), (600, 1000), (1000, 1200), (1200, 1500),
                         (1500, 1800), (1800, 2200)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (lo, hi) in zip(axes.flatten(), sci_bins_for_hist):
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        sub = df[mask]
        ax.hist(sub["large_frac"].clip(0, 2), bins=80, range=(0, 2),
                color="C2", alpha=0.7, edgecolor="black", lw=0.3)
        med = sub["large_frac"].median()
        ax.axvline(med, color="red", ls="--", lw=2, label=f"median={med:.3f}")
        ax.set_xlabel("Large / Sci")
        ax.set_ylabel("count")
        ax.set_title(f"Sci {lo}-{hi} (N={mask.sum():,})")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Test bimodality of Large/Sci ratio within fixed Sci bins\n"
                 "(if bimodal: two populations mixed; if unimodal: smooth physics)",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "diag_large_bimodality.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ========== Test 2: Large/Sci by DATE ==========
    print(f"\n=== Test 2: Large/Sci by date (Sci 1000-1500 main band) ===")
    sub = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)]
    by_date = sub.groupby("date")["large_frac"].agg(["median", "count"])
    by_date = by_date[by_date["count"] > 100]
    print(f"  Dates with >100 rows in Sci 1000-1500: {len(by_date)}")
    print(f"  Large/Sci median:  min={by_date['median'].min():.3f}, "
          f"max={by_date['median'].max():.3f}, "
          f"mean={by_date['median'].mean():.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    ax.scatter(range(len(by_date)), by_date["median"], c=by_date["count"],
               cmap="viridis", s=20, alpha=0.7)
    ax.set_xlabel("date index (sorted)")
    ax.set_ylabel("Large/Sci median (Sci 1000-1500)")
    ax.set_title(f"Large/Sci by date in main band (Sci 1000-1500): "
                 f"{len(by_date)} dates, color=N rows")
    ax.grid(alpha=0.3)

    # ========== Test 3: Large/Sci median by Sci bin for different date subsets ==========
    sci_bins = np.logspace(np.log10(300), np.log10(3000), 30)
    bc = 0.5 * (sci_bins[:-1] + sci_bins[1:])

    # Compare full dataset to "low Large/Sci dates" and "high Large/Sci dates"
    sub = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)]
    by_date_main = sub.groupby("date")["large_frac"].median()
    low_dates = by_date_main[by_date_main < 0.4].index
    high_dates = by_date_main[by_date_main > 0.55].index
    print(f"  'low Large/Sci' dates (median < 0.4): {len(low_dates)}")
    print(f"  'high Large/Sci' dates (median > 0.55): {len(high_dates)}")

    ax = axes[1]
    for label, dates, color in [
        ("All data", None, "black"),
        (f"'low Large/Sci' dates ({len(low_dates)})", low_dates, "C0"),
        (f"'high Large/Sci' dates ({len(high_dates)})", high_dates, "C3"),
    ]:
        if dates is None:
            sub_d = df
        else:
            sub_d = df[df["date"].isin(dates)]
        med_large = np.array([
            sub_d.loc[(sub_d["sci_rate"] >= sci_bins[i]) & (sub_d["sci_rate"] < sci_bins[i+1]),
                      "large_frac"].median()
            for i in range(len(sci_bins) - 1)
        ])
        ax.plot(bc, med_large, "-", color=color, lw=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Sci [cnt/s/det]")
    ax.set_ylabel("Large/Sci median")
    ax.set_title("Large/Sci vs Sci — split by date subsets")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / "diag_large_by_date.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ========== Test 4: refit M7 in tighter regions ==========
    print(f"\n=== Test 4: M7 fit in different Sci windows ===")
    fit_ranges = [
        ("400 < Sci < 2000", 400, 2000),
        ("400 < Sci < 1500", 400, 1500),
        ("400 < Sci < 1200", 400, 1200),
        ("400 < Sci < 1000", 400, 1000),
    ]
    print(f"{'Window':>25s}  {'Box':>3s}  {'N':>10s}  "
          f"{'b':>8s}  {'c_pure':>8s}  {'c_ACD1':>8s}  {'c_ACDN':>8s}  "
          f"{'β':>8s}  {'γ':>8s}")
    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["scipure_rate"] = df["Sci_pure"] / df["length"]
    df["acd1_rate"] = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"] = df["Sci_ACDN"] / df["length"]
    for name, lo, hi in fit_ranges:
        for box in "ABC":
            mask = (df["box"] == box) & (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
            sub = df[mask]
            X = np.column_stack([
                np.ones(len(sub)),
                sub["scipure_rate"].values,
                sub["acd1_rate"].values,
                sub["acdn_rate"].values,
                sub["wide_rate"].values,
                sub["large_rate"].values,
            ])
            coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
            b, c0, c1, cN, beta, gamma = coef
            print(f"  {name:>25s}  {box:>3s}  {len(sub):>10,d}  "
                  f"{b:>+8.1f}  {c0:>+8.3f}  {c1:>+8.3f}  {cN:>+8.3f}  "
                  f"{beta:>+8.3f}  {gamma:>+8.3f}")

    # ========== Test 5: very narrow look at Sci=1000-1500 transition ==========
    print(f"\n=== Test 5: detailed Sci scan at 800-1700 ===")
    fine_bins = [(800, 900), (900, 1000), (1000, 1100), (1100, 1200),
                 (1200, 1300), (1300, 1400), (1400, 1500), (1500, 1600), (1600, 1700)]
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'PHO/Sci':>9s}  {'Wide/Sci':>9s}  "
          f"{'Large/Sci':>9s}  {'PHO_med':>9s}  {'Large_med':>9s}")
    for lo, hi in fine_bins:
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        sub = df[mask]
        if len(sub) < 100: continue
        pho_sci = (sub["pho_rate"] / sub["sci_rate"]).median()
        wide_sci = sub["wide_frac"].median()
        large_sci = sub["large_frac"].median()
        pho_med = sub["pho_rate"].median()
        large_med = sub["large_rate"].median()
        print(f"  {lo:>5d}-{hi:>5d}  {len(sub):>10,d}  "
              f"{pho_sci:>9.3f}  {wide_sci:>9.3f}  {large_sci:>9.3f}  "
              f"{pho_med:>9.0f}  {large_med:>9.0f}")


if __name__ == "__main__":
    main()
