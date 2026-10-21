#!/usr/bin/env python3
"""Verify whether Sci_ACDN is a subset of Sci (i.e., Sci_ACDN <= Sci pointwise).

This is critical for interpreting M6a's coefficients:
  M6a: PHO = (1+α')Sci + δ_N·Sci_ACDN + ...

If Sci_ACDN ⊆ Sci (i.e., Sci_ACDN counts a subset of Sci events that also
triggered ACD), then we can rewrite:
  Sci = Sci_pure + Sci_ACDN  (Sci_pure = Sci events without ACD coincidence)
  PHO = (1+α')·Sci_pure + (1+α'+δ_N)·Sci_ACDN + ...

  So the "effective" PHO/Sci ratio decomposes into:
    - Pure-Sci channel:   (1+α')
    - ACDN-coinc channel: (1+α'+δ_N)

If the data show Sci_ACDN > Sci sometimes, then it's an independent counter
(e.g., HVT events that just happened to be tagged), and M6a's interpretation
is different.

Also examine:
  - Distribution of Sci_ACDN/Sci ratio
  - Where in (Sci, det, box) space does the ratio peak
  - Same for Sci_ACD1
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
    N = len(df)

    # ============ Subset test ============
    print(f"\n=== Test 1: Is Sci_ACDN a subset of Sci? ===")
    print(f"Total rows: {N:,}")

    acdn_gt_sci = (df["Sci_ACDN"] > df["Sci"]).sum()
    acd1_gt_sci = (df["Sci_ACD1"] > df["Sci"]).sum()
    acdn_eq_sci = (df["Sci_ACDN"] == df["Sci"]).sum()
    acdn_zero   = (df["Sci_ACDN"] == 0).sum()
    sci_zero    = (df["Sci"] == 0).sum()

    print(f"  Sci_ACDN > Sci : {acdn_gt_sci:,} ({100*acdn_gt_sci/N:.3f}%)")
    print(f"  Sci_ACDN = Sci : {acdn_eq_sci:,} ({100*acdn_eq_sci/N:.3f}%)")
    print(f"  Sci_ACDN = 0   : {acdn_zero:,} ({100*acdn_zero/N:.3f}%)")
    print(f"  Sci = 0        : {sci_zero:,} ({100*sci_zero/N:.3f}%)")
    print(f"  Sci_ACD1 > Sci : {acd1_gt_sci:,} ({100*acd1_gt_sci/N:.3f}%)")

    # Combined: ACD1 + ACDN > Sci?
    sum_gt_sci = (df["Sci_ACD1"] + df["Sci_ACDN"] > df["Sci"]).sum()
    print(f"  Sci_ACD1 + Sci_ACDN > Sci : {sum_gt_sci:,} ({100*sum_gt_sci/N:.3f}%)")

    # ============ ACDN / Sci ratio distribution ============
    print(f"\n=== Test 2: Sci_ACDN/Sci ratio statistics (Sci > 0) ===")
    sub = df[df["Sci"] > 100].copy()
    sub["acdn_frac"] = sub["Sci_ACDN"] / sub["Sci"]
    sub["acd1_frac"] = sub["Sci_ACD1"] / sub["Sci"]
    print(f"  ACDN/Sci percentiles: "
          f"5%={np.percentile(sub['acdn_frac'],5):.4f}, "
          f"50%={np.percentile(sub['acdn_frac'],50):.4f}, "
          f"95%={np.percentile(sub['acdn_frac'],95):.4f}, "
          f"max={sub['acdn_frac'].max():.4f}")
    print(f"  ACD1/Sci percentiles: "
          f"5%={np.percentile(sub['acd1_frac'],5):.4f}, "
          f"50%={np.percentile(sub['acd1_frac'],50):.4f}, "
          f"95%={np.percentile(sub['acd1_frac'],95):.4f}, "
          f"max={sub['acd1_frac'].max():.4f}")

    # ============ Cross-check: ACDN at high Sci ============
    print(f"\n=== Test 3: ACDN fraction by Sci bin ===")
    bin_edges = [100, 300, 600, 1000, 1500, 2000, 2500, 3500, 5000]
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'med ACDN/Sci':>14s}  {'med ACD1/Sci':>14s}  "
          f"{'med (ACD1+ACDN)/Sci':>20s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        m = (sub["sci_rate"] >= lo) & (sub["sci_rate"] < hi)
        if m.sum() < 100:
            continue
        med_n = sub.loc[m, "acdn_frac"].median()
        med_1 = sub.loc[m, "acd1_frac"].median()
        sumfrac = (sub.loc[m, "Sci_ACD1"] + sub.loc[m, "Sci_ACDN"]) / sub.loc[m, "Sci"]
        med_sum = sumfrac.median()
        print(f"  {lo:>5d}-{hi:>5d}  {m.sum():>10,d}  "
              f"{med_n:>14.4f}  {med_1:>14.4f}  {med_sum:>20.4f}")

    # ============ Plot: histogram of Sci_ACDN/Sci ratio ============
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ["acdn_frac", "acd1_frac"],
                              ["Sci_ACDN / Sci", "Sci_ACD1 / Sci"]):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub_b = sub[sub["box"] == box]
            ax.hist(sub_b[col].clip(0, 1.5), bins=80, range=(0, 1.5),
                    histtype="step", color=color, lw=1.5, label=f"Box {box}", density=True)
        ax.axvline(1.0, color="red", ls="--", lw=1, label="ratio = 1 (subset)")
        ax.set_xlabel(title)
        ax.set_ylabel("normalized count")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Distribution of ACD-coincidence fraction\n"
                 "(>1 means ACDN is NOT a subset of Sci)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "diag_acdn_subset.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ============ Interpretation summary ============
    print(f"\n=== Interpretation for M6a ===")
    if acdn_gt_sci / N < 0.01:
        print("✓ Sci_ACDN IS effectively a subset of Sci (<1% violations).")
        print("  → M6a's δ_N reflects 'extra PHO per ACDN-tagged event'.")
        print("  → Decomposing M6a (Box A: α'=0.69, δ_N=+2.52):")
        print("      Sci = Sci_pure + Sci_ACDN")
        print("      Pure-Sci channel ratio:   (1+α')      = 1.69")
        print("      ACDN-coinc channel ratio: (1+α'+δ_N) = 4.21")
        print("  → ACDN-coincident events bring ~4.2 PHO each vs 1.7 for pure Sci.")
        print("  → Physical: ACD coincidence indicates Compton scattering or")
        print("              charged particle traversal, which deposits energy")
        print("              in multiple detectors hence inflates PHO above Sci.")
    else:
        print("✗ Sci_ACDN is NOT a strict subset of Sci.")
        print("  → Need different interpretation. Check if Sci_ACDN counts something")
        print("    independent (e.g., HVT-only triggers).")


if __name__ == "__main__":
    main()
