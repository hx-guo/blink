#!/usr/bin/env python3
"""Test pile-up hypothesis: does the S-curve residual depend on GROUP rate
(sci_sec_total, the 6-det sum that shares 1 PDAU/ADC), not just per-det Sci?

If pile-up: at fixed per-det Sci, residual should vary with group rate.
  - When the OTHER 5 dets in the group are also busy → more ADC collisions →
    more pile-up → bigger residual.
  - When the others are quiet → less pile-up → smaller residual.

Diagnostic plan:
1. Compute group_rate = sci_sec_total / length (total Sci in box per second).
2. Slice rows into Sci bins.
3. Within each Sci bin, slice by group_rate quantile.
4. Plot residual_M1 median vs Sci, color-coded by group_rate quartile.

If residual SEPARATES by group_rate at fixed per-det Sci → pile-up confirmed.
If residual is the same across group_rate → not pile-up, look elsewhere.
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
MAIN_BAND_LO = 300.0


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
    df["group_rate"]  = df["sci_sec_total"] / df["length"]  # 6-det sum per second
    df["other_rate"]  = df["group_rate"] - df["sci_rate"]   # 5 OTHER dets in this box
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


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef  # [b, 1+α, β, γ]


def predict_m1_resid(sub, coef):
    b, c1, beta, gamma = coef
    pho_pred = b + c1*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values
    sci_pred = (sub["pho_rate"].values - b - beta*sub["wide_rate"].values - gamma*sub["large_rate"].values) / c1
    return sci_pred - sub["sci_rate"].values


def main():
    df = load()

    # Fit M1 per box (just for residual computation)
    print("\n=== M1 coefs per box ===")
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        coef = fit_m1(df[mask_fit])
        print(f"  Box {box}: b={coef[0]:.1f}, 1+α={coef[1]:.4f}, β={coef[2]:.4f}, γ={coef[3]:.4f}")
        mask_apply = df["box"] == box
        df.loc[mask_apply, "resid_M1"] = predict_m1_resid(df[mask_apply], coef)

    # ============ Pile-up signature test ============
    # At fixed sci_rate, does residual depend on OTHER dets' rate (other_rate)?
    print(f"\n=== Pile-up signature: residual vs other-dets rate, sliced by Sci bin ===")
    sci_bins = [(300,600), (600,1000), (1000,1500), (1500,2000), (2000,3000)]

    # For each sci_rate bin, slice by other_rate quartile
    fig, axes = plt.subplots(1, len(sci_bins), figsize=(20, 4.5), sharey=False)
    for ax, (lo, hi) in zip(axes, sci_bins):
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        sub = df[mask].copy()
        if len(sub) < 100:
            ax.set_title(f"Sci {lo}-{hi}: too few"); continue

        # other_rate quartiles within this Sci bin
        q = sub["other_rate"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
        edges = q
        bin_centers, meds, counts = [], [], []
        # Linear-spaced quartile bins
        for i in range(4):
            qm = (sub["other_rate"] >= edges[i]) & (sub["other_rate"] < edges[i+1])
            if qm.sum() < 50: continue
            bin_centers.append(sub.loc[qm, "other_rate"].median())
            meds.append(sub.loc[qm, "resid_M1"].median())
            counts.append(qm.sum())
        ax.plot(bin_centers, meds, "o-", color="C0", lw=2, markersize=8)
        for x, y, n in zip(bin_centers, meds, counts):
            ax.annotate(f"n={n//1000}k", (x, y), fontsize=7, textcoords="offset points",
                        xytext=(0, 8), ha="center")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xlabel("other-5-dets total rate [cnt/s]")
        ax.set_title(f"Sci {lo}-{hi} cnt/s/det")
        if lo == 300:
            ax.set_ylabel("median M1 residual [cnt/s/det]")
        ax.grid(alpha=0.3)
    fig.suptitle("Pile-up signature test: M1 residual vs OTHER-5-dets rate, at fixed per-det Sci\n"
                 "(If pile-up: residual should INCREASE with other_rate at mid-Sci, "
                 "DECREASE at high-Sci)", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "diag_pileup_signature.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ============ Numeric table ============
    print(f"\n=== Numeric: median resid_M1 by (Sci bin × other-rate quartile) ===")
    print(f"{'Sci bin':>15s}  {'Q1 other':>12s}  {'Q2':>10s}  {'Q3':>10s}  {'Q4':>10s}  {'Q4-Q1':>10s}")
    for lo, hi in sci_bins:
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        sub = df[mask]
        if len(sub) < 500: continue
        q = sub["other_rate"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
        meds = []
        for i in range(4):
            qm = (sub["other_rate"] >= q[i]) & (sub["other_rate"] < q[i+1])
            meds.append(sub.loc[qm, "resid_M1"].median() if qm.sum() > 50 else np.nan)
        row = f"  {lo:>5d}-{hi:>5d}  " + "  ".join(f"{m:>+10.1f}" for m in meds)
        row += f"  {meds[3]-meds[0]:>+10.1f}"
        print(row)


if __name__ == "__main__":
    main()
