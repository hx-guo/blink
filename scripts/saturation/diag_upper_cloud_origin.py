#!/usr/bin/env python3
"""Diagnose the upper cloud in M7 CLEAN density plot.

Hypothesis: upper cloud = FIFO-drop affected bins in 2017-2019 normal data.
  - Sci_obs is undercount (1B events dropped)
  - PHO_eng/Wide_eng/Large_eng are correct
  - Model gives Sci_pred ≈ Sci_true > Sci_obs → ABOVE y=x

Test: color the density plot by Dt/L_cycles (dead-time fraction). If upper
cloud has high Dt/L, FIFO drops are confirmed.

Also test: group_rate (box total rate). High group_rate → FIFO drops likely.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

SCI_LO_CLEAN = 400.0
SCI_HI_CLEAN = 1000.0
BOX_RATE_CAP = 6000.0


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
    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["sci_rate"]      = df["Sci"]      / df["length"]
    df["scipure_rate"]  = df["Sci_pure"] / df["length"]
    df["acd1_rate"]     = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]     = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]     = df["Wide"]     / df["length"]
    df["large_rate"]    = df["Large"]    / df["length"]
    df["pho_rate"]      = df["PHO"]      / df["length"]
    df["group_rate"]    = df["sci_sec_total"] / df["length"]
    df["dt_frac"]       = df["Dt"] / df["L_cycles"]
    df["det_global"]    = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
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
    print(f"  normal-mode rows: {len(df):,}")
    return df


def fit_m7(sub):
    X = np.column_stack([np.ones(len(sub)),
                         sub["scipure_rate"].values,
                         sub["acd1_rate"].values,
                         sub["acdn_rate"].values,
                         sub["wide_rate"].values,
                         sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def main():
    df = load()

    # Fit M7 on CLEAN band
    fits = {}
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
                & (df["group_rate"] < BOX_RATE_CAP))
        fits[box] = fit_m7(df[mask])

    # Compute Sci_pred for all data
    df["box_str"] = df["box"].astype(str)
    df["b_v"]  = df["box_str"].map(lambda b: fits[b][0])
    df["c0_v"] = df["box_str"].map(lambda b: fits[b][1])
    df["c1_v"] = df["box_str"].map(lambda b: fits[b][2])
    df["cN_v"] = df["box_str"].map(lambda b: fits[b][3])
    df["beta_v"]  = df["box_str"].map(lambda b: fits[b][4])
    df["gamma_v"] = df["box_str"].map(lambda b: fits[b][5])
    df["sci_pred"] = ((df["pho_rate"]
                       - (df["c1_v"] - df["c0_v"]) * df["acd1_rate"]
                       - (df["cN_v"] - df["c0_v"]) * df["acdn_rate"]
                       - df["beta_v"] * df["wide_rate"]
                       - df["gamma_v"] * df["large_rate"]
                       - df["b_v"]) / df["c0_v"])

    # Define cloud membership: above main band by > 50%
    df["log_ratio"] = np.log10(df["sci_pred"].clip(lower=1) / df["sci_rate"].clip(lower=1))
    df["upper_cloud"] = (df["log_ratio"] > 0.2) & (df["sci_rate"] > 300)  # log10(1.6) ≈ 0.2 → ratio > 1.6
    df["main_band"] = (df["log_ratio"].abs() < 0.1) & (df["sci_rate"] > 300)  # within ±26%

    print(f"\n=== Cloud counts ===")
    for box in "ABC":
        sub = df[df["box"]==box]
        n_upper = sub["upper_cloud"].sum()
        n_main = sub["main_band"].sum()
        print(f"  Box {box}: upper cloud N={n_upper:,} ({100*n_upper/len(sub):.2f}%)  "
              f"main band N={n_main:,} ({100*n_main/len(sub):.2f}%)")

    # ============ Compare upper cloud vs main band properties ============
    print(f"\n=== Property comparison: upper cloud vs main band ===")
    print(f"  Property            |  Main band    Upper cloud   Ratio U/M")
    for prop in ["dt_frac", "group_rate", "sci_rate", "large_rate",
                 "large_frac" if False else "large_rate"]:
        if prop == "large_frac": continue  # skip
        main_med = df.loc[df["main_band"], prop].median()
        upper_med = df.loc[df["upper_cloud"], prop].median()
        ratio = upper_med / main_med if main_med != 0 else np.nan
        print(f"  {prop:>18s}  |  {main_med:>10.4f}   {upper_med:>10.4f}    {ratio:>5.2f}")

    # Large/Sci specifically
    df["large_frac"] = df["large_rate"] / df["sci_rate"].clip(lower=1)
    print(f"  {'Large/Sci':>18s}  |  "
          f"{df.loc[df['main_band'],'large_frac'].median():>10.4f}   "
          f"{df.loc[df['upper_cloud'],'large_frac'].median():>10.4f}    "
          f"{df.loc[df['upper_cloud'],'large_frac'].median()/df.loc[df['main_band'],'large_frac'].median():>5.2f}")

    # ============ Plot: density colored by group_rate ============
    fig, axes = plt.subplots(3, 1, figsize=(8, 14), sharex=True)

    for ax, box in zip(axes, "ABC"):
        sub = df[(df["box"]==box) & (df["sci_rate"] > 100)]
        # Scatter colored by Dt fraction
        sc = ax.scatter(sub["sci_rate"], sub["sci_pred"].clip(0.5, 1e5),
                         c=sub["dt_frac"], cmap="plasma",
                         s=0.5, alpha=0.3, vmin=0, vmax=0.15)
        line = np.array([50, 4500])
        ax.plot(line, line, "--", color="red", lw=1.5, label="y=x")
        ax.plot(line, 2*line, ":", color="blue", lw=1, alpha=0.7,
                label="y=2x (50% FIFO drop)")
        ax.plot(line, 1.5*line, ":", color="cyan", lw=1, alpha=0.5,
                label="y=1.5x (33% drop)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(80, 4500); ax.set_ylim(50, 6000)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.set_title(f"Box {box}: density colored by Dt/L_cycles (dead-time fraction)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, which="both")
        if box == "A":
            fig.colorbar(sc, ax=ax, label="Dt / L_cycles")

    fig.suptitle("Diagnosis: is the upper cloud caused by FIFO drops?\n"
                 "If yes, upper cloud bins should have HIGH Dt/L (yellow-purple)",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "diag_upper_cloud_origin.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
