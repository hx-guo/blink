#!/usr/bin/env python3
"""Sci_pred vs Sci_obs scatter plot colored by mode classification.

Goal: visually confirm that:
  - Upper cloud (parallel band above y=x) = LOW-Large mode bins
  - Main band = HIGH-Large mode bins
  - Sci<300 anomaly (FIFO contamination) = separately marked

Uses M7 CLEAN coefficients (current best model).
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
    df["large_frac"]    = df["large_rate"] / df["sci_rate"].clip(lower=1)
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

    # Classify date-level mode
    main = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    by_date = main.groupby("date").agg(
        large_frac=("large_frac", "median"),
        N=("large_frac", "count"),
    )
    by_date = by_date[by_date["N"] > 200]
    HIGH_TH, LOW_TH = 0.55, 0.40
    high_dates = set(by_date[by_date["large_frac"] > HIGH_TH].index)
    low_dates  = set(by_date[by_date["large_frac"] < LOW_TH].index)
    df["mode"] = np.where(df["date"].isin(high_dates), "HIGH",
                          np.where(df["date"].isin(low_dates), "LOW", "MID"))

    # Fit M7 CLEAN (single, all modes mixed)
    fits = {}
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
                & (df["group_rate"] < BOX_RATE_CAP))
        fits[box] = fit_m7(df[mask])
        b, c0, c1, cN, beta, gamma = fits[box]
        print(f"  Box {box}: b={b:+.1f}, c0={c0:.3f}, c1={c1:.3f}, cN={cN:.3f}, "
              f"β={beta:.3f}, γ={gamma:.3f}")

    # Compute Sci_pred
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

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 14), sharex=True)
    for ax, box in zip(axes, "ABC"):
        sub = df[(df["box"]==box) & (df["sci_rate"] > 80)]

        # Plot each mode separately with different color
        for mode, color, alpha, size in [
            ("MID", "lightgray", 0.15, 0.2),
            ("HIGH", "tab:blue", 0.25, 0.3),
            ("LOW", "tab:red", 0.35, 0.5),
        ]:
            sub_m = sub[sub["mode"]==mode]
            ax.scatter(sub_m["sci_rate"], sub_m["sci_pred"].clip(0.5, 1e5),
                        s=size, color=color, alpha=alpha,
                        label=f"{mode} mode (N={len(sub_m):,})", rasterized=True)

        # Reference lines
        line = np.array([50, 5000])
        ax.plot(line, line, "--", color="black", lw=1.5, label="y=x (perfect fit)")
        ax.plot(line, 2*line, ":", color="darkorange", lw=1.2,
                label="y=2x (LOW mode prediction)")
        ax.axvline(SCI_LO_CLEAN, color="purple", ls=":", lw=1, alpha=0.5)
        ax.axvline(SCI_HI_CLEAN, color="purple", ls=":", lw=1, alpha=0.5)
        ax.axvspan(50, 300, alpha=0.08, color="brown",
                   label="Sci<300 (FIFO heavily contaminated)")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(50, 4500); ax.set_ylim(20, 6000)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted (M7 CLEAN) [cnt/s/det]")
        ax.set_title(f"Box {box}: density colored by date-level Large/Sci mode")
        # Custom legend handles for size visibility
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8, loc="upper left",
                  markerscale=20, framealpha=0.95)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("M7 CLEAN density colored by mode\n"
                 "→ Upper band (~2× y=x) = LOW-Large mode (red)\n"
                 "→ Main band on y=x = HIGH-Large mode (blue)\n"
                 "→ Sci<300 (brown shade) = FIFO heavily-contaminated bins (separate phenomenon)",
                 fontsize=11, y=0.998)
    fig.tight_layout()
    out = OUT_DIR / "sci_pred_M7_mode_colored.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
