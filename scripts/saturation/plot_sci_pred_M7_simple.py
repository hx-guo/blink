#!/usr/bin/env python3
"""Simple Sci_pred vs Sci_obs density plot.

Two panels per box:
  Left:  standard density (color = log count per bin)
  Right: hexbin colored by median per-second Large/Sci ratio
         (shows whether each region has high or low Large/Sci, without
          arbitrary HIGH/LOW classification)

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

    # Fit M7 CLEAN
    fits = {}
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
                & (df["group_rate"] < BOX_RATE_CAP))
        fits[box] = fit_m7(df[mask])

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

    # Plot: 3 rows × 2 cols. Left: density. Right: hexbin colored by Large/Sci.
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    xy_bins = np.logspace(np.log10(40), np.log10(4500), 200)

    for row, box in enumerate("ABC"):
        sub = df[df["box"]==box]
        c = fits[box]
        b, c0, c1, cN, beta, gamma = c

        # ---- Left: standard density ----
        ax = axes[row, 0]
        H, xedges, yedges = np.histogram2d(
            sub["sci_rate"].values, sub["sci_pred"].values.clip(0.5, 1e5),
            bins=[xy_bins, xy_bins])
        X, Y = np.meshgrid(xedges, yedges)
        pcm = ax.pcolormesh(X, Y, H.T, norm=LogNorm(vmin=1, vmax=H.max()),
                             cmap="viridis", shading="auto")
        line = np.array([50, 4500])
        ax.plot(line, line, "--", color="red", lw=1.5, label="y=x")
        ax.plot(line, 2*line, ":", color="orange", lw=1.2, label="y=2x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(50, 4500); ax.set_ylim(20, 6000)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.set_title(f"Box {box}: density (log count)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.colorbar(pcm, ax=ax, label="count")

        # ---- Right: hexbin colored by median Large/Sci ----
        ax = axes[row, 1]
        # Filter to plottable range
        m = (sub["sci_rate"] >= 50) & (sub["sci_rate"] <= 4500) & \
            (sub["sci_pred"] >= 20) & (sub["sci_pred"] <= 6000)
        s = sub[m]
        hb = ax.hexbin(s["sci_rate"], s["sci_pred"],
                       C=s["large_frac"],
                       reduce_C_function=np.median,
                       gridsize=80, xscale="log", yscale="log",
                       cmap="RdYlGn_r", vmin=0, vmax=1.0, mincnt=3)
        ax.plot(line, line, "--", color="black", lw=1.5, label="y=x")
        ax.plot(line, 2*line, ":", color="black", lw=1.2, label="y=2x")
        ax.set_xlim(50, 4500); ax.set_ylim(20, 6000)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.set_title(f"Box {box}: cells colored by median Large/Sci  "
                      f"(red=low, green=high)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.colorbar(hb, ax=ax, label="median Large/Sci per cell")

    fig.suptitle("M7 CLEAN: Sci_pred vs Sci_obs density",
                 fontsize=12, y=0.998)
    fig.tight_layout()
    out = OUT_DIR / "sci_pred_M7_simple.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
