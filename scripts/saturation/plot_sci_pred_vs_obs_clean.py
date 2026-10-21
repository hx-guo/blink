#!/usr/bin/env python3
"""Reproduce Sci_pred vs Sci_obs density plot using M1 fit on CLEAN band
(Sci ∈ [400, 1000] per det, group_rate < 6000) — current best model.

For comparison: original plot used M1 fit on Sci > 300 (FIFO-contaminated).
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

# CLEAN band — current best fit window
SCI_LO_CLEAN = 400.0
SCI_HI_CLEAN = 1000.0
BOX_RATE_CAP = 6000.0


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
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["group_rate"]  = df["sci_sec_total"] / df["length"]
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
    print(f"  normal-mode rows: {len(df):,}")
    return df


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef  # b, 1+α, β, γ


def main():
    df = load()

    # Fit on CLEAN band per Box (ALL modes mixed — no HIGH/LOW split for visualization)
    print(f"\n=== M1 fit on CLEAN band [Sci ∈ [{SCI_LO_CLEAN}, {SCI_HI_CLEAN}]/det, "
          f"box_rate < {BOX_RATE_CAP}] ===")
    fits = {}
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
                & (df["group_rate"] < BOX_RATE_CAP))
        coef = fit_m1(df[mask])
        fits[box] = coef
        print(f"  Box {box} (N={mask.sum():,}): "
              f"b={coef[0]:+.1f}, 1+α={coef[1]:.4f} (α={coef[1]-1:+.4f}), "
              f"β={coef[2]:.4f}, γ={coef[3]:.4f}")

    # Compute Sci_pred for all data
    df["box_str"] = df["box"].astype(str)
    df["b_v"] = df["box_str"].map(lambda b: fits[b][0])
    df["c1plus_v"] = df["box_str"].map(lambda b: fits[b][1])
    df["beta_v"] = df["box_str"].map(lambda b: fits[b][2])
    df["gamma_v"] = df["box_str"].map(lambda b: fits[b][3])
    df["sci_pred"] = ((df["pho_rate"] - df["beta_v"]*df["wide_rate"]
                       - df["gamma_v"]*df["large_rate"] - df["b_v"])
                       / df["c1plus_v"])

    # Compute RMS on Sci > 300 (matching original plot scope)
    rms_main = {}
    for box in "ABC":
        mask = (df["box"]==box) & (df["sci_rate"] > 300)
        sub = df[mask]
        rms_main[box] = np.sqrt(((sub["sci_pred"] - sub["sci_rate"])**2).mean())

    # Plot 3 boxes vertically (matching original style)
    fig, axes = plt.subplots(3, 1, figsize=(8, 14), sharex=True)
    xy_bins = np.logspace(np.log10(40), np.log10(4500), 200)

    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"]==box]
        N = len(sub)
        # 2D density
        H, xedges, yedges = np.histogram2d(
            sub["sci_rate"].values, sub["sci_pred"].values.clip(0.5, 1e5),
            bins=[xy_bins, xy_bins])
        X, Y = np.meshgrid(xedges, yedges)
        pcm = ax.pcolormesh(X, Y, H.T, norm=LogNorm(vmin=1, vmax=H.max()),
                             cmap="viridis", shading="auto")

        # y=x line
        line = np.array([50, 4500])
        ax.plot(line, line, "--", color="red", lw=1.5,
                label=f"y=x  fit_main: β={fits[box][2]:.3f}, γ={fits[box][3]:.3f}, "
                       f"α={fits[box][1]-1:+.3f}, b={fits[box][0]:+.0f}  RMS={rms_main[box]:.1f}")
        ax.axvline(SCI_LO_CLEAN, color="purple", ls=":", lw=1, alpha=0.6,
                    label=f"clean fit lo: Sci={SCI_LO_CLEAN:.0f}")
        ax.axvline(SCI_HI_CLEAN, color="purple", ls=":", lw=1, alpha=0.6,
                    label=f"clean fit hi: Sci={SCI_HI_CLEAN:.0f}")

        # Binned median for Sci > 300
        bins_med = np.logspace(np.log10(300), np.log10(4500), 30)
        bc = 0.5 * (bins_med[:-1] + bins_med[1:])
        med = np.array([
            sub.loc[(sub["sci_rate"] >= bins_med[i]) & (sub["sci_rate"] < bins_med[i+1]),
                    "sci_pred"].median() if ((sub["sci_rate"] >= bins_med[i])
                                              & (sub["sci_rate"] < bins_med[i+1])).sum() > 100
                    else np.nan
            for i in range(len(bins_med)-1)
        ])
        ax.plot(bc, med, "-", color="orange", lw=2, label="binned median (Sci > 300)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(40, 4500)
        ax.set_ylim(40, 4500)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted = (PHO − β·W − γ·L − b)/(1+α) [cnt/s/det]")
        ax.set_title(f"Box {box}  (N={N:,})  RMS_main = {rms_main[box]:.0f}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, which="both")

    # Color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label="per-det-sec bin count (log scale)")

    fig.suptitle(f"M1 fit on CLEAN band (Sci ∈ [{SCI_LO_CLEAN:.0f}, {SCI_HI_CLEAN:.0f}]/det, "
                 f"group_rate < {BOX_RATE_CAP:.0f}). Coefficients applied to full data.",
                 fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.96, bottom=0.04, hspace=0.18)
    out = OUT_DIR / "sci_pred_vs_obs_M1_CLEAN.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
