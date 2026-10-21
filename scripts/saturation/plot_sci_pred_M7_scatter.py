#!/usr/bin/env python3
"""Real scatter plot of Sci_pred vs Sci_obs (M7 CLEAN), each point colored
by local density. No binned median line. Sci > 300 only."""
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
SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
X_LO = 300
N_SCATTER = 200_000  # subsample for scatter (4M is too many)


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Sci": "int32", "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
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
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd1_rate","Sci_ACD1"),("acdn_rate","Sci_ACDN"),
                    ("wide_rate","Wide"),("large_rate","Large"),
                    ("pho_rate","PHO")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
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
    return df


def main():
    df = load()
    print(f"rows: {len(df):,}")

    # Fit M7 CLEAN
    fits = {}
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
                & (df["group_rate"] < BOX_RATE_CAP))
        s = df[mask]
        X = np.column_stack([np.ones(len(s)), s["scipure_rate"], s["acd1_rate"],
                              s["acdn_rate"], s["wide_rate"], s["large_rate"]])
        fits[box], *_ = np.linalg.lstsq(X, s["pho_rate"].values, rcond=None)

    df["box_str"] = df["box"].astype(str)
    b   = df["box_str"].map(lambda b: fits[b][0])
    c0  = df["box_str"].map(lambda b: fits[b][1])
    c1  = df["box_str"].map(lambda b: fits[b][2])
    cN  = df["box_str"].map(lambda b: fits[b][3])
    bet = df["box_str"].map(lambda b: fits[b][4])
    gam = df["box_str"].map(lambda b: fits[b][5])
    df["sci_pred"] = ((df["pho_rate"] - (c1-c0)*df["acd1_rate"]
                       - (cN-c0)*df["acdn_rate"]
                       - bet*df["wide_rate"] - gam*df["large_rate"] - b) / c0)

    # ============ Plot ============
    fig, axes = plt.subplots(3, 1, figsize=(8, 14), sharex=True)
    xb = np.logspace(np.log10(X_LO), np.log10(4500), 150)
    yb = np.logspace(np.log10(X_LO/2), np.log10(7000), 150)

    for ax, box in zip(axes, "ABC"):
        sub = df[(df["box"]==box) & (df["sci_rate"] >= X_LO)
                  & (df["sci_pred"] > 0)]
        # density lookup per point via 2D histogram
        H, xedges, yedges = np.histogram2d(sub["sci_rate"].values,
                                             sub["sci_pred"].values,
                                             bins=[xb, yb])
        ix = np.clip(np.searchsorted(xedges, sub["sci_rate"].values) - 1,
                     0, len(xedges)-2)
        iy = np.clip(np.searchsorted(yedges, sub["sci_pred"].values) - 1,
                     0, len(yedges)-2)
        density = H[ix, iy].astype(float)
        density[density < 1] = 1

        # subsample
        if len(sub) > N_SCATTER:
            idx = np.random.RandomState(0).choice(len(sub), N_SCATTER, replace=False)
        else:
            idx = np.arange(len(sub))
        x_plot = sub["sci_rate"].values[idx]
        y_plot = sub["sci_pred"].values[idx]
        c_plot = density[idx]

        # sort by density so dense points draw on top
        order = np.argsort(c_plot)
        sc = ax.scatter(x_plot[order], y_plot[order], c=c_plot[order],
                         cmap="viridis", norm=LogNorm(vmin=1, vmax=density.max()),
                         s=2, alpha=0.7, rasterized=True, edgecolor="none")

        line = np.array([X_LO, 4500])
        c = fits[box]
        ax.plot(line, line, "--", color="red", lw=1.5,
                label=f"y=x   M7: c0={c[1]:.2f}, c1={c[2]:.2f}, cN={c[3]:.2f}, "
                       f"β={c[4]:.2f}, γ={c[5]:.2f}")
        ax.plot(line, 2*line, ":", color="orange", lw=1.2, alpha=0.7, label="y=2x")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(X_LO, 4500); ax.set_ylim(X_LO/2, 7000)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted (M7 CLEAN) [cnt/s/det]")
        ax.set_title(f"Box {box}  (N={len(sub):,}, scatter subsampled to {len(idx):,})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, which="both")
        fig.colorbar(sc, ax=ax, label="local point density (log)")

    fig.suptitle(f"M7 CLEAN scatter density (Sci > {X_LO})", fontsize=11, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "sci_pred_M7_scatter.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
