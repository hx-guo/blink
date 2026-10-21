#!/usr/bin/env python3
"""M7 fit with iterative Sci_pred < 2000 filtering.

Round 1: initial fit on Sci_obs ∈ [400, 1000] + box_rate < 6000
Round 2-N: refit on rows with Sci_pred (from previous fit) < 2000
Until coefficients converge.

Then plot scatter density (Sci > 300)."""
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

SCI_OBS_LO = 300
SCI_PRED_HI = 2000  # iterative filter target
N_SCATTER = 200_000


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


def fit_m7(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd1_rate"],
                          sub["acdn_rate"], sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_sci_pred(df, fits):
    df = df.copy()
    df["box_str"] = df["box"].astype(str)
    b   = df["box_str"].map(lambda b: fits[b][0]).values
    c0  = df["box_str"].map(lambda b: fits[b][1]).values
    c1  = df["box_str"].map(lambda b: fits[b][2]).values
    cN  = df["box_str"].map(lambda b: fits[b][3]).values
    bet = df["box_str"].map(lambda b: fits[b][4]).values
    gam = df["box_str"].map(lambda b: fits[b][5]).values
    sci_pred = ((df["pho_rate"].values - (c1-c0)*df["acd1_rate"].values
                  - (cN-c0)*df["acdn_rate"].values
                  - bet*df["wide_rate"].values - gam*df["large_rate"].values - b) / c0)
    return sci_pred


def main():
    df = load()
    print(f"rows: {len(df):,}")

    # Initial fit: Sci_obs ∈ [400, 1000], box_rate < 6000
    fits = {}
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= 400) & (df["sci_rate"] < 1000)
                & (df["group_rate"] < 6000))
        fits[box] = fit_m7(df[mask])
        print(f"  Iter 0 Box {box} (N={mask.sum():,}): "
              f"c0={fits[box][1]:.3f}, c1={fits[box][2]:.3f}, "
              f"cN={fits[box][3]:.3f}, β={fits[box][4]:.3f}, γ={fits[box][5]:.3f}")

    # Iterative refit using Sci_pred < 2000 (and Sci_obs > 300 to exclude anomaly)
    for it in range(1, 6):
        df["sci_pred"] = predict_sci_pred(df, fits)
        # Build new mask: Sci_obs > 300 (exclude low-Sci anomaly)
        #                 AND Sci_pred ∈ [300, 2000]
        new_fits = {}
        max_dc = 0
        for box in "ABC":
            mask = ((df["box"] == box)
                    & (df["sci_rate"] > SCI_OBS_LO)
                    & (df["sci_pred"] > SCI_OBS_LO)
                    & (df["sci_pred"] < SCI_PRED_HI))
            new_fits[box] = fit_m7(df[mask])
            dc = max(abs(new_fits[box][i] - fits[box][i]) for i in range(6))
            max_dc = max(max_dc, dc)
        print(f"  Iter {it}: max |Δcoef| = {max_dc:.5f}")
        fits = new_fits
        if max_dc < 0.001:
            print(f"  Converged at iter {it}")
            break

    for box in "ABC":
        c = fits[box]
        N_fit = ((df["box"]==box) & (df["sci_rate"] > SCI_OBS_LO)
                  & (df["sci_pred"] > SCI_OBS_LO) & (df["sci_pred"] < SCI_PRED_HI)).sum()
        print(f"  FINAL Box {box} (N_fit={N_fit:,}): "
              f"b={c[0]:+.1f}, c0={c[1]:.3f}, c1={c[2]:.3f}, "
              f"cN={c[3]:.3f}, β={c[4]:.3f}, γ={c[5]:.3f}")

    # Final Sci_pred
    df["sci_pred"] = predict_sci_pred(df, fits)

    # ============ Scatter plot ============
    fig, axes = plt.subplots(3, 1, figsize=(8, 14), sharex=True)
    xb = np.logspace(np.log10(SCI_OBS_LO), np.log10(4500), 150)
    yb = np.logspace(np.log10(SCI_OBS_LO/2), np.log10(7000), 150)

    rms_main = {}
    for ax, box in zip(axes, "ABC"):
        sub = df[(df["box"]==box) & (df["sci_rate"] >= SCI_OBS_LO)
                  & (df["sci_pred"] > 0)]
        rms_main[box] = np.sqrt(((sub["sci_pred"] - sub["sci_rate"])**2).mean())

        H, xedges, yedges = np.histogram2d(sub["sci_rate"].values,
                                             sub["sci_pred"].values,
                                             bins=[xb, yb])
        ix = np.clip(np.searchsorted(xedges, sub["sci_rate"].values) - 1,
                     0, len(xedges)-2)
        iy = np.clip(np.searchsorted(yedges, sub["sci_pred"].values) - 1,
                     0, len(yedges)-2)
        density = H[ix, iy].astype(float)
        density[density < 1] = 1

        if len(sub) > N_SCATTER:
            idx = np.random.RandomState(0).choice(len(sub), N_SCATTER, replace=False)
        else:
            idx = np.arange(len(sub))
        x_plot = sub["sci_rate"].values[idx]
        y_plot = sub["sci_pred"].values[idx]
        c_plot = density[idx]
        order = np.argsort(c_plot)
        sc = ax.scatter(x_plot[order], y_plot[order], c=c_plot[order],
                         cmap="viridis", norm=LogNorm(vmin=1, vmax=density.max()),
                         s=2, alpha=0.7, rasterized=True, edgecolor="none")

        line = np.array([SCI_OBS_LO, 4500])
        c = fits[box]
        ax.plot(line, line, "--", color="red", lw=1.5,
                label=f"y=x   M7: c0={c[1]:.2f}, c1={c[2]:.2f}, cN={c[3]:.2f}, "
                       f"β={c[4]:.2f}, γ={c[5]:.2f}   RMS={rms_main[box]:.0f}")
        ax.plot(line, 2*line, ":", color="orange", lw=1.2, alpha=0.7, label="y=2x")
        ax.axhline(SCI_PRED_HI, color="gray", ls=":", lw=1, alpha=0.5,
                    label=f"fit cap: Sci_pred={SCI_PRED_HI}")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(SCI_OBS_LO, 4500); ax.set_ylim(SCI_OBS_LO/2, 7000)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted (M7) [cnt/s/det]")
        ax.set_title(f"Box {box}  (N={len(sub):,}, scatter subsampled to {len(idx):,})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, which="both")
        fig.colorbar(sc, ax=ax, label="local density (log)")

    fig.suptitle(f"M7 with iterative Sci_pred < {SCI_PRED_HI} filter "
                 f"(Sci_obs > {SCI_OBS_LO} shown)", fontsize=11, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "sci_pred_M7_iter_scipred2000.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
