#!/usr/bin/env python3
"""Sci_pred vs Sci_obs density plot with PER-MODE coefficients (clean band).

Each second is classified by its DATE's median Large/Sci in Sci 1000-1500 bin:
  HIGH-Large mode dates: apply M1_HIGH_NEW coefficients
  LOW-Large mode dates:  apply M1_LOW_NEW coefficients

Both sets of coefficients fit on CLEAN band (Sci ∈ [400, 1000] per det,
group_rate < 6000) to avoid FIFO contamination.
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
    df["large_frac"]  = df["large_rate"] / df["sci_rate"].clip(lower=1)
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
    return coef


def main():
    df = load()

    # Classify date modes
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
    print(f"\nMode counts: HIGH={(df['mode']=='HIGH').sum():,}, "
          f"LOW={(df['mode']=='LOW').sum():,}, MID={(df['mode']=='MID').sum():,}")

    # Fit per-Box per-Mode on CLEAN band
    print(f"\n=== M1 fit per (Box, Mode) on CLEAN band ===")
    print(f"  {'Mode':>5s} {'Box':>3s} {'N':>10s}  {'b':>8s} {'1+α':>8s} {'β':>7s} {'γ':>7s}")
    fits = {}
    for mode in ["HIGH","LOW","MID"]:
        for box in "ABC":
            mask = ((df["box"] == box) & (df["mode"] == mode)
                    & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
                    & (df["group_rate"] < BOX_RATE_CAP))
            sub = df[mask]
            if len(sub) < 1000:
                # Fall back to ALL mode for missing categories
                fits[(mode, box)] = fits.get(("HIGH", box), fits.get(("LOW", box)))
                continue
            coef = fit_m1(sub)
            fits[(mode, box)] = coef
            print(f"  {mode:>5s} {box:>3s} {len(sub):>10,d}  "
                  f"{coef[0]:>+8.1f} {coef[1]:>+8.3f} {coef[2]:>+7.3f} {coef[3]:>+7.3f}")

    # Apply per-mode per-box coefficients
    df["mode_box"] = list(zip(df["mode"], df["box"].astype(str)))
    df["b_v"] = df["mode_box"].map(lambda k: fits[k][0])
    df["c1plus_v"] = df["mode_box"].map(lambda k: fits[k][1])
    df["beta_v"] = df["mode_box"].map(lambda k: fits[k][2])
    df["gamma_v"] = df["mode_box"].map(lambda k: fits[k][3])
    df["sci_pred"] = ((df["pho_rate"] - df["beta_v"]*df["wide_rate"]
                       - df["gamma_v"]*df["large_rate"] - df["b_v"])
                       / df["c1plus_v"])

    # Compute RMS
    rms_main = {}
    for box in "ABC":
        mask = (df["box"]==box) & (df["sci_rate"] > 300)
        sub = df[mask]
        rms_main[box] = np.sqrt(((sub["sci_pred"] - sub["sci_rate"])**2).mean())

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 14), sharex=True)
    xy_bins = np.logspace(np.log10(40), np.log10(4500), 200)

    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"]==box]
        N = len(sub)

        H, xedges, yedges = np.histogram2d(
            sub["sci_rate"].values, sub["sci_pred"].values.clip(0.5, 1e5),
            bins=[xy_bins, xy_bins])
        X, Y = np.meshgrid(xedges, yedges)
        pcm = ax.pcolormesh(X, Y, H.T, norm=LogNorm(vmin=1, vmax=H.max()),
                             cmap="viridis", shading="auto")

        line = np.array([50, 4500])
        ax.plot(line, line, "--", color="red", lw=1.5, label="y=x")

        # Show coefficients for each mode in legend (HIGH and LOW)
        c_h = fits[("HIGH", box)]; c_l = fits[("LOW", box)]
        label_text = (f"HIGH: β={c_h[2]:.2f}, γ={c_h[3]:.2f}, α={c_h[1]-1:+.2f}, b={c_h[0]:+.0f}\n"
                       f"LOW:  β={c_l[2]:.2f}, γ={c_l[3]:.2f}, α={c_l[1]-1:+.2f}, b={c_l[0]:+.0f}")
        ax.plot([], [], " ", label=label_text)

        # Mode-aware binned median
        for mode, color in [("HIGH","cyan"), ("LOW","orange")]:
            sub_m = sub[sub["mode"]==mode]
            bins_med = np.logspace(np.log10(300), np.log10(4500), 30)
            bc = 0.5 * (bins_med[:-1] + bins_med[1:])
            med = np.array([
                sub_m.loc[(sub_m["sci_rate"] >= bins_med[i]) & (sub_m["sci_rate"] < bins_med[i+1]),
                          "sci_pred"].median() if ((sub_m["sci_rate"] >= bins_med[i])
                                                    & (sub_m["sci_rate"] < bins_med[i+1])).sum() > 100
                        else np.nan
                for i in range(len(bins_med)-1)
            ])
            ax.plot(bc, med, "-", color=color, lw=2, label=f"{mode} median")

        ax.axvline(SCI_LO_CLEAN, color="purple", ls=":", lw=1, alpha=0.5)
        ax.axvline(SCI_HI_CLEAN, color="purple", ls=":", lw=1, alpha=0.5)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(40, 4500); ax.set_ylim(40, 4500)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.set_title(f"Box {box}  (N={N:,})  RMS_main = {rms_main[box]:.0f}")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.grid(alpha=0.3, which="both")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label="per-det-sec bin count (log scale)")

    fig.suptitle(f"M1 per-mode fit (CLEAN band Sci ∈ [{SCI_LO_CLEAN:.0f}, {SCI_HI_CLEAN:.0f}]/det, "
                 f"group_rate < {BOX_RATE_CAP:.0f}); coefficients applied per (date-mode, box)",
                 fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.96, bottom=0.04, hspace=0.18)
    out = OUT_DIR / "sci_pred_vs_obs_mode_aware.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
