#!/usr/bin/env python3
"""Plot the true time variation of C(t), split by low-mlat (baseline) and
high-mlat (CR-dominated), to see if g(t) and k(t) really have different shapes
or if they can be merged into a single F(t).

C_truth per row is computed model-independently:
  base = (PHO - Large_unwrap) · lf / L - Wide / L
  C_truth = base - Sci_obs
(unwrap_v2 uses neutral C=150; result is insensitive to that choice because
the 1024-grid wrap distance is much larger than ±C variation.)

Then per (year-month, mlat-bin):
  C_baseline(month) = median over rows with |mlat| < 5°
  C_highmlat(month) = median over rows with 35° ≤ |mlat| < 50°

We plot both. If they have the same shape after rescaling, F(t) can be one
function; if they don't, the model needs two independent time factors.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    pho=np.asarray(pho,float); large=np.asarray(large,float); wide=np.asarray(wide,float)
    sci=np.asarray(sci,float); LL=np.asarray(lc,float)*L
    lf=1.0-np.asarray(dt,float)/np.asarray(lc,float)
    pred=pho-(wide+(sci+C)*LL)/lf
    n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0)
        out=large+np.where(ov,nm,n)*1024.
    return out


def main():
    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))

    # Collect per row: month, mlat, C_truth
    print("Streaming cache to compute per-row C_truth...", flush=True)
    months_low  = []; Cs_low  = []
    months_high = []; Cs_high = []
    for f in files:
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        for rg in np.unique(np.linspace(0, n_rg-1, 6).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            am = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
            am = np.where(np.isnan(am), 0.0, am)
            pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
            wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
            lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
            LL = lc*L; lf = 1.0 - dtv/lc
            lv = unwrap_v2(pho, lg, wd, sci, lc, dtv, 150.0)
            base = (pho - lv)*lf/LL - wd/LL
            C_truth = base - sci
            # Sanity: keep finite & bounded
            ok = np.isfinite(C_truth) & (np.abs(C_truth) < 500) & (sci > 50)
            months_arr = np.array([d[:7] for d in df["date"].values])
            mask_low  = ok & (am < 5)
            mask_high = ok & (am >= 35) & (am < 50)
            months_low.extend(months_arr[mask_low].tolist());  Cs_low.extend(C_truth[mask_low].tolist())
            months_high.extend(months_arr[mask_high].tolist()); Cs_high.extend(C_truth[mask_high].tolist())
        print(f"  {os.path.basename(f)}: scanned", flush=True)

    import pandas as pd
    df_low  = pd.DataFrame({"month": months_low,  "C": Cs_low})
    df_high = pd.DataFrame({"month": months_high, "C": Cs_high})
    print(f"\nlow-mlat (<5°)  rows: {len(df_low):,}")
    print(f"high-mlat (35-50°) rows: {len(df_high):,}")

    monthly_low  = df_low.groupby("month")["C"].agg(["median", "count", "std"]).reset_index()
    monthly_high = df_high.groupby("month")["C"].agg(["median", "count", "std"]).reset_index()
    monthly_low["t"]  = pd.to_datetime(monthly_low["month"] + "-15")
    monthly_high["t"] = pd.to_datetime(monthly_high["month"] + "-15")
    # Filter months with too few samples
    monthly_low  = monthly_low [monthly_low ["count"] > 500].reset_index(drop=True)
    monthly_high = monthly_high[monthly_high["count"] > 500].reset_index(drop=True)
    print(f"\nKept low-mlat months: {len(monthly_low)},  high-mlat months: {len(monthly_high)}")

    # Merge for ratio (only months that exist in both)
    merged = pd.merge(monthly_low, monthly_high, on="month",
                      suffixes=("_low", "_high"))
    merged["t"] = pd.to_datetime(merged["month"] + "-15")
    merged["ratio"] = merged["median_high"] / merged["median_low"]
    print(f"Merged (both bins available): {len(merged)} months")

    # ─── Plot ───
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Time variation of C — model-independent (using unwrap with C=150)\n"
                 "If C_baseline and C_highmlat have the SAME shape, F(t) can be a single function",
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.errorbar(monthly_low["t"], monthly_low["median"],
                yerr=monthly_low["std"]/np.sqrt(monthly_low["count"]),
                fmt='o-', markersize=4, lw=1, color='steelblue',
                label="C(|mlat|<5°) — baseline, monthly median ± std/√N")
    ax.set_ylabel(r"C$_{baseline}$ (cnt/s)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_title("LOW mlat — pure baseline (dark count + electronics)", fontsize=11)

    ax = axes[1]
    ax.errorbar(monthly_high["t"], monthly_high["median"],
                yerr=monthly_high["std"]/np.sqrt(monthly_high["count"]),
                fmt='o-', markersize=4, lw=1, color='darkorange',
                label="C(35°≤|mlat|<50°) — CR-dominated, monthly median ± std/√N")
    ax.set_ylabel(r"C$_{highmlat}$ (cnt/s)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_title("HIGH mlat — CR-dominated (≈ baseline + CR secondary)", fontsize=11)

    ax = axes[2]
    ax.errorbar(merged["t"], merged["ratio"],
                yerr=(merged["std_high"]/np.sqrt(merged["count_high"])) / merged["median_low"],
                fmt='o-', markersize=4, lw=1, color='purple',
                label=r"C(high)/C(baseline) per month")
    ax.set_ylabel("ratio", fontsize=11)
    ax.set_xlabel("date", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_title("RATIO — strips out baseline trend, leaves CR-modulation shape", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_C_time_shape.png"
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # Print numeric values for inspection
    print(f"\n=== Monthly C_baseline (low mlat) ===")
    print(monthly_low[["month", "median", "count"]].to_string(index=False))
    print(f"\n=== Monthly C_highmlat (high mlat) ===")
    print(monthly_high[["month", "median", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
