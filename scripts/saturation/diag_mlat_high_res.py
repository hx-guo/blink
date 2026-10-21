#!/usr/bin/env python3
"""High-resolution mlat marginal in 0-30°.

Extract C_truth per row from cache (sample one row group per yearly parquet),
bin into 0.2° mlat bins (150 bins over 0-30°), and look at the 14-18° region
in detail — is it a true step or smooth transition?
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
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

    # 0.2° bins over 0-30°
    mlat_edges = np.arange(0, 30 + 0.001, 0.2)
    n_mlat = len(mlat_edges) - 1
    print(f"Will produce {n_mlat} bins of 0.2° width over 0-30°")

    sum_C  = np.zeros(n_mlat, dtype=np.float64)
    n_C    = np.zeros(n_mlat, dtype=np.int64)

    print("Scanning cache (one row group per file)...")
    for f in files:
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        # Read the middle row group for a representative sample
        rg_pick = n_rg // 2
        df = pf.read_row_group(int(rg_pick), columns=NEEDED).to_pandas()
        am = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
        am = np.where(np.isnan(am), 0.0, am)
        pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
        wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
        lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
        LL = lc*L; lf = 1.0 - dtv/lc
        lv = unwrap_v2(pho, lg, wd, sci, lc, dtv, 150.0)
        base = (pho - lv)*lf/LL - wd/LL
        C_truth = base - sci
        ok = np.isfinite(C_truth) & (np.abs(C_truth) < 800) & (sci > 50)
        mlat_idx = np.digitize(am, mlat_edges) - 1
        valid = ok & (mlat_idx >= 0) & (mlat_idx < n_mlat)
        np.add.at(sum_C, mlat_idx[valid], C_truth[valid])
        np.add.at(n_C  , mlat_idx[valid], 1)
        print(f"  {os.path.basename(f)} rg{rg_pick}: {valid.sum():,} rows kept")

    valid = n_C > 500   # require enough samples per 0.2° bin
    C_mean = np.full(n_mlat, np.nan)
    C_mean[valid] = sum_C[valid] / n_C[valid]
    mlat_centers = 0.5 * (mlat_edges[:-1] + mlat_edges[1:])
    print(f"\nValid bins: {valid.sum()}/{n_mlat}")
    print(f"Total rows accumulated: {n_C.sum():,}")

    np.savez("n_below_study/v5_npz/C_mlat_highres.npz",
             C_mean=C_mean, n=n_C,
             mlat_edges=mlat_edges, mlat_centers=mlat_centers)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"High-resolution marginal C(|mlat|) — 0.2° bins, time-averaged "
                 f"({n_C.sum()/1e6:.0f}M rows)",
                 fontsize=13, fontweight='bold')

    # Panel 1: full 0-30°
    ax = axes[0]
    ax.plot(mlat_centers[valid], C_mean[valid], 'o-', ms=4, lw=1, color='black')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("Full 0-30° (0.2° bins)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: zoom 12-20° to see the kink in detail
    ax = axes[1]
    zm = (mlat_centers >= 12) & (mlat_centers <= 20)
    ax.plot(mlat_centers[zm & valid], C_mean[zm & valid],
            'o-', ms=8, lw=1.5, color='black', label='data (0.2° bins)')
    # Annotate each point
    for i, m in enumerate(mlat_centers):
        if 13 <= m <= 19 and valid[i] and (i % 3 == 0):
            ax.annotate(f"{C_mean[i]:.1f}",
                        (m, C_mean[i]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=8, color='C3')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.axvline(14.5, ls=':', color='C0', alpha=0.7, lw=1.5, label='fit μ_2=14.5° (sigmoid)')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("Zoomed 12-20°: step or smooth?", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_mlat_highres.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # Print bin-level values
    print("\n=== Bin-level data (12-20°, 0.2° bins) ===")
    for i, m in enumerate(mlat_centers):
        if 12 <= m <= 20 and valid[i]:
            print(f"  |mlat|={m:5.2f}°  N={n_C[i]:>9,}  ⟨C⟩={C_mean[i]:7.2f}")


if __name__ == "__main__":
    main()
