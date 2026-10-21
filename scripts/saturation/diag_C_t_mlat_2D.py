#!/usr/bin/env python3
"""2D heatmap of C(year-month, |mlat|) — see the full structure before fitting.

X-axis: month (2017-06 → 2026-05)
Y-axis: |mlat| 0–60°, 2° bins
Color: median C_truth per (month, mlat-bin)

C_truth per row computed model-independently (unwrap with neutral C=150):
  base = (PHO − Large_unwrap) · lf / L − Wide / L
  C_truth = base − Sci_obs
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
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

    # Finer mlat binning (1°), dense time binning (week) for high resolution.
    mlat_edges = np.arange(0, 61, 1)        # 0, 1, 2, ... 60 → 60 bins of 1° each
    n_mlat = len(mlat_edges) - 1

    months_iter = []
    for y in range(2017, 2027):
        for m in range(1, 13):
            if (y == 2017 and m < 6) or (y == 2026 and m > 5):
                continue
            months_iter.append(f"{y:04d}-{m:02d}")
    n_month = len(months_iter)
    month_to_idx = {m: i for i, m in enumerate(months_iter)}

    print(f"Will produce {n_month} × {n_mlat} = {n_month*n_mlat} bins", flush=True)
    # STREAMING accumulators (constant memory): mean via sum/n
    sum_C  = np.zeros((n_mlat, n_month), dtype=np.float64)
    sum_C2 = np.zeros((n_mlat, n_month), dtype=np.float64)
    n_C    = np.zeros((n_mlat, n_month), dtype=np.int64)

    print("Streaming cache (ALL row groups)...", flush=True)
    for f in files:
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        for rg in range(n_rg):
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
            ok = np.isfinite(C_truth) & (np.abs(C_truth) < 800) & (sci > 50)
            months_arr = np.array([d[:7] for d in df["date"].values])
            m_idx = np.array([month_to_idx.get(mm, -1) for mm in months_arr])
            mlat_idx = np.digitize(am, mlat_edges) - 1
            valid = ok & (m_idx >= 0) & (mlat_idx >= 0) & (mlat_idx < n_mlat)
            # Use add.at for streaming accumulation (handles repeated indices)
            np.add.at(sum_C , (mlat_idx[valid], m_idx[valid]), C_truth[valid])
            np.add.at(sum_C2, (mlat_idx[valid], m_idx[valid]), C_truth[valid]**2)
            np.add.at(n_C   , (mlat_idx[valid], m_idx[valid]), 1)
        print(f"  {os.path.basename(f)}: scanned", flush=True)

    C_med = np.full((n_mlat, n_month), np.nan)     # rename: actually mean now
    C_n   = n_C.copy()
    with np.errstate(invalid='ignore', divide='ignore'):
        mask = n_C > 200
        C_med[mask] = sum_C[mask] / n_C[mask]
    print(f"\nValid bins: {mask.sum()} / {n_month*n_mlat}", flush=True)
    print(f"Total rows accumulated: {n_C.sum():,}", flush=True)

    np.savez("n_below_study/v5_npz/C_2D_heatmap.npz",
             C_med=C_med, C_n=C_n,
             months=np.array(months_iter), mlat_edges=mlat_edges)

    month_dt = np.array([np.datetime64(m + "-15") for m in months_iter])
    mlat_centers = 0.5 * (mlat_edges[:-1] + mlat_edges[1:])

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle(
        f"C(t, |mlat|) — model-independent (unwrap with C=150) — {n_C.sum()/1e6:.1f}M rows\n"
        f"1° × month bins (≥200 rows/bin), color = mean C (cnt/s)",
        fontsize=12, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1])/2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    pcm = ax.pcolormesh(x_edges, mlat_edges, C_med,
                        cmap='viridis', vmin=0, vmax=400, shading='flat')
    ax.set_ylabel("|mlat| (deg)", fontsize=12)
    ax.set_xlabel("date", fontsize=12)
    cb = fig.colorbar(pcm, ax=ax, pad=0.01)
    cb.set_label("mean C (cnt/s)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(20, color='red', ls='--', lw=1, alpha=0.7, label='|mlat|=20° v5t threshold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_C_t_mlat_2D.png"
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
