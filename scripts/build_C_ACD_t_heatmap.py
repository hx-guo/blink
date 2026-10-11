#!/usr/bin/env python3
"""Build the C(ACD, t) 2D heatmap by streaming all cache rows.

Replaces the mlat axis with ACD lookup value (in-situ radiation environment).
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
ACD_PATH = "/Users/skyair/Developer/ihep/astro_sift/astro_sift/satellites/hxmt/acd.txt"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]


def load_acd_lookup():
    a = np.loadtxt(ACD_PATH)
    lat_grid = a[180:360][:, 0]
    lon_grid_orig = a[0:180][0, :]
    acd = a[360:540]
    lon_grid_360 = (lon_grid_orig + 360) % 360
    sort_idx = np.argsort(lon_grid_360)
    return RegularGridInterpolator(
        (lat_grid, lon_grid_360[sort_idx]), acd[:, sort_idx],
        bounds_error=False, fill_value=np.nan)


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
    acd_lookup = load_acd_lookup()
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))

    # 50 ACD bins, log-spaced over 1..2000 cnt/s
    acd_edges = np.logspace(0, np.log10(2000), 51)
    n_acd = len(acd_edges) - 1
    months_iter = []
    for y in range(2017, 2027):
        for m in range(1, 13):
            if (y == 2017 and m < 6) or (y == 2026 and m > 5):
                continue
            months_iter.append(f"{y:04d}-{m:02d}")
    n_month = len(months_iter)
    month_to_idx = {m: i for i, m in enumerate(months_iter)}
    print(f"Will produce {n_acd} ACD bins × {n_month} months = {n_acd*n_month} bins",
          flush=True)

    sum_C  = np.zeros((n_acd, n_month), dtype=np.float64)
    sum_C2 = np.zeros((n_acd, n_month), dtype=np.float64)
    n_C    = np.zeros((n_acd, n_month), dtype=np.int64)

    print("Streaming all row groups...", flush=True)
    for f in files:
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        for rg in range(n_rg):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            acd_v = acd_lookup(np.column_stack([df["Lat"].values, df["Lon"].values]))
            pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
            wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
            lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
            LL = lc*L; lf = 1.0 - dtv/lc
            lv = unwrap_v2(pho, lg, wd, sci, lc, dtv, 150.0)
            base = (pho - lv)*lf/LL - wd/LL
            C_truth = base - sci
            ok = np.isfinite(C_truth) & (np.abs(C_truth) < 800) & (sci > 50) & np.isfinite(acd_v)
            months_arr = np.array([d[:7] for d in df["date"].values])
            m_idx = np.array([month_to_idx.get(mm, -1) for mm in months_arr])
            acd_idx = np.digitize(acd_v, acd_edges) - 1
            valid = ok & (m_idx >= 0) & (acd_idx >= 0) & (acd_idx < n_acd)
            np.add.at(sum_C , (acd_idx[valid], m_idx[valid]), C_truth[valid])
            np.add.at(sum_C2, (acd_idx[valid], m_idx[valid]), C_truth[valid]**2)
            np.add.at(n_C   , (acd_idx[valid], m_idx[valid]), 1)
        print(f"  {os.path.basename(f)}: scanned", flush=True)

    C_mean = np.full((n_acd, n_month), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        mask = n_C > 200
        C_mean[mask] = sum_C[mask] / n_C[mask]
    print(f"\nValid bins: {mask.sum()} / {n_acd*n_month}", flush=True)
    print(f"Total rows: {n_C.sum():,}", flush=True)

    np.savez("n_below_study/v5_npz/C_ACD_t_heatmap.npz",
             C_mean=C_mean, n=n_C,
             acd_edges=acd_edges,
             months=np.array(months_iter))
    print("Saved n_below_study/v5_npz/C_ACD_t_heatmap.npz")


if __name__ == "__main__":
    main()
