#!/usr/bin/env python3
"""Per-detector 3D heatmap: C(det=18, mlat=60, t=108)."""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
BOX_ID = {"a": 0, "b": 1, "c": 2, "A": 0, "B": 1, "C": 2}


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
    aacgm = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))

    n_det = 18
    mlat_edges = np.arange(0, 61, 1); n_mlat = 60
    months = []
    for y in range(2017, 2027):
        for m in range(1, 13):
            if (y == 2017 and m < 6) or (y == 2026 and m > 5): continue
            months.append(f"{y:04d}-{m:02d}")
    n_month = len(months)
    midx = {m: i for i, m in enumerate(months)}
    print(f"Will produce {n_det}×{n_mlat}×{n_month} = {n_det*n_mlat*n_month} bins",
          flush=True)

    sum_C = np.zeros((n_det, n_mlat, n_month), dtype=np.float64)
    n_C   = np.zeros((n_det, n_mlat, n_month), dtype=np.int64)

    print("Streaming all row groups...", flush=True)
    for f in files:
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        for rg in range(n_rg):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            am = np.abs(aacgm(np.column_stack([df["Lat"].values, df["Lon"].values])))
            am = np.where(np.isnan(am), 0.0, am)
            pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
            wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
            lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
            LL = lc*L; lf = 1.0 - dtv/lc
            lv = unwrap_v2(pho, lg, wd, sci, lc, dtv, 150.0)
            base = (pho - lv)*lf/LL - wd/LL
            C_truth = base - sci
            box_arr = df["box"].values
            det_arr = df["det"].astype(int).values
            box_i = np.array([BOX_ID.get(b, -1) for b in box_arr])
            det_id = box_i*6 + det_arr
            months_arr = np.array([d[:7] for d in df["date"].values])
            m_idx = np.array([midx.get(mm, -1) for mm in months_arr])
            mlat_idx = np.digitize(am, mlat_edges) - 1
            ok = (np.isfinite(C_truth) & (np.abs(C_truth) < 800) & (sci > 50)
                  & (det_id >= 0) & (det_id < n_det)
                  & (m_idx >= 0) & (mlat_idx >= 0) & (mlat_idx < n_mlat))
            np.add.at(sum_C, (det_id[ok], mlat_idx[ok], m_idx[ok]), C_truth[ok])
            np.add.at(n_C  , (det_id[ok], mlat_idx[ok], m_idx[ok]), 1)
        print(f"  {os.path.basename(f)}: scanned", flush=True)

    C_mean = np.full((n_det, n_mlat, n_month), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        mask = n_C > 50      # lower threshold since we have 18x finer bins
        C_mean[mask] = sum_C[mask] / n_C[mask]
    print(f"\nValid bins: {mask.sum()} / {C_mean.size}", flush=True)
    print(f"Total rows: {n_C.sum():,}", flush=True)

    np.savez("n_below_study/v5_npz/C_det_mlat_t_heatmap.npz",
             C_mean=C_mean, n=n_C, months=np.array(months), mlat_edges=mlat_edges)
    print("Saved n_below_study/v5_npz/C_det_mlat_t_heatmap.npz")


if __name__ == "__main__":
    main()
