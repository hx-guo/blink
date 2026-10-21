#!/usr/bin/env python3
"""v5 streaming aggregator — yearly clean_relaxed parquet variant.

Reads one yearly clean_relaxed_YEAR.parquet (~400M rows, 6-14 GB on disk),
streams batches via pyarrow.iter_batches, accumulates per-date buffers,
processes each calendar date through the v5 pipeline, and emits one .npz
with combined histograms + per-day s_det estimates.

Differs from v5_aggregator_worker.py only in input shape: that one was
per-day per_sec_parquet files; this one is per-year clean cache files.

Usage:
    python3 v5_aggregator_yearly.py \
        --input n_below_study/clean_relaxed_2020.parquet \
        --aacgm-grid aacgm_grid_2020.npz \
        --output v5_agg_2020.npz \
        [--k 0.00188] [--batch-size 1000000]
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator

L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0

LO, HI = 30.0, 10_000.0
Y_LO, Y_HI = -500.0, 1500.0
N_BINS = 150
XE = np.logspace(np.log10(LO), np.log10(HI), N_BINS + 1)
YE_LIN = np.linspace(Y_LO, Y_HI, N_BINS + 1)
YE_LOG = np.logspace(np.log10(LO), np.log10(HI), N_BINS + 1)

NEEDED_COLS = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s", "L_cycles", "Dt", "Lat", "Lon"]


def unwrap_large_v2(pho, large, wide, sci, l_cycles, dt, C):
    pho = np.asarray(pho, dtype=np.float64)
    large = np.asarray(large, dtype=np.float64)
    wide = np.asarray(wide, dtype=np.float64)
    sci = np.asarray(sci, dtype=np.float64)
    L = np.asarray(l_cycles, dtype=np.float64) * L_CYCLES_TO_SEC
    lf = 1.0 - np.asarray(dt, dtype=np.float64) / np.asarray(l_cycles, dtype=np.float64)
    predicted = pho - (wide + (sci + C) * L) / lf
    n_raw = np.round((predicted - large) / 1024.0).astype(int)
    n_raw = np.maximum(n_raw, 0)
    max_allowed = pho - wide
    large_corr = large + n_raw * 1024.0
    over = large_corr > max_allowed
    if over.any():
        n_max = np.maximum(np.floor((max_allowed - large) / 1024.0).astype(int), 0)
        n = np.where(over, n_max, n_raw)
        large_corr = large + n * 1024.0
    return large_corr


def process_day(df, interp, k_global, calib=None):
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)
    abs_mlat_safe = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)
    mlat_term = np.maximum(0.0, abs_mlat_safe - B_THRESHOLD) ** 2

    pho = df["PHO"].astype("float64").values
    large_raw = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc
    box_arr = df["box"].values
    det_arr = df["det"].values

    if calib is not None:
        # FIXED closed-form formula, zero refit:
        #   C = s0_det * g(t) * [1 + k(t)*mlat_term] + C0
        ty = (np.datetime64(str(df["date"].iloc[0])) - calib["t0"]).astype("timedelta64[D]").astype(float) / 365.25
        g = 1.0 - calib["beta"] * ty
        w = calib["w"]; kc = calib["k_coeffs"]
        k_t = (kc[0] + kc[1]*np.cos(w*ty) + kc[2]*np.sin(w*ty)
               + kc[3]*np.cos(2*w*ty) + kc[4]*np.sin(2*w*ty))
        box_idx = np.select([box_arr == b for b in "ABC"], [0, 1, 2], default=0)
        detid = box_idx * 6 + det_arr
        s_det_per_row = calib["s0_det"][detid] * g
        C_per_row = s_det_per_row * (1.0 + k_t * mlat_term) + calib["C0"]
        s_det_map = (calib["s0_det"] * g).reshape(3, 6)
        n_eq_map = np.zeros((3, 6), dtype=np.int64)
        is_clean_base = (sci > 100) & ~np.isnan(mlat) & np.isfinite(C_per_row)
    else:
        large_v2 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, 150.0)
        base_v2 = (pho - large_v2) * lf / L - wide / L
        resid_v2 = base_v2 - sci
        is_clean_base = (
            (wide / np.maximum(pho, 1) < 0.3)
            & (sci > 100)
            & np.isfinite(resid_v2)
            & ~np.isnan(mlat)
            & (np.abs(resid_v2) < 2000)
        )
        is_eq = is_clean_base & (abs_mlat < 5)
        s_det_map = np.full((3, 6), np.nan, dtype=np.float64)
        n_eq_map = np.zeros((3, 6), dtype=np.int64)
        for bi, box in enumerate("ABC"):
            for det in range(6):
                m = is_eq & (box_arr == box) & (det_arr == det)
                n = int(m.sum())
                n_eq_map[bi, det] = n
                if n > 100:
                    s_det_map[bi, det] = float(np.mean(resid_v2[m]))
        pop_median = np.nanmedian(s_det_map)
        if not np.isfinite(pop_median):
            pop_median = 120.0
        s_det_filled = np.where(np.isnan(s_det_map), pop_median, s_det_map)
        s_det_per_row = np.zeros(len(df))
        for bi, box in enumerate("ABC"):
            for det in range(6):
                m = (box_arr == box) & (det_arr == det)
                s_det_per_row[m] = s_det_filled[bi, det]
        C_per_row = s_det_per_row * (1.0 + k_global * mlat_term)

    large_v3 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C_per_row)

    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n_wraps_v3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    n_max = np.maximum(np.floor((max_large_event - large_raw) / 1024.0).astype(int), 0)
    n_wraps_v5 = np.where(n_wraps_v3 > n_max, n_max, n_wraps_v3)
    large_v5 = large_raw + n_wraps_v5 * 1024.0
    n_capped = int((n_wraps_v3 > n_max).sum())

    base_v5 = (pho - large_v5) * lf / L - wide / L
    resid_v5 = base_v5 - sci
    resid_clean = resid_v5 - C_per_row

    is_valid = (
        np.isfinite(base_v5) & np.isfinite(resid_v5) & (sci > 0) & (base_v5 > 0)
    )

    H_loglog, _, _ = np.histogram2d(sci[is_valid], base_v5[is_valid], bins=[XE, YE_LOG])
    H_before, _, _ = np.histogram2d(sci[is_valid], resid_v5[is_valid], bins=[XE, YE_LIN])
    H_after, _, _ = np.histogram2d(sci[is_valid], resid_clean[is_valid], bins=[XE, YE_LIN])

    is_blob = is_valid & (sci >= 800) & (sci <= 2500) & (resid_clean >= -300) & (resid_clean <= -50)
    is_main = is_valid & (sci >= 800) & (sci <= 2500) & (resid_clean >= -50) & (resid_clean <= 100)
    below_yx = (base_v5 < sci) & is_valid

    return {
        "s_det_map": s_det_map,
        "n_eq_map": n_eq_map,
        "H_loglog": H_loglog.astype(np.int64),
        "H_before": H_before.astype(np.int64),
        "H_after": H_after.astype(np.int64),
        "n_total": int(len(df)),
        "n_clean": int(is_clean_base.sum()),
        "n_valid": int(is_valid.sum()),
        "n_capped": n_capped,
        "n_blob": int(is_blob.sum()),
        "n_main": int(is_main.sum()),
        "n_below_yx": int(below_yx.sum()),
    }


def stream_by_date(input_path, batch_size):
    """Yield (date_str, df_for_that_date) by streaming the file.

    Relies on rows being grouped by date in the file (clean_relaxed cache is
    daily-aggregated, so dates appear in contiguous blocks).
    """
    pf = pq.ParquetFile(input_path)
    print(f"[worker] parquet metadata: {pf.metadata.num_rows:,} rows, "
          f"{pf.metadata.num_row_groups} row groups", flush=True)

    current_date = None
    buffer = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=NEEDED_COLS):
        bdf = batch.to_pandas()
        # find date transitions within this batch
        dates = bdf["date"].values
        if len(dates) == 0:
            continue
        # split positions where date changes
        change_idx = np.where(np.array(dates[1:]) != np.array(dates[:-1]))[0] + 1
        boundaries = [0] + change_idx.tolist() + [len(bdf)]
        for i in range(len(boundaries) - 1):
            chunk = bdf.iloc[boundaries[i]:boundaries[i + 1]]
            chunk_date = chunk["date"].iloc[0]
            if current_date is None:
                current_date = chunk_date
            if chunk_date == current_date:
                buffer.append(chunk)
            else:
                yield current_date, pd.concat(buffer, ignore_index=True)
                buffer = [chunk]
                current_date = chunk_date

    if buffer:
        yield current_date, pd.concat(buffer, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="One clean_relaxed_YEAR.parquet")
    p.add_argument("--aacgm-grid", required=True, help="Path to aacgm grid npz")
    p.add_argument("--output", required=True, help="Output .npz path")
    p.add_argument("--k", type=float, default=0.00188)
    p.add_argument("--calib", default=None, help="v5t_calib.npz for fixed closed-form mode (zero refit)")
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument("--limit-days", type=int, default=0, help="0 = all")
    args = p.parse_args()

    print(f"[worker] input={args.input}  output={args.output}", flush=True)

    calib = None
    if args.calib:
        cz = np.load(args.calib)
        calib = {"s0_det": cz["s0_det"], "beta": float(cz["beta"]),
                 "w": float(cz["w"]), "k_coeffs": cz["k_coeffs"], "C0": float(cz["C0"]),
                 "t0": np.datetime64(str(cz["t0"]))}
        print(f"[worker] FIXED-formula mode: calib={args.calib}, C0={calib['C0']:+.1f}, "
              f"g(t)=1-{calib['beta']:.4f}t (linear)", flush=True)

    t_init = time.time()
    print(f"[worker] loading aacgm grid {args.aacgm_grid}...", flush=True)
    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]),
        grid["mlat"],
        bounds_error=False,
        fill_value=np.nan,
    )
    print(f"[worker] aacgm ready in {time.time()-t_init:.1f}s", flush=True)

    H_loglog = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    H_before = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    H_after = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    dates = []
    s_det_daily = []
    n_eq_daily = []
    stats = {"n_total": 0, "n_clean": 0, "n_valid": 0, "n_capped": 0,
             "n_blob": 0, "n_main": 0, "n_below_yx": 0}

    t0 = time.time()
    for i, (date, day_df) in enumerate(stream_by_date(args.input, args.batch_size)):
        if args.limit_days and i >= args.limit_days:
            break
        t_d = time.time()
        try:
            r = process_day(day_df, interp, args.k, calib)
        except Exception as e:
            print(f"[worker] FAIL date={date}: {type(e).__name__}: {e}", flush=True)
            continue
        H_loglog += r["H_loglog"]
        H_before += r["H_before"]
        H_after += r["H_after"]
        dates.append(str(date))
        s_det_daily.append(r["s_det_map"])
        n_eq_daily.append(r["n_eq_map"])
        for k in stats:
            stats[k] += r[k]
        if (i + 1) % 30 == 0 or i < 3:
            dt = time.time() - t0
            print(f"[worker] day {i+1} date={date} rows={len(day_df):,} "
                  f"proc={time.time()-t_d:.1f}s tot={dt:.1f}s", flush=True)

    np.savez_compressed(
        args.output,
        H_loglog=H_loglog, H_before=H_before, H_after=H_after,
        XE=XE, YE_LIN=YE_LIN, YE_LOG=YE_LOG,
        dates=np.array(dates),
        s_det_daily=np.array(s_det_daily),
        n_eq_daily=np.array(n_eq_daily),
        k_global=args.k,
        n_total=stats["n_total"], n_clean=stats["n_clean"],
        n_valid=stats["n_valid"], n_capped=stats["n_capped"],
        n_blob=stats["n_blob"], n_main=stats["n_main"],
        n_below_yx=stats["n_below_yx"],
    )
    print(f"[worker] DONE in {time.time()-t0:.1f}s -> {args.output}", flush=True)
    print(f"[worker] {len(dates)} days, total={stats['n_total']:,} valid={stats['n_valid']:,} "
          f"below_yx={stats['n_below_yx']:,} blob/main={stats['n_blob']}/{stats['n_main']}", flush=True)


if __name__ == "__main__":
    main()
