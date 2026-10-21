#!/usr/bin/env python3
"""Cross-year validation of PHO model parameters.

For each of 9-10 yearly caches (2017-2026 on server), compute:
  1. per-(box, det) C_det at |mlat|<5° (equatorial baseline)
  2. B(|mlat|) at selected mlat bins (10°, 20°, 30°, 40°)
  3. Sample count + filter survival fractions

Per-year output: a small JSON summary file.
Final plot: time evolution of C_det per det + B(|mlat|) per bin across years.

This script runs SERVER-SIDE per year to avoid downloading 100+ GB.

Usage on server:
  cd /scratchfs/gecam/guohx/blink
  for yr in 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
      python3 -u scripts/saturation/diag_cross_year_validation.py \\
          --cache n_below_study/clean_relaxed_${yr}.parquet \\
          --out n_below_study/year_summary_${yr}.json
  done

Then locally:
  python3 scripts/saturation/plot_cross_year.py n_below_study/year_summary_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

L_CYCLES_TO_SEC = 16e-6

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]

# Pre-compute AACGM grid once (independent of cache year)
# Use mid-2020 as reference; mlat differences across 2017-2026 are <1° (~10° per decade for magnetic pole drift)
AACGM_DATE = datetime.datetime(2020, 6, 15)


def build_aacgm_grid():
    """Load precomputed AACGM grid (no aacgmv2 dependency on compute nodes)."""
    grid_path = Path("n_below_study/aacgm_grid_2020.npz")
    if not grid_path.exists():
        raise FileNotFoundError(f"AACGM grid not found at {grid_path}. Run locally and scp first.")
    data = np.load(grid_path)
    return data["lat_grid"], data["lon_grid"], data["mlat"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-frac", type=float, default=0.10,
                    help="Subsample fraction to reduce memory (default 0.10)")
    args = p.parse_args()

    cache_path = Path(args.cache)
    out_path = Path(args.out)
    year = cache_path.stem.split("_")[-1]
    print(f"=== Year {year} ===")
    print(f"  Loading {cache_path}...")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(cache_path)
    total_rows = pf.metadata.num_rows
    print(f"  Total rows: {total_rows:,}, subsample fraction = {args.sample_frac}")

    # Streaming read with subsampling to keep memory bounded
    rng = np.random.RandomState(int(year))
    chunks = []
    rows_kept = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=USE_COLS):
        n = batch.num_rows
        mask = rng.random(n) < args.sample_frac
        sub = batch.filter(__import__("pyarrow").array(mask))
        chunks.append(sub.to_pandas())
        rows_kept += sub.num_rows
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Sampled rows: {rows_kept:,}")

    # Build AACGM grid + interp
    print("  Building AACGM grid + interpolating mlat...")
    lat_grid, lon_grid, mlat_grid = build_aacgm_grid()
    interp = RegularGridInterpolator((lat_grid, lon_grid), mlat_grid,
                                      bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)

    # Unwrap + filter
    pho = df["PHO"].values; large_raw = df["Large"].values
    wide = df["Wide"].values; sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values; dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    large_corr, conf = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0,
                                        return_confidence=True)
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean = ((conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100)
                & (n_wraps == 0) & np.isfinite(residual) & ~np.isnan(mlat))

    # 1. per-(box, det) C_det at |mlat|<5°
    is_eq = is_clean & (abs_mlat < 5)
    C_det = {}
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_eq
            if m.sum() < 200:
                C_det[f"{box}{det}"] = None
            else:
                C_det[f"{box}{det}"] = {
                    "C": float(np.mean(residual[m])),
                    "C_std": float(np.std(residual[m]) / np.sqrt(m.sum())),
                    "N": int(m.sum()),
                }

    # 2. B(|mlat|) at bins, subtracting per-det C_det
    # First compute C_det_per_row
    C_per_row = np.zeros(len(df))
    for box in "ABC":
        for det in range(6):
            key = f"{box}{det}"
            if C_det[key] is None:
                continue
            m = ((df["box"] == box) & (df["det"] == det)).values
            C_per_row[m] = C_det[key]["C"]
    B_per_row = residual - C_per_row

    mlat_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45)]
    B_lat = {}
    for lo, hi in mlat_bins:
        m = is_clean & (abs_mlat >= lo) & (abs_mlat < hi)
        if m.sum() < 1000:
            B_lat[f"{lo}-{hi}"] = None
        else:
            B_lat[f"{lo}-{hi}"] = {
                "B": float(np.mean(B_per_row[m])),
                "B_std": float(np.std(B_per_row[m]) / np.sqrt(m.sum())),
                "N": int(m.sum()),
            }

    # 3. summary stats
    summary = {
        "year": year,
        "cache_path": str(cache_path),
        "total_rows": int(total_rows),
        "sampled_rows": int(rows_kept),
        "clean_rows": int(is_clean.sum()),
        "n_wraps_distribution": {
            f"n={k}": int((n_wraps == k).sum()) for k in sorted(set(n_wraps)) if k >= 0 and (n_wraps == k).sum() > 0
        },
        "low_conf_rows": int((conf == CONF_LOW).sum()),
        "C_det_per_det": C_det,
        "B_per_mlat_bin": B_lat,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
