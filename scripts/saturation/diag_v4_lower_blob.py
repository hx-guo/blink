#!/usr/bin/env python3
"""Diagnose the secondary blob below main cloud in v4 panel 3.

Location: Sci_obs ~ 1000-2000, residual_clean ~ -250 to -50.
After full model (C_det + B), main cloud sits at 0 but a separate blob sits at -150.

What is it? Suspects:
  (a) one (box, det) with C_det overestimated → consistent negative offset
  (b) one |mlat| range where B overestimates
  (c) magnetar-mode rows (Wide/PHO > 0.3) leaking through
  (d) dt model bias at high rates
  (e) high-rate PHO under-count
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
B_COEF = 0.26
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0


def main():
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df):,} rows")

    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                      bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)
    abs_mlat_safe = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)

    pho = df["PHO"].astype("float64").values
    large_raw = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc

    # v2 → C_det
    large_v2, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0, return_confidence=True)
    base_v2 = (pho - large_v2) * lf / L - wide / L
    resid_v2 = base_v2 - sci
    is_clean_v2 = ((wide / np.maximum(pho, 1) < 0.3) & (sci > 100)
                   & np.isfinite(resid_v2) & ~np.isnan(mlat) & (abs_mlat < 5))
    C_det_map = np.full((3, 6), 120.0)
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_clean_v2
            if m.sum() > 100:
                C_det_map[bi, det] = float(np.mean(resid_v2[m]))
    print(f"C_det matrix:")
    print(f"     det0  det1  det2  det3  det4  det5")
    for bi, box in enumerate("ABC"):
        row = " ".join(f"{C_det_map[bi,d]:>5.0f}" for d in range(6))
        print(f"  {box}: {row}")

    # v3 + v4
    C_per_row = np.zeros(len(df))
    box_arr = df["box"].values
    det_arr = df["det"].values
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = (box_arr == box) & (det_arr == det)
            C_per_row[m] = C_det_map[bi, det]
    C_per_row += B_COEF * np.maximum(0, abs_mlat_safe - B_THRESHOLD)**2

    large_v3, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=C_per_row, return_confidence=True)

    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n_wraps_v3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    n_max = np.floor((max_large_event - large_raw) / 1024.0).astype(int)
    n_max = np.maximum(n_max, 0)
    n_wraps_v4 = np.where(n_wraps_v3 > n_max, n_max, n_wraps_v3)
    large_v4 = large_raw + n_wraps_v4 * 1024.0

    base_v4 = (pho - large_v4) * lf / L - wide / L
    residual_v4 = base_v4 - sci

    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = (box_arr == box) & (det_arr == det)
            C_det_per_row[m] = C_det_map[bi, det]
    B_per_row = B_COEF * np.maximum(0, abs_mlat_safe - B_THRESHOLD)**2

    residual_clean = residual_v4 - C_det_per_row - B_per_row

    is_valid = np.isfinite(base_v4) & np.isfinite(residual_v4) & (sci > 0) & (base_v4 > 0)
    wide_pho = wide / np.maximum(pho, 1)

    # Define blob: Sci_obs in [800, 2500], residual_clean in [-300, -50]
    is_blob = is_valid & (sci >= 800) & (sci <= 2500) & (residual_clean >= -300) & (residual_clean <= -50)
    is_main = is_valid & (sci >= 800) & (sci <= 2500) & (residual_clean >= -50) & (residual_clean <= 100)

    print(f"\n=== Blob characterization ===")
    print(f"Blob rows (Sci 800-2500, resid_clean -300 to -50): {is_blob.sum():,}")
    print(f"Main cloud (Sci 800-2500, resid_clean -50 to 100): {is_main.sum():,}")
    print(f"Blob / Main ratio: {is_blob.sum() / is_main.sum():.3f}")

    def chars(mask, label):
        sub_box = df["box"].values[mask]
        sub_det = df["det"].values[mask]
        sub_mlat = abs_mlat[mask]
        sub_wide_pho = wide_pho[mask]
        sub_dt = (dtv / lc)[mask]
        sub_n_wraps = n_wraps_v4[mask]
        sub_date = df["date"].values[mask]
        sub_C_det = C_det_per_row[mask]
        sub_B = B_per_row[mask]
        sub_sci = sci[mask]
        sub_residual_clean = residual_clean[mask]
        sub_lat = df["Lat"].values[mask]
        sub_lon = df["Lon"].values[mask]
        print(f"\n--- {label} ({mask.sum():,} rows) ---")
        print(f"  (box,det) breakdown:")
        for bi, box in enumerate("ABC"):
            for det in range(6):
                cnt = ((sub_box == box) & (sub_det == det)).sum()
                pct = cnt / mask.sum() * 100
                marker = " ★" if pct > 8 else ""
                print(f"    {box}-{det}: {cnt:>7,} ({pct:>4.1f}%){marker}")
        print(f"  |mlat|:   median={np.nanmedian(sub_mlat):.1f}, Q25={np.nanquantile(sub_mlat, 0.25):.1f}, Q75={np.nanquantile(sub_mlat, 0.75):.1f}")
        print(f"  Wide/PHO: median={np.median(sub_wide_pho):.3f}, Q25={np.quantile(sub_wide_pho, 0.25):.3f}, Q75={np.quantile(sub_wide_pho, 0.75):.3f}")
        print(f"  dt frac:  median={np.median(sub_dt):.3f}, Q75={np.quantile(sub_dt, 0.75):.3f}, Q99={np.quantile(sub_dt, 0.99):.3f}")
        print(f"  n_wraps:  distribution {dict((k, int((sub_n_wraps==k).sum())) for k in sorted(set(sub_n_wraps)))}")
        print(f"  C_det used: median={np.median(sub_C_det):.0f}")
        print(f"  B(|mlat|) used: median={np.median(sub_B):.0f}")
        print(f"  Sci_obs: median={np.median(sub_sci):.0f}, Q75={np.quantile(sub_sci, 0.75):.0f}")
        print(f"  residual_clean: median={np.median(sub_residual_clean):+.0f}")
        print(f"  Lat: median={np.median(sub_lat):.1f}, Lon: median={np.median(sub_lon):.1f}")
        unique_dates = pd.Series(sub_date).str[:7].value_counts().head(5)
        print(f"  Top 5 date months: {dict(unique_dates)}")

    chars(is_blob, "BLOB (residual ~ -150)")
    chars(is_main, "MAIN CLOUD (residual ~ 0)")

    # Differential ratio per (box, det): blob_fraction
    print("\n=== Blob-to-main ratio per (box, det) ===")
    print("     det0    det1    det2    det3    det4    det5")
    for bi, box in enumerate("ABC"):
        row = []
        for det in range(6):
            m_det = (df["box"].values == box) & (df["det"].values == det)
            b = (m_det & is_blob).sum()
            m = (m_det & is_main).sum()
            r = b / m if m > 0 else 0
            row.append(f"{r:>6.3f}")
        print(f"  {box}: " + "  ".join(row))


if __name__ == "__main__":
    main()
