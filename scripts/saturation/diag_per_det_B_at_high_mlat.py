#!/usr/bin/env python3
"""Per-det B at high |mlat| — does it match the uniform 0.26·(|mlat|-20)² model?

Compute actual residual (base - Sci - C_det) for each (box, det) in mlat bins,
to see whether B's slope is the same across all 18 dets or varies per-det.

If the lower blob is "per-det B mismatch":
  - low-C_det dets (A-4 88, B-5 90, B-4 102) should show B significantly LESS
    than 0.26·(|mlat|-20)² average prediction
  - high-C_det dets (B-2 200) should show B significantly MORE
  - correlation: real B ∝ C_det
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
    box_arr = df["box"].values
    det_arr = df["det"].values

    # v2 → C_det
    large_v2, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0, return_confidence=True)
    base_v2 = (pho - large_v2) * lf / L - wide / L
    resid_v2 = base_v2 - sci
    is_clean_v2 = ((wide / np.maximum(pho, 1) < 0.3) & (sci > 100)
                   & np.isfinite(resid_v2) & ~np.isnan(mlat) & (abs_mlat < 5))
    C_det_map = np.full((3, 6), 120.0)
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((box_arr == box) & (det_arr == det)) & is_clean_v2
            if m.sum() > 100:
                C_det_map[bi, det] = float(np.mean(resid_v2[m]))

    # v4 unwrap
    C_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = (box_arr == box) & (det_arr == det)
            C_per_row[m] = C_det_map[bi, det]
    C_per_row += B_COEF * np.maximum(0, abs_mlat_safe - B_THRESHOLD)**2

    large_v3, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=C_per_row, return_confidence=True)
    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n_wraps_v3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    n_max = np.maximum(np.floor((max_large_event - large_raw) / 1024.0).astype(int), 0)
    n_wraps_v4 = np.where(n_wraps_v3 > n_max, n_max, n_wraps_v3)
    large_v4 = large_raw + n_wraps_v4 * 1024.0

    base_v4 = (pho - large_v4) * lf / L - wide / L
    residual_v4 = base_v4 - sci

    # Subtract C_det only (NOT B), so we look at pure mlat-dependent residual
    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = (box_arr == box) & (det_arr == det)
            C_det_per_row[m] = C_det_map[bi, det]
    residual_B = residual_v4 - C_det_per_row   # this should be ≈ B(|mlat|) only

    is_valid = (np.isfinite(residual_B) & ~np.isnan(mlat) & (sci > 100)
                & (wide / np.maximum(pho, 1) < 0.3) & (n_wraps_v4 == 0))

    # Per-det B at |mlat| bins
    mlat_bins = [(0, 5), (10, 15), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
    print("\n=== Per-det B(|mlat|) measured residual (n_wraps=0 only) ===\n")
    print(f"  Model B(|mlat|) = 0.26 · max(0, |mlat|-20)²")
    for lo, hi in mlat_bins:
        mid = (lo + hi) / 2
        B_model = B_COEF * max(0, mid - B_THRESHOLD)**2
        print(f"\n  |mlat| in [{lo},{hi}) — model B = {B_model:.0f} cnt/s")
        print(f"        det0    det1    det2    det3    det4    det5")
        for bi, box in enumerate("ABC"):
            row_str = []
            for det in range(6):
                m_dt = (box_arr == box) & (det_arr == det)
                m = m_dt & is_valid & (abs_mlat >= lo) & (abs_mlat < hi)
                if m.sum() < 100:
                    row_str.append("  N/A ")
                else:
                    B_actual = float(np.median(residual_B[m]))
                    delta = B_actual - B_model
                    row_str.append(f"{B_actual:>4.0f}({delta:+4.0f})")
            cdet_str = " ".join(f"{int(C_det_map[bi, d]):>3}" for d in range(6))
            print(f"    {box}: " + "  ".join(row_str) + f"   | C_det: {cdet_str}")

    # Direct check: at high mlat (45-50°), correlate B_actual with C_det
    print("\n=== Correlation: high-|mlat| B_actual vs equatorial C_det ===\n")
    lo, hi = 45, 50
    B_model = B_COEF * (47.5 - 20)**2
    C_det_list, B_actual_list, label_list = [], [], []
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m_dt = (box_arr == box) & (det_arr == det)
            m = m_dt & is_valid & (abs_mlat >= lo) & (abs_mlat < hi)
            if m.sum() < 100:
                continue
            B_actual = float(np.median(residual_B[m]))
            C_det_list.append(C_det_map[bi, det])
            B_actual_list.append(B_actual)
            label_list.append(f"{box}-{det}")

    C_arr = np.array(C_det_list)
    B_arr = np.array(B_actual_list)
    print(f"  Model B at |mlat|=47.5: {B_model:.0f}")
    print(f"  C_det range: [{C_arr.min():.0f}, {C_arr.max():.0f}]")
    print(f"  B_actual range: [{B_arr.min():.0f}, {B_arr.max():.0f}]")
    print(f"  Pearson correlation B_actual vs C_det: {np.corrcoef(C_arr, B_arr)[0,1]:.3f}")
    # Linear fit B_actual = a + b*C_det
    coef = np.polyfit(C_arr, B_arr, 1)
    print(f"  Linear fit: B_actual = {coef[0]:.2f} · C_det + {coef[1]:.0f}")
    print(f"\n  Sorted by C_det:")
    print(f"    det   C_det   B_actual   B_actual - B_model")
    order = np.argsort(C_arr)
    for i in order:
        delta = B_arr[i] - B_model
        marker = " ★" if abs(delta) > 30 else ""
        print(f"    {label_list[i]:>5}    {C_arr[i]:>3.0f}      {B_arr[i]:>4.0f}        {delta:+4.0f}{marker}")


if __name__ == "__main__":
    main()
