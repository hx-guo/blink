#!/usr/bin/env python3
"""Re-do C(|Lat|) analysis using AACGM magnetic latitude (better than geographic).

Strategy:
1. Precompute AACGM mlat on a (Lat, Lon) grid at 540 km altitude, date 2020-06-15
   (HXMT alt varies ±20 km, mlat is essentially insensitive; 2020 epoch is mid-mission)
2. Interpolate to each row's (Lat, Lon) — 20M rows in seconds via scipy
3. Re-bin C(|mlat|) and compare scatter against C(|Lat|)
"""
from __future__ import annotations

import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import aacgmv2

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT_PNG = Path("plots/C_vs_aacgm_lat.png")
L_CYCLES_TO_SEC = 16e-6

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]


def build_aacgm_grid(date=datetime.datetime(2020, 6, 15), alt_km=540.0):
    """Compute AACGM mlat on a (Lat, Lon) grid at fixed altitude."""
    print(f"Building AACGM grid for {date.date()} at alt={alt_km}km...")
    lat_grid = np.linspace(-45, 45, 91)   # 1° resolution
    lon_grid = np.linspace(0, 360, 181)   # 2° resolution

    mlat_arr = np.zeros((len(lat_grid), len(lon_grid)))
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            mlat, _, _ = aacgmv2.get_aacgm_coord(float(lat), float(lon), alt_km, date)
            mlat_arr[i, j] = mlat if not np.isnan(mlat) else 0.0
    return lat_grid, lon_grid, mlat_arr


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    # Build AACGM mlat grid + interpolator
    lat_grid, lon_grid, mlat_grid = build_aacgm_grid()
    interp = RegularGridInterpolator((lat_grid, lon_grid), mlat_grid,
                                      bounds_error=False, fill_value=np.nan)

    # Compute mlat for each row
    print("Interpolating mlat for each row...")
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)
    print(f"  Valid mlat rows: {np.sum(~np.isnan(mlat)):,} / {len(mlat):,}")

    # Apply unwrap
    pho = df["PHO"].values
    large_raw = df["Large"].values
    wide = df["Wide"].values
    sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values
    dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    large_corr, conf = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=195.0, return_confidence=True,
    )
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean = ((conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100)
                & (n_wraps == 0) & np.isfinite(residual) & ~np.isnan(mlat))
    print(f"  Clean rows: {is_clean.sum():,}")

    # Per-det equatorial baseline (|mlat|<5°)
    abs_lat = np.abs(df["Lat"].values)
    is_eq_mag = is_clean & (abs_mlat < 5)
    C_det_map = np.zeros((3, 6))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_eq_mag
            C_det_map[bi, det] = float(np.mean(residual[m])) if m.sum() > 100 else 120.0
    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            C_det_per_row[m] = C_det_map[bi, det]

    B_per_row = residual - C_det_per_row

    # Bin by |mlat| AND by |Lat| (for comparison)
    edges = np.linspace(0, 45, 10)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def bin_per_det(coord, edges, mask):
        """Returns table[box, det, bin] of mean residual."""
        n_bins = len(edges) - 1
        table = np.full((3, 6, n_bins), np.nan)
        for bi, box in enumerate("ABC"):
            for det in range(6):
                for li in range(n_bins):
                    m = (((df["box"] == box) & (df["det"] == det)).values
                         & mask & (coord >= edges[li]) & (coord < edges[li + 1]))
                    if m.sum() < 200:
                        continue
                    table[bi, det, li] = float(np.mean(B_per_row[m]))
        return table

    print("Binning by |mlat| (AACGM)...")
    B_aacgm = bin_per_det(abs_mlat, edges, is_clean)
    print("Binning by |Lat| (geographic) for comparison...")
    B_geo = bin_per_det(abs_lat, edges, is_clean)

    # Compute common B function (mean across 18 dets, ref bin 0)
    def common_B(table):
        # Subtract bin0 per-det
        ref = table[:, :, 0]
        delta = table - ref[:, :, None]
        mean = np.nanmean(delta.reshape(18, -1), axis=0)
        std = np.nanstd(delta.reshape(18, -1), axis=0)
        return mean, std

    aacgm_mean, aacgm_std = common_B(B_aacgm)
    geo_mean, geo_std = common_B(B_geo)

    print("\n=== Comparison: B(|coord|) using AACGM mlat vs geographic |Lat| ===")
    print(f"  {'bin':<6}{'center':>8}{'B_geo':>10}{'std_geo':>10}{'B_aacgm':>10}{'std_aacgm':>12}{'scatter_ratio':>16}")
    for i, c in enumerate(centers):
        ratio = aacgm_std[i] / max(geo_std[i], 1e-9)
        print(f"  bin{i:<3}{c:>7.1f}°  {geo_mean[i]:>+8.1f}  {geo_std[i]:>8.1f}  "
              f"{aacgm_mean[i]:>+8.1f}  {aacgm_std[i]:>10.1f}  {ratio:>14.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Panel 1: B vs |Lat| with error bars
    ax = axes[0]
    ax.errorbar(centers, geo_mean, yerr=geo_std, fmt="bo-", markersize=6, lw=1.5,
                 label=f"geographic |Lat|  (std avg = {np.nanmean(geo_std):.1f})")
    ax.errorbar(centers, aacgm_mean, yerr=aacgm_std, fmt="rs-", markersize=6, lw=1.5,
                 label=f"AACGM |mlat|  (std avg = {np.nanmean(aacgm_std):.1f})")
    ax.set_xlabel("|coord| (deg)", fontsize=11)
    ax.set_ylabel("B (cnt/s, above equatorial baseline)", fontsize=11)
    ax.set_title("B(|Lat|) vs B(|mlat|) — error bars show across-det std", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", ls=":", lw=0.7)

    # Panel 2: 18 per-det curves vs |mlat|
    ax = axes[1]
    colors = plt.cm.tab20(np.linspace(0, 1, 18))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            idx = bi * 6 + det
            ax.plot(centers, B_aacgm[bi, det, :] - C_det_map[bi, det] * 0 + C_det_map[bi, det],
                    "o-", color=colors[idx], markersize=4, lw=1, label=f"{box}-{det}")
    ax.set_xlabel("|AACGM mlat| (deg)", fontsize=11)
    ax.set_ylabel("C(box, det, |mlat|)  (cnt/s)", fontsize=11)
    ax.set_title("per-det C vs |mlat| — should be parallel if mlat is right coordinate", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"C decomposition with AACGM magnetic latitude\n"
        f"avg scatter: geographic |Lat| std = {np.nanmean(geo_std):.1f}, AACGM |mlat| std = {np.nanmean(aacgm_std):.1f}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
