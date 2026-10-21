#!/usr/bin/env python3
"""C vs (|Lat|, Lon) 2D diagnostic — looks for SAA or other spatial structure
that might explain the 32.5° kink in the 1D C(|Lat|) plot.

Subtracts per-det C_det (equatorial baseline) so we look at the ENVIRONMENTAL
contribution B(lat, lon) only.

Output: plots/C_lat_lon_2d.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT_PNG = Path("plots/C_lat_lon_2d.png")
L_CYCLES_TO_SEC = 16e-6

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    pho = df["PHO"].values
    large_raw = df["Large"].values
    wide = df["Wide"].values
    sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values
    dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")
    lat_signed = df["Lat"].values
    abs_lat = np.abs(lat_signed)
    lon = df["Lon"].values

    large_corr, conf = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=195.0, return_confidence=True,
    )
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean = ((conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100)
                & (n_wraps == 0) & np.isfinite(residual))
    print(f"  Clean rows: {is_clean.sum():,}")

    # Subtract per-det equatorial baseline C_det (compute from |Lat|<5°)
    print("\nComputing per-det equatorial baseline (|Lat|<5°)...")
    is_eq = is_clean & (abs_lat < 5)
    C_det_map = np.zeros((3, 6))
    boxes = "ABC"
    for bi, box in enumerate(boxes):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_eq
            if m.sum() < 100:
                C_det_map[bi, det] = 120.0
            else:
                C_det_map[bi, det] = float(np.mean(residual[m]))

    # Per-row C_det
    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate(boxes):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            C_det_per_row[m] = C_det_map[bi, det]
    print(f"  C_det range: [{C_det_map.min():.0f}, {C_det_map.max():.0f}]")

    # B = residual - C_det
    B_per_row = residual - C_det_per_row

    # === 2D binning: signed Lat × Lon ===
    print("\nBuilding 2D Lat-Lon map...")
    # HXMT Lat range: -43 to +43
    # HXMT Lon: 0 to 360 (HXMT convention)
    lat_edges = np.linspace(-45, 45, 19)   # 5° per bin
    lon_edges = np.linspace(0, 360, 25)    # 15° per bin
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    B_2d = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    N_2d = np.zeros_like(B_2d, dtype=int)

    # Use vectorized binning
    lat_idx = np.digitize(lat_signed, lat_edges) - 1
    lon_idx = np.digitize(lon, lon_edges) - 1
    valid_idx = is_clean & (lat_idx >= 0) & (lat_idx < len(lat_edges) - 1) & (lon_idx >= 0) & (lon_idx < len(lon_edges) - 1)

    for li in range(len(lat_edges) - 1):
        for loi in range(len(lon_edges) - 1):
            m = valid_idx & (lat_idx == li) & (lon_idx == loi)
            if m.sum() < 100:
                continue
            B_2d[li, loi] = float(np.mean(B_per_row[m]))
            N_2d[li, loi] = m.sum()

    # === |Lat| × Lon (folded) ===
    abs_lat_edges = np.linspace(0, 45, 10)
    abs_lat_centers = 0.5 * (abs_lat_edges[:-1] + abs_lat_edges[1:])

    B_2d_abs = np.full((len(abs_lat_edges) - 1, len(lon_edges) - 1), np.nan)

    abs_lat_idx = np.digitize(abs_lat, abs_lat_edges) - 1
    valid_idx2 = is_clean & (abs_lat_idx >= 0) & (abs_lat_idx < len(abs_lat_edges) - 1) & (lon_idx >= 0) & (lon_idx < len(lon_edges) - 1)
    for li in range(len(abs_lat_edges) - 1):
        for loi in range(len(lon_edges) - 1):
            m = valid_idx2 & (abs_lat_idx == li) & (lon_idx == loi)
            if m.sum() < 100:
                continue
            B_2d_abs[li, loi] = float(np.mean(B_per_row[m]))

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    # Panel 1: signed Lat × Lon
    ax = axes[0]
    vmax = np.nanpercentile(B_2d, 98)
    im = ax.imshow(B_2d, origin="lower", aspect="auto",
                    extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
                    cmap="hot_r", norm=Normalize(vmin=0, vmax=vmax))
    ax.set_xlabel("Lon (°)", fontsize=11)
    ax.set_ylabel("signed Lat (°)", fontsize=11)
    ax.set_title("B (Lat, Lon) — environmental background (cnt/s above eq. baseline)", fontsize=11)
    plt.colorbar(im, ax=ax, label="B (cnt/s)")
    # Mark SAA approximate boundary
    saa_box_lon = [270, 30 + 360]  # SAA covers Lon roughly 270-30 wrapping at 360
    # Easier: SAA is roughly at Lat -40° to +10°, Lon 270-360 and 0-30
    # Draw a box for visualization
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((270, -40), 90, 50, fill=False, edgecolor="blue", lw=1.5, ls="--", label="SAA Lon 270-360°"))
    ax.add_patch(Rectangle((0, -40), 30, 50, fill=False, edgecolor="blue", lw=1.5, ls="--"))
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: |Lat| × Lon (folded)
    ax = axes[1]
    vmax2 = np.nanpercentile(B_2d_abs, 98)
    im2 = ax.imshow(B_2d_abs, origin="lower", aspect="auto",
                     extent=[lon_edges[0], lon_edges[-1], abs_lat_edges[0], abs_lat_edges[-1]],
                     cmap="hot_r", norm=Normalize(vmin=0, vmax=vmax2))
    ax.set_xlabel("Lon (°)", fontsize=11)
    ax.set_ylabel("|Lat| (°)", fontsize=11)
    ax.set_title("B (|Lat|, Lon) — folded around equator", fontsize=11)
    plt.colorbar(im2, ax=ax, label="B (cnt/s)")
    # Mark suspected SAA again
    ax.add_patch(Rectangle((270, 0), 90, 40, fill=False, edgecolor="blue", lw=1.5, ls="--"))
    ax.add_patch(Rectangle((0, 0), 30, 40, fill=False, edgecolor="blue", lw=1.5, ls="--"))

    fig.suptitle(
        "B = residual − C_det per (box, det)  on n_wraps=0 clean rows\n"
        "Looking for Lon-dependent structure (SAA?) explaining 32.5° kink in 1D fit",
        fontsize=12, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")

    # Print: B at |Lat|=30-35° as function of Lon (the suspicious bin)
    print("\n=== B(|Lat|=30-35°, Lon) — the suspicious bin ===")
    # |Lat| bin 30-35° → abs_lat_idx where (30, 35] → bin 6 (centers 32.5)
    # Use signed
    li_north = np.argmin(np.abs(lat_centers - 32.5))
    li_south = np.argmin(np.abs(lat_centers - (-32.5)))
    print(f"  Lon bin    B(Lat=+32.5°)  B(Lat=-32.5°)")
    for loi in range(len(lon_centers)):
        b_n = B_2d[li_north, loi]
        b_s = B_2d[li_south, loi]
        bn_s = f"{b_n:>+8.0f}" if not np.isnan(b_n) else "    -    "
        bs_s = f"{b_s:>+8.0f}" if not np.isnan(b_s) else "    -    "
        print(f"  Lon={lon_centers[loi]:>5.0f}°    {bn_s}        {bs_s}")


if __name__ == "__main__":
    main()
