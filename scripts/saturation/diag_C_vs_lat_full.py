#!/usr/bin/env python3
"""Detailed analysis of C(|Lat|, box, det) — find the per-det baseline AND
the common |Lat|-dependent particle background contribution.

Decomposition:
    C(|Lat|, box, det) = C_det(box, det) + B(|Lat|)

Where:
- C_det is per-detector electronics offset (the "equator" value, intrinsic to PMT/electronics)
- B(|Lat|) is a common function of latitude (cosmic ray secondary background)

Then plot B(|Lat|) and try fitting to a functional form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT_PNG = Path("plots/C_vs_lat.png")
L_CYCLES_TO_SEC = 16e-6

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat"]


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
    abs_lat = np.abs(df["Lat"].values)

    # v2 unwrap with C=195 (one pass; we're going to use n_wraps=0 only anyway)
    large_corr, conf = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=195.0, return_confidence=True,
    )
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean = ((conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100)
                & (n_wraps == 0) & np.isfinite(residual))
    print(f"  Clean (HIGH conf, non-magnetar, n_wraps=0, Sci>100): {is_clean.sum():,}")

    # Per-det × per-Lat-bin C estimation
    lat_edges = np.linspace(0, 45, 10)  # finer bins
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    n_bins = len(lat_centers)

    # C_table[box, det, lat_bin] = mean residual
    boxes = "ABC"
    C_table = np.full((3, 6, n_bins), np.nan)
    N_table = np.zeros((3, 6, n_bins), dtype=int)

    for bi, box in enumerate(boxes):
        for det in range(6):
            for li in range(n_bins):
                m = (((df["box"] == box) & (df["det"] == det)).values
                     & is_clean
                     & (abs_lat >= lat_edges[li]) & (abs_lat < lat_edges[li + 1]))
                if m.sum() < 500:
                    continue
                C_table[bi, det, li] = float(np.mean(residual[m]))
                N_table[bi, det, li] = m.sum()

    # Print table
    print("\n=== C(box, det, |Lat| bin) — mean residual ===")
    print(f"  {'(box,det)':<10}" + "  ".join(f"|Lat|={lat_centers[i]:>5.1f}" for i in range(n_bins)))
    for bi, box in enumerate(boxes):
        for det in range(6):
            cells = []
            for li in range(n_bins):
                v = C_table[bi, det, li]
                cells.append(f"{v:>+8.1f}" if not np.isnan(v) else "    -    ")
            print(f"  {box}-{det}:    " + "  ".join(cells))

    # Decomposition: C(box, det, lat) = C_det(box, det) + B(lat)
    # Define C_det as C at lowest valid lat bin (closest to equator), per (box, det)
    C_det_eq = C_table[:, :, 0]  # bin 0 (~2.5°)
    # B(lat) = average across (box, det) of (C(lat) − C_det_eq)
    delta = C_table - C_det_eq[:, :, None]  # shape (3, 6, n_bins)
    B_lat = np.nanmean(delta.reshape(18, n_bins), axis=0)
    B_lat_std = np.nanstd(delta.reshape(18, n_bins), axis=0)

    print(f"\n=== Common particle-background term B(|Lat|) (averaged across 18 dets, ref = bin0) ===")
    print(f"  {'|Lat|':<10}{'B (mean)':>10}{'B (std)':>10}")
    for li in range(n_bins):
        print(f"  {lat_centers[li]:>5.1f}°    {B_lat[li]:>+8.1f}  {B_lat_std[li]:>+8.1f}")

    print(f"\nC_det (equatorial baseline) per (box, det):")
    for bi, box in enumerate(boxes):
        row = "  " + " ".join(f"{box}{d}:{C_det_eq[bi,d]:+5.0f}" for d in range(6))
        print(row)
    print(f"  mean = {np.mean(C_det_eq):.1f}, std = {np.std(C_det_eq):.1f}, range [{np.min(C_det_eq):.0f}, {np.max(C_det_eq):.0f}]")

    # Try fitting B(|Lat|) to a model
    # Cosmic ray secondaries vs lat: rigidity cutoff R_c ≈ 14.9·cos⁴(λ_mag) GV
    # Background flux ∝ rigidity^(-1) approximately
    # Simpler empirical: B(lat) = a · sin^n(lat) + b·lat^2 etc.
    # Let me try: B(lat) = A · (1 − cos^n(lat·π/180))
    # Or polynomial: B(lat) = a + b·lat + c·lat^2

    # Polynomial fit
    valid = ~np.isnan(B_lat)
    if valid.sum() >= 3:
        coef = np.polyfit(lat_centers[valid], B_lat[valid], 2)
        print(f"\n  Quadratic fit B(lat) = {coef[0]:+.4f}·lat² + {coef[1]:+.4f}·lat + {coef[2]:+.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: per-det C(|Lat|) curves
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, 18))
    for bi, box in enumerate(boxes):
        for det in range(6):
            idx = bi * 6 + det
            ax.plot(lat_centers, C_table[bi, det, :], "o-", color=colors[idx],
                    markersize=4, lw=1, label=f"{box}-{det}")
    ax.set_xlabel("|Lat| (deg)", fontsize=11)
    ax.set_ylabel("C(box, det, |Lat|)  (cnt/s)", fontsize=11)
    ax.set_title("per-det C vs |Lat| — all 18 detectors", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2: common B(|Lat|) and fit
    ax = axes[1]
    ax.errorbar(lat_centers, B_lat, yerr=B_lat_std, fmt="ko-", markersize=6, lw=1.5,
                 label="B(|Lat|) = mean(C − C_eq) across 18 dets")
    if valid.sum() >= 3:
        xx = np.linspace(0, 45, 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, "r-", lw=2,
                 label=fr"quadratic fit: {coef[0]:+.4f}·lat² {coef[1]:+.4f}·lat {coef[2]:+.2f}")
    ax.set_xlabel("|Lat| (deg)", fontsize=11)
    ax.set_ylabel("B(|Lat|)  (cnt/s above equatorial baseline)", fontsize=11)
    ax.set_title("Common latitude-dependent background term", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", ls=":", lw=0.7)

    fig.suptitle(
        "C decomposition: C(|Lat|, box, det) = C_det(box, det) + B(|Lat|)\n"
        f"C_det range: [{np.min(C_det_eq):.0f}, {np.max(C_det_eq):.0f}] cnt/s. "
        f"B(|Lat|) rises from 0 to ~{np.nanmax(B_lat):.0f} cnt/s.",
        fontsize=12, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
