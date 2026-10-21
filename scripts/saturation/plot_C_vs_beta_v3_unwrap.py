#!/usr/bin/env python3
"""v3 unwrap: use full per-row C in predictor.

The v2 unwrap uses C=150 (global). For high-|mlat| rows on high-C detectors,
predicted_Large is biased high, causing over-correction (extra wrap added
when truth is 0 or fewer wraps).

v3 uses per-row C = C_det(box, det) + B(|mlat|), so predictor is accurate
across all detectors and orbital positions.

Output: plots/C_vs_beta_v3_unwrap.png
"""
from __future__ import annotations

import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_v3_unwrap.png")
L_CYCLES_TO_SEC = 16e-6

B_COEF = 0.26
B_THRESHOLD = 20.0

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]


def model_B(abs_mlat):
    return B_COEF * np.maximum(0.0, abs_mlat - B_THRESHOLD)**2


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    # AACGM grid
    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                      bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)
    abs_mlat_safe = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)

    pho = df["PHO"].values; large_raw = df["Large"].values
    wide = df["Wide"].values; sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values; dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    # === First pass: v2 with global C=150 to get rough C_det per det ===
    print("Pass 1: v2 with global C=150 to bootstrap...")
    large_corr_v2, conf_v2 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv,
                                              C=150.0, return_confidence=True)
    base_v2 = ((pho.astype("float64") - large_corr_v2) * lf - wide.astype("float64")) / L
    residual_v2 = (base_v2 - sci).values if hasattr(base_v2, "values") else (base_v2 - sci)
    is_clean_v2 = ((conf_v2 > CONF_LOW) & (wide / np.maximum(pho, 1) < 0.3) & (sci > 100)
                   & np.isfinite(residual_v2) & ~np.isnan(mlat) & (abs_mlat < 5))

    # Compute per-det C_det from |mlat|<5° clean rows
    C_det_map = np.full((3, 6), 120.0)
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_clean_v2
            if m.sum() > 100:
                C_det_map[bi, det] = float(np.mean(residual_v2[m]))
    print(f"  C_det: mean={C_det_map.mean():.1f}, range=[{C_det_map.min():.0f}, {C_det_map.max():.0f}]")

    # === Build per-row C array: C_det(box, det) + B(|mlat|) ===
    C_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m_dt = ((df["box"] == box) & (df["det"] == det)).values
            C_per_row[m_dt] = C_det_map[bi, det]
    C_per_row += model_B(abs_mlat_safe)
    print(f"  Per-row C: mean={C_per_row.mean():.1f}, median={np.median(C_per_row):.1f}, "
          f"range=[{C_per_row.min():.1f}, {C_per_row.max():.1f}]")

    # === Pass 2: v2 with full per-row C (= v3) ===
    print("\nPass 2 (v3): v2 unwrap with per-row C = C_det + B(|mlat|)...")
    large_corr_v3, conf_v3 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv,
                                              C=C_per_row, return_confidence=True)
    n_wraps_v3 = ((large_corr_v3 - large_raw.astype("float64")) / 1024).round().astype(int)
    base_v3 = ((pho.astype("float64") - large_corr_v3) * lf - wide.astype("float64")) / L
    residual_v3 = (base_v3 - sci).values if hasattr(base_v3, "values") else (base_v3 - sci)

    # n_wraps comparison
    n_wraps_v2 = ((large_corr_v2 - large_raw.astype("float64")) / 1024).round().astype(int)
    diff = n_wraps_v3 - n_wraps_v2
    print(f"\n=== v3 vs v2 n_wraps differences ===")
    for k in sorted(set(diff)):
        cnt = (diff == k).sum()
        if cnt > 0:
            print(f"  diff = {k:>+3}: {cnt:>12,}  ({cnt/len(diff)*100:>6.3f}%)")

    # Apply full model: residual_clean = residual_v3 - C_det - B(|mlat|)
    B_per_row = model_B(abs_mlat_safe)
    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            C_det_per_row[m] = C_det_map[bi, det]
    residual_clean = residual_v3 - C_det_per_row - B_per_row

    # === Plot ===
    is_valid = np.isfinite(base_v3.values if hasattr(base_v3, "values") else base_v3) & np.isfinite(residual_v3) & (sci > 0)
    base_arr = base_v3.values if hasattr(base_v3, "values") else base_v3
    is_valid = is_valid & (base_arr > 0)
    base_s = base_arr[is_valid]
    sci_s = sci[is_valid].astype("float32")
    resid_s = residual_v3[is_valid]
    resid_clean_s = residual_clean[is_valid]

    N = min(300_000, len(base_s))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base_s), N, replace=False)
    base_p = base_s[idx]; sci_p = sci_s[idx]
    resid_p = resid_s[idx]; resid_clean_p = resid_clean_s[idx]

    LO, HI = 30.0, 10_000.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7.5))

    # Panel 1
    xb = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb = np.logspace(np.log10(LO), np.log10(HI), 150)
    H, xe, ye = np.histogram2d(sci_p, base_p, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xe, sci_p) - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(ye, base_p) - 1, 0, len(ye) - 2)
    dens = H[ix, iy].astype(float); dens[dens<1]=1
    order = np.argsort(dens)
    ax1.scatter(sci_p[order], base_p[order], c=dens[order], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    xx = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax1.plot(xx, xx, "k--", lw=1.5, label="y = x")
    c_mid = (C_det_map.min() + C_det_map.max()) / 2
    ax1.plot(xx, xx + C_det_map.min(), "b-", lw=1.5, label=fr"y = x + {C_det_map.min():.0f} (C_det min)")
    ax1.plot(xx, xx + c_mid, "b-", lw=2.0, label=fr"y = x + {c_mid:.0f} (C_det mean)")
    ax1.plot(xx, xx + C_det_map.max() + model_B(43), "b-", lw=1.5,
              label=fr"y = x + {C_det_map.max() + model_B(43):.0f} (C_det max + B at 43°)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci_1s observed (cnt/s)")
    ax1.set_ylabel("Sci_pred_base with v3 unwrap (cnt/s)")
    ax1.set_title("log-log Sci_pred vs Sci_obs — v3 unwrap with per-row C", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel 2
    Y_LO, Y_HI = -500, 1500
    in_range2 = (sci_p >= LO) & (sci_p <= HI) & (resid_p >= Y_LO) & (resid_p <= Y_HI)
    sci_2 = sci_p[in_range2]; resid_2 = resid_p[in_range2]
    xb2 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb2 = np.linspace(Y_LO, Y_HI, 150)
    H2, xe2, ye2 = np.histogram2d(sci_2, resid_2, bins=[xb2, yb2])
    ix2 = np.clip(np.searchsorted(xe2, sci_2) - 1, 0, len(xe2) - 2)
    iy2 = np.clip(np.searchsorted(ye2, resid_2) - 1, 0, len(ye2) - 2)
    dens2 = H2[ix2, iy2].astype(float); dens2[dens2<1]=1
    order2 = np.argsort(dens2)
    ax2.scatter(sci_2[order2], resid_2[order2], c=dens2[order2], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens2.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    ax2.axhline(C_det_map.mean(), color="blue", lw=2.0,
                 label=fr"C_det mean = {C_det_map.mean():.0f}")
    ax2.axhline(0, color="k", ls=":", lw=0.7)
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci_1s observed (cnt/s)")
    ax2.set_ylabel("residual = base − Sci_obs (cnt/s, BEFORE model)")
    ax2.set_title("density scatter: residual BEFORE model (v3 unwrap)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3
    in_range3 = (sci_p >= LO) & (sci_p <= HI) & (resid_clean_p >= Y_LO) & (resid_clean_p <= Y_HI)
    sci_3 = sci_p[in_range3]; resid_clean_3 = resid_clean_p[in_range3]
    H3, xe3, ye3 = np.histogram2d(sci_3, resid_clean_3, bins=[xb2, yb2])
    ix3 = np.clip(np.searchsorted(xe3, sci_3) - 1, 0, len(xe3) - 2)
    iy3 = np.clip(np.searchsorted(ye3, resid_clean_3) - 1, 0, len(ye3) - 2)
    dens3 = H3[ix3, iy3].astype(float); dens3[dens3<1]=1
    order3 = np.argsort(dens3)
    ax3.scatter(sci_3[order3], resid_clean_3[order3], c=dens3[order3], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens3.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    ax3.axhline(0, color="r", lw=2.0, label="zero (perfect model)")
    ax3.set_xscale("log"); ax3.set_xlim(LO, HI); ax3.set_ylim(Y_LO, Y_HI)
    ax3.set_xlabel("Sci_1s observed (cnt/s)")
    ax3.set_ylabel("residual_clean (cnt/s, AFTER full model)")
    ax3.set_title("residual AFTER full model — should be ~0 (v3)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "v3 unwrap: predictor uses per-row C = C_det(box, det) + B(|mlat|)  →  fixes over-correction at high |mlat|\n"
        fr"$C_\mathrm{{det}}$: 18 values [{C_det_map.min():.0f}, {C_det_map.max():.0f}].  "
        fr"$B(|m\mathrm{{lat}}|) = {B_COEF} \cdot \max(0, |m\mathrm{{lat}}| − {B_THRESHOLD:.0f}°)^2$",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")

    # Stats
    in_main = is_valid & (sci > 300) & (sci < 2000) & (np.abs(residual_v3) < 2000)
    print(f"\nv2 unwrap residual BEFORE model: median={np.median(residual_v2[in_main]):+.1f}")
    print(f"v3 unwrap residual BEFORE model: median={np.median(residual_v3[in_main]):+.1f}")
    print(f"v3 unwrap residual AFTER  full model: median={np.median(residual_clean[in_main]):+.1f}, "
          f"Q25={np.quantile(residual_clean[in_main], 0.25):+.1f}, Q75={np.quantile(residual_clean[in_main], 0.75):+.1f}")

    # Count negative residual cloud
    neg_clean = (residual_clean > -800) & (residual_clean < -200)
    print(f"\nNegative-residual cloud (-800 < resid_clean < -200):")
    print(f"  v3 count: {(neg_clean & is_valid).sum():,}")
    # Compare with v2 for same definition
    residual_clean_v2 = residual_v2 - C_det_per_row - B_per_row
    neg_v2 = (residual_clean_v2 > -800) & (residual_clean_v2 < -200)
    print(f"  v2 count: {(neg_v2 & is_valid).sum():,}")


if __name__ == "__main__":
    main()
