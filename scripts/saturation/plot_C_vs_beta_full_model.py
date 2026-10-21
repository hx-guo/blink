#!/usr/bin/env python3
"""C-vs-β diagnostic with the FINAL model applied:
    Sci_pred = base − C_det(box, det) − B(|mlat|)
    where B(|mlat|) = 0.26 · max(0, |mlat| − 20°)²

Three panels:
  1. log-log Sci_pred vs Sci_obs (with model line)
  2. residual_raw vs Sci_obs (before model subtraction)
  3. residual_clean = residual − C_det − B(|mlat|) vs Sci_obs (after — should be ~0)

Output: plots/C_vs_beta_full_model.png
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
import aacgmv2

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_full_model.png")
L_CYCLES_TO_SEC = 16e-6

# Final model params
B_COEF = 0.26
B_THRESHOLD = 20.0

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]


def build_aacgm_grid(date=datetime.datetime(2020, 6, 15), alt_km=540.0):
    print(f"Building AACGM grid for {date.date()} at alt={alt_km}km...")
    lat_grid = np.linspace(-45, 45, 91)
    lon_grid = np.linspace(0, 360, 181)
    mlat_arr = np.zeros((len(lat_grid), len(lon_grid)))
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                mlat, _, _ = aacgmv2.get_aacgm_coord(float(lat), float(lon), alt_km, date)
                mlat_arr[i, j] = mlat if not np.isnan(mlat) else 0.0
    return lat_grid, lon_grid, mlat_arr


def model_B(abs_mlat):
    return B_COEF * np.maximum(0.0, abs_mlat - B_THRESHOLD)**2


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    lat_grid, lon_grid, mlat_grid = build_aacgm_grid()
    interp = RegularGridInterpolator((lat_grid, lon_grid), mlat_grid,
                                      bounds_error=False, fill_value=np.nan)
    print("Interpolating mlat...")
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)

    pho = df["PHO"].values; large_raw = df["Large"].values
    wide = df["Wide"].values; sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values; dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    print("Applying unwrap_large_v2...")
    large_corr, conf = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0,
                                        return_confidence=True)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean_eq = ((conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100)
                   & np.isfinite(residual) & ~np.isnan(mlat) & (abs_mlat < 5))

    # Compute per-det C_det from equatorial (|mlat|<5°) clean rows
    print("Computing per-det C_det from |mlat|<5°...")
    C_det_map = np.zeros((3, 6))
    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m_dt = ((df["box"] == box) & (df["det"] == det)).values
            m = m_dt & is_clean_eq
            cval = float(np.mean(residual[m])) if m.sum() > 100 else 120.0
            C_det_map[bi, det] = cval
            C_det_per_row[m_dt] = cval
    print(f"  C_det: mean={C_det_map.mean():.1f}, range=[{C_det_map.min():.0f}, {C_det_map.max():.0f}]")

    # B(|mlat|)
    B_per_row = model_B(abs_mlat)
    B_per_row[np.isnan(B_per_row)] = 0.0

    # Cleaned residual
    residual_clean = residual - C_det_per_row - B_per_row

    # For plot, use a sample
    is_valid = np.isfinite(base) & np.isfinite(residual) & (base > 0) & (sci > 0)
    base_arr = base.values if hasattr(base, "values") else base
    base_s = base_arr[is_valid]
    sci_s = sci[is_valid].astype("float32")
    resid_s = residual[is_valid]
    resid_clean_s = residual_clean[is_valid]
    print(f"Valid rows for plot: {is_valid.sum():,}")

    N = min(300_000, len(base_s))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base_s), N, replace=False)
    base_p = base_s[idx]
    sci_p = sci_s[idx]
    resid_p = resid_s[idx]
    resid_clean_p = resid_clean_s[idx]

    LO, HI = 30.0, 10_000.0

    # =================== Plot ===================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7.5))

    # Panel 1: Sci_pred vs Sci_obs (log-log) with FINAL MODEL band
    xb = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb = np.logspace(np.log10(LO), np.log10(HI), 150)
    H, xe, ye = np.histogram2d(sci_p, base_p, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xe, sci_p) - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(ye, base_p) - 1, 0, len(ye) - 2)
    dens = H[ix, iy].astype(float); dens[dens < 1] = 1
    order = np.argsort(dens)
    ax1.scatter(sci_p[order], base_p[order], c=dens[order], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    xx = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax1.plot(xx, xx, "k--", lw=1.5, label="y = x")
    # Model band: y = x + C_det + B(|mlat|), where C_det varies [87, 174] and B varies [0, 128]
    # So y ranges from x + 87 (low mlat, low C det) to x + 174 + 128 = x + 302 (high mlat, B-2 det)
    c_min = C_det_map.min(); c_max = C_det_map.max()
    c_mid = (c_min + c_max) / 2
    ax1.plot(xx, xx + c_min, "b-", lw=1.5, label=fr"y = x + {c_min:.0f} (C_det min, eq.)")
    ax1.plot(xx, xx + c_mid, "b-", lw=2.0, label=fr"y = x + {c_mid:.0f} (C_det mean)")
    ax1.plot(xx, xx + c_max + model_B(43), "b-", lw=1.5,
              label=fr"y = x + {c_max + model_B(43):.0f} (C_det max + B at |mlat|=43°)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci_1s observed (cnt/s)")
    ax1.set_ylabel("Sci_pred_base (cnt/s)")
    ax1.set_title("log-log Sci_pred vs Sci_obs (unwrap applied)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel 2: raw residual vs Sci_obs (linear Y)
    Y_LO, Y_HI = -500, 1500
    in_range2 = (sci_p >= LO) & (sci_p <= HI) & (resid_p >= Y_LO) & (resid_p <= Y_HI)
    sci_2 = sci_p[in_range2]; resid_2 = resid_p[in_range2]
    xb2 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb2 = np.linspace(Y_LO, Y_HI, 150)
    H2, xe2, ye2 = np.histogram2d(sci_2, resid_2, bins=[xb2, yb2])
    ix2 = np.clip(np.searchsorted(xe2, sci_2) - 1, 0, len(xe2) - 2)
    iy2 = np.clip(np.searchsorted(ye2, resid_2) - 1, 0, len(ye2) - 2)
    dens2 = H2[ix2, iy2].astype(float); dens2[dens2 < 1] = 1
    order2 = np.argsort(dens2)
    ax2.scatter(sci_2[order2], resid_2[order2], c=dens2[order2], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens2.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    ax2.axhline(C_det_map.mean(), color="blue", lw=2.0,
                 label=fr"C_det mean = {C_det_map.mean():.0f} (no B term)")
    ax2.axhline(0, color="k", ls=":", lw=0.7)
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci_1s observed (cnt/s)")
    ax2.set_ylabel("residual = base − Sci_obs (cnt/s, BEFORE model)")
    ax2.set_title("density scatter: residual BEFORE model — linear Y", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3: CLEANED residual vs Sci_obs (after model subtracted)
    in_range3 = (sci_p >= LO) & (sci_p <= HI) & (resid_clean_p >= Y_LO) & (resid_clean_p <= Y_HI)
    sci_3 = sci_p[in_range3]; resid_clean_3 = resid_clean_p[in_range3]
    xb3 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb3 = np.linspace(Y_LO, Y_HI, 150)
    H3, xe3, ye3 = np.histogram2d(sci_3, resid_clean_3, bins=[xb3, yb3])
    ix3 = np.clip(np.searchsorted(xe3, sci_3) - 1, 0, len(xe3) - 2)
    iy3 = np.clip(np.searchsorted(ye3, resid_clean_3) - 1, 0, len(ye3) - 2)
    dens3 = H3[ix3, iy3].astype(float); dens3[dens3 < 1] = 1
    order3 = np.argsort(dens3)
    ax3.scatter(sci_3[order3], resid_clean_3[order3], c=dens3[order3], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens3.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    ax3.axhline(0, color="r", lw=2.0, label="zero (perfect model)")
    ax3.set_xscale("log"); ax3.set_xlim(LO, HI); ax3.set_ylim(Y_LO, Y_HI)
    ax3.set_xlabel("Sci_1s observed (cnt/s)")
    ax3.set_ylabel("residual_clean = base − Sci_obs − C_det − B(|mlat|)  (cnt/s)")
    ax3.set_title("density scatter: residual AFTER full model — should be ~0", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "FINAL model applied:  Sci_pred = (PHO − Large_unwrap)·(1−dt)/L − Wide/L − C_det(box,det) − B(|mlat|)\n"
        fr"$C_\mathrm{{det}}$: 18 per-det values [{C_det_map.min():.0f}, {C_det_map.max():.0f}] cnt/s.    "
        fr"$B(|m\mathrm{{lat}}|) = {B_COEF:.2f} \cdot \max(0, |m\mathrm{{lat}}| − {B_THRESHOLD:.0f}°)^2$  cnt/s",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")

    # Print residual statistics before vs after
    in_main = is_valid & (sci > 300) & (sci < 2000) & (np.abs(residual) < 2000)
    print(f"\nResidual BEFORE model: median={np.median(residual[in_main]):+.1f}, "
          f"Q25={np.quantile(residual[in_main], 0.25):+.1f}, Q75={np.quantile(residual[in_main], 0.75):+.1f}")
    print(f"Residual AFTER full model (clean): median={np.median(residual_clean[in_main]):+.1f}, "
          f"Q25={np.quantile(residual_clean[in_main], 0.25):+.1f}, Q75={np.quantile(residual_clean[in_main], 0.75):+.1f}")


if __name__ == "__main__":
    main()
