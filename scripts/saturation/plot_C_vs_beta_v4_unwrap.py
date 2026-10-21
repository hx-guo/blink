#!/usr/bin/env python3
"""v4 unwrap: v3 + strict event-balance sanity cap.

Constraint: base ≥ Sci + min_C (= 50 cnt/s), i.e., Sci_pred should never go
below Sci_obs (would mean Large_corr is unphysically high).

This catches over-correction that v3's predictor doesn't prevent —
specifically high-|mlat| + high-C_det rows where any rounding error >0.5 wrap
pushes Large_corr too high.

Algorithm: after v3 unwrap, check each row. If base < Sci + min_C, reduce
n_wraps by 1 until satisfied.

Output: plots/C_vs_beta_v4_unwrap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_v4_unwrap.png")
L_CYCLES_TO_SEC = 16e-6

B_COEF = 0.26
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0   # min residual we allow (in cnt/s); below this triggers wrap reduction


def model_B(abs_mlat):
    return B_COEF * np.maximum(0.0, abs_mlat - B_THRESHOLD)**2


def apply_event_balance_cap(large_corr, pho, wide, sci, lc, dtv, min_C):
    """Reduce n_wraps for rows where base would be < Sci + min_C."""
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")
    large = large_corr.copy()

    # max allowed large_corr by event-balance:
    # (PHO - large_corr)*lf/L - Wide/L >= Sci + min_C
    # → large_corr <= PHO - ((Sci + min_C)*L + Wide) / lf
    max_event = pho - ((sci.astype("float64") + min_C) * L + wide.astype("float64")) / lf
    # never below raw (can't subtract wraps we didn't add)
    large_raw_float = (large_corr - 1024 * np.round((large_corr - large_corr.astype(int)) / 1024)).astype("float64")
    # actually we just need to track raw separately; pass raw in instead. Refactor: caller passes raw too.
    return large  # placeholder — overridden below


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  rows: {len(df):,}")

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

    # Pass 1: v2 with C=150
    large_corr_v2, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0,
                                         return_confidence=True)

    # Estimate per-det C_det
    base_v2 = (pho - large_corr_v2) * lf / L - wide / L
    residual_v2 = base_v2 - sci
    is_clean_v2 = ((wide / np.maximum(pho, 1) < 0.3) & (sci > 100)
                   & np.isfinite(residual_v2) & ~np.isnan(mlat) & (abs_mlat < 5))
    C_det_map = np.full((3, 6), 120.0)
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & is_clean_v2
            if m.sum() > 100:
                C_det_map[bi, det] = float(np.mean(residual_v2[m]))
    print(f"  C_det: mean={C_det_map.mean():.1f}, range=[{C_det_map.min():.0f}, {C_det_map.max():.0f}]")

    # Per-row C = C_det + B(|mlat|)
    C_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m_dt = ((df["box"] == box) & (df["det"] == det)).values
            C_per_row[m_dt] = C_det_map[bi, det]
    C_per_row += model_B(abs_mlat_safe)

    # Pass 2: v3 with per-row C
    print("v3 unwrap with per-row C...")
    large_corr_v3, conf_v3 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv,
                                              C=C_per_row, return_confidence=True)

    # ===== v4: event-balance sanity cap =====
    print(f"v4: applying event-balance cap (require base ≥ Sci + {MIN_C_SLACK:.0f})...")
    # Max allowed large_corr from event balance: base ≥ Sci + min_C
    # → (PHO - large_corr)*lf/L - Wide/L ≥ Sci + min_C
    # → large_corr ≤ PHO - ((Sci + min_C)*L + Wide)/lf
    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf

    n_wraps_v3 = np.round((large_corr_v3 - large_raw) / 1024).astype(int)
    n_wraps_v4 = n_wraps_v3.copy()

    # If large_corr_v3 > max_large_event AND n_wraps > 0, reduce n_wraps
    # Vectorized: max_n_wraps_allowed = floor((max_large_event - large_raw) / 1024)
    n_max = np.floor((max_large_event - large_raw) / 1024.0).astype(int)
    n_max = np.maximum(n_max, 0)
    over_event = n_wraps_v3 > n_max
    n_wraps_v4 = np.where(over_event, n_max, n_wraps_v3)
    large_corr_v4 = large_raw + n_wraps_v4 * 1024.0
    print(f"  Rows reduced by event-balance cap: {over_event.sum():,}")
    diff = n_wraps_v4 - n_wraps_v3
    for k in sorted(set(diff)):
        if (diff == k).sum() > 0:
            print(f"    diff = {k:>+3}: {(diff == k).sum():>10,}")

    # Compute base + residual
    base_v4 = (pho - large_corr_v4) * lf / L - wide / L
    residual_v4 = base_v4 - sci

    # Full model residual
    B_per_row = model_B(abs_mlat_safe)
    C_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            C_det_per_row[m] = C_det_map[bi, det]
    residual_clean = residual_v4 - C_det_per_row - B_per_row

    # === Plot ===
    is_valid = np.isfinite(base_v4) & np.isfinite(residual_v4) & (sci > 0) & (base_v4 > 0)
    base_s = base_v4[is_valid]
    sci_s = sci[is_valid].astype("float32")
    resid_s = residual_v4[is_valid]
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
    ax1.plot(xx, xx, "k--", lw=1.5, label="y = x  (lower bound, should have ~no points below)")
    c_mid = (C_det_map.min() + C_det_map.max()) / 2
    ax1.plot(xx, xx + C_det_map.min(), "b-", lw=1.5, label=fr"y = x + {C_det_map.min():.0f}")
    ax1.plot(xx, xx + c_mid, "b-", lw=2.0, label=fr"y = x + {c_mid:.0f} (C_det mean)")
    ax1.plot(xx, xx + C_det_map.max() + model_B(43), "b-", lw=1.5,
              label=fr"y = x + {C_det_map.max() + model_B(43):.0f}")
    ax1.plot(xx, xx + MIN_C_SLACK, "g-", lw=1.5, label=fr"y = x + {MIN_C_SLACK:.0f}  (event-balance floor)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci_1s observed (cnt/s)")
    ax1.set_ylabel("Sci_pred_base with v4 unwrap (cnt/s)")
    ax1.set_title("log-log Sci_pred vs Sci_obs — v4 (event-balance cap)", fontsize=11)
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
    ax2.axhline(C_det_map.mean(), color="blue", lw=2.0, label=fr"C_det mean = {C_det_map.mean():.0f}")
    ax2.axhline(MIN_C_SLACK, color="green", lw=1.5, ls=":", label=fr"event-balance floor = {MIN_C_SLACK:.0f}")
    ax2.axhline(0, color="k", ls=":", lw=0.7)
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci_1s observed (cnt/s)")
    ax2.set_ylabel("residual = base − Sci_obs (cnt/s, BEFORE model)")
    ax2.set_title("residual BEFORE model (v4)", fontsize=11)
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
    ax3.set_title("residual AFTER full model (v4)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"v4 unwrap: v3 + event-balance cap (base ≥ Sci + {MIN_C_SLACK:.0f}) prevents under-y=x cloud\n"
        f"v4 reduces n_wraps for {over_event.sum():,} rows that would have over-corrected",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")

    # Statistics
    in_main = is_valid & (sci > 300) & (sci < 2000) & (np.abs(residual_v4) < 2000)
    print(f"\nv4 residual BEFORE model: median={np.median(residual_v4[in_main]):+.1f}")
    print(f"v4 residual AFTER full model: median={np.median(residual_clean[in_main]):+.1f}, "
          f"Q25={np.quantile(residual_clean[in_main], 0.25):+.1f}, Q75={np.quantile(residual_clean[in_main], 0.75):+.1f}")

    # Count below-y=x rows
    below_yx = (base_v4 < sci) & is_valid
    print(f"\nRows below y=x line (Sci_pred < Sci_obs):")
    print(f"  v4: {below_yx.sum():,} ({below_yx.sum()/is_valid.sum()*100:.3f}%)")


if __name__ == "__main__":
    main()
