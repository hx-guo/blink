#!/usr/bin/env python3
"""v5: Unified per-det sensitivity model.

Sci_1s = (PHO - Large_v4)·(1-dt)/L - Wide/L - s_det·[1 + k·max(0,|mlat|-20)²]

Where:
  - s_det (box, det): 18 per-det electronics sensitivity (== old C_det at equator)
  - k: 1 global parameter, ≈ 0.002, replaces both 0.26 of B and the
    separation between C_det and B(|mlat|)
  - Threshold 20° kept fixed (CR cutoff rigidity)

Total: 18 + 1 = 19 parameters.

Algorithm:
  1. v2 unwrap with C=150 → estimate s_det from |mlat|<5°
  2. Fit k from clean rows: k ≈ median((resid_v2 - s_det) / (s_det·(|mlat|-20)²))
  3. v3 unwrap with C_per_row = s_det · [1 + k·(|mlat|-20)²]
  4. v5 = v3 + event-balance cap
  5. Compute residual_clean and plot

Output: plots/C_vs_beta_v5_unified.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_v5_unified.png")
L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0


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
    mlat_term = np.maximum(0.0, abs_mlat_safe - B_THRESHOLD)**2

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

    # --- Pass 1: v2 with C=150 to get rough Large_corr ---
    large_v2, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=150.0,
                                    return_confidence=True)
    base_v2 = (pho - large_v2) * lf / L - wide / L
    resid_v2 = base_v2 - sci

    # --- Step 1: s_det from |mlat|<5° (same as old C_det) ---
    is_clean_base = ((wide / np.maximum(pho, 1) < 0.3) & (sci > 100)
                     & np.isfinite(resid_v2) & ~np.isnan(mlat))
    is_clean_eq = is_clean_base & (abs_mlat < 5)
    s_det_map = np.full((3, 6), 120.0)
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = ((box_arr == box) & (det_arr == det)) & is_clean_eq
            if m.sum() > 100:
                s_det_map[bi, det] = float(np.mean(resid_v2[m]))
    print(f"\n  s_det matrix:")
    print(f"       det0  det1  det2  det3  det4  det5")
    for bi, box in enumerate("ABC"):
        row = "  ".join(f"{s_det_map[bi,d]:>4.0f}" for d in range(6))
        print(f"    {box}: {row}")

    s_det_per_row = np.zeros(len(df))
    for bi, box in enumerate("ABC"):
        for det in range(6):
            m = (box_arr == box) & (det_arr == det)
            s_det_per_row[m] = s_det_map[bi, det]

    # --- Step 2: fit global k ---
    # B = resid_v2 - s_det at high |mlat|, then k = B / (s_det · (|mlat|-20)²)
    # Exclude B-2 (blind det, anomalous), use only |mlat| > 25° to avoid noise floor
    is_for_fit = (is_clean_base & (abs_mlat >= 25) & ~((box_arr == "B") & (det_arr == 2)))
    B_observed = resid_v2 - s_det_per_row
    # Per-det k estimate: B / (s_det · (|mlat|-20)²)
    # Use bin medians at several |mlat| to get robust fit
    mlat_bins = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
    k_estimates = []
    for lo, hi in mlat_bins:
        m = is_for_fit & (abs_mlat >= lo) & (abs_mlat < hi)
        if m.sum() < 1000:
            continue
        # k = B / (s_det · (|mlat|-20)²)
        mid = (lo + hi) / 2
        k_per_row = B_observed[m] / (s_det_per_row[m] * (mid - B_THRESHOLD)**2)
        k_med = float(np.median(k_per_row))
        k_estimates.append((mid, k_med, m.sum()))
        print(f"  |mlat| {lo}-{hi}: k_median={k_med:.5f}, N={m.sum():,}")

    # Take median of all binned k estimates
    k_global = float(np.median([k for _, k, _ in k_estimates]))
    print(f"\n  k_global (median over bins) = {k_global:.5f}")

    # --- Pass 2: v3 unwrap with per-row C from unified model ---
    C_per_row_v3 = s_det_per_row * (1 + k_global * mlat_term)
    print(f"\n  Per-row unified C: median={np.median(C_per_row_v3):.0f}, "
          f"range=[{np.min(C_per_row_v3):.0f}, {np.max(C_per_row_v3):.0f}]")

    large_v3, _ = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv,
                                    C=C_per_row_v3, return_confidence=True)

    # --- v5: v3 + event-balance cap ---
    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n_wraps_v3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    n_max = np.maximum(np.floor((max_large_event - large_raw) / 1024.0).astype(int), 0)
    n_wraps_v5 = np.where(n_wraps_v3 > n_max, n_max, n_wraps_v3)
    large_v5 = large_raw + n_wraps_v5 * 1024.0
    n_capped = (n_wraps_v3 > n_max).sum()
    print(f"\n  Event-balance cap reduced n_wraps for {n_capped:,} rows")

    base_v5 = (pho - large_v5) * lf / L - wide / L
    residual_v5 = base_v5 - sci
    residual_clean = residual_v5 - C_per_row_v3

    # --- Plot ---
    is_valid = np.isfinite(base_v5) & np.isfinite(residual_v5) & (sci > 0) & (base_v5 > 0)
    base_s = base_v5[is_valid]
    sci_s = sci[is_valid].astype("float32")
    resid_s = residual_v5[is_valid]
    resid_clean_s = residual_clean[is_valid]

    N = min(300_000, len(base_s))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base_s), N, replace=False)
    base_p = base_s[idx]; sci_p = sci_s[idx]
    resid_p = resid_s[idx]; resid_clean_p = resid_clean_s[idx]

    LO, HI = 30.0, 10_000.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7.5))

    # Panel 1: log-log
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
    ax1.plot(xx, xx + 88, "b-", lw=1.0, alpha=0.6, label=fr"y = x + 88 (s_det min)")
    ax1.plot(xx, xx + 124, "b-", lw=2.0, label=fr"y = x + 124 (s_det mean)")
    ax1.plot(xx, xx + 88 + k_global*88*(43-20)**2, "m-", lw=1.0, alpha=0.7,
              label=fr"y = x + s_det·(1+k·(43-20)²), min,|mlat|=43")
    ax1.plot(xx, xx + 200 + k_global*200*(43-20)**2, "m-", lw=1.0, alpha=0.7,
              label=fr"y = x + s_det·(1+k·(43-20)²), max,|mlat|=43")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci_1s observed (cnt/s)")
    ax1.set_ylabel("Sci_pred_base with v5 unified (cnt/s)")
    ax1.set_title(f"log-log Sci_pred vs Sci_obs — v5 unified (k={k_global:.5f})", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel 2: residual BEFORE model
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
    ax2.axhline(np.median(C_per_row_v3), color="blue", lw=2.0,
                 label=fr"unified C median = {np.median(C_per_row_v3):.0f}")
    ax2.axhline(0, color="k", ls=":", lw=0.7)
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci_1s observed (cnt/s)")
    ax2.set_ylabel("residual = base − Sci_obs (cnt/s, BEFORE model)")
    ax2.set_title("residual BEFORE model (v5)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3: residual AFTER unified model
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
    ax3.set_ylabel("residual_clean (cnt/s, AFTER unified model)")
    ax3.set_title(f"residual AFTER unified model (v5)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"v5: unified per-det sensitivity   s_det × [1 + k·max(0,|mlat|-20)²]\n"
        f"19 params (18 s_det + 1 k);  k={k_global:.5f};  event-balance cap on {n_capped:,} rows",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")

    # Stats
    in_main = is_valid & (sci > 300) & (sci < 2000) & (np.abs(residual_v5) < 2000)
    print(f"\nv5 residual BEFORE model: median={np.median(residual_v5[in_main]):+.1f}")
    print(f"v5 residual AFTER unified model: median={np.median(residual_clean[in_main]):+.1f}, "
          f"Q25={np.quantile(residual_clean[in_main], 0.25):+.1f}, "
          f"Q75={np.quantile(residual_clean[in_main], 0.75):+.1f}")

    below_yx = (base_v5 < sci) & is_valid
    print(f"\nRows below y=x: {below_yx.sum():,} ({below_yx.sum()/is_valid.sum()*100:.3f}%)")

    # Check the previous "blob" region: Sci 800-2500, resid_clean -300 to -50
    is_blob = is_valid & (sci >= 800) & (sci <= 2500) & (residual_clean >= -300) & (residual_clean <= -50)
    is_main_check = is_valid & (sci >= 800) & (sci <= 2500) & (residual_clean >= -50) & (residual_clean <= 100)
    print(f"\nBlob check (Sci 800-2500, resid -300 to -50):")
    print(f"  v5: {is_blob.sum():,} rows, ratio to main = {is_blob.sum()/is_main_check.sum():.3f}")
    print(f"  (v4 had: ratio = 0.128)")

    # Per-det blob fraction
    print("\n  v5 blob/main ratio per (box, det):")
    print("     det0    det1    det2    det3    det4    det5")
    for bi, box in enumerate("ABC"):
        row = []
        for det in range(6):
            m_det = (box_arr == box) & (det_arr == det)
            b = (m_det & is_blob).sum()
            m = (m_det & is_main_check).sum()
            r = b / m if m > 0 else 0
            row.append(f"{r:>6.3f}")
        print(f"  {box}: " + "  ".join(row))


if __name__ == "__main__":
    main()
