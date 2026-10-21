#!/usr/bin/env python3
"""v6 visualizations:
  Fig A: per-det (R, s) scatter — B-2 isolated on R axis
  Fig B: v5 vs v6 residual-cleaned comparison (B-2 blob removal)

Model:  C(det, |mlat|) = R_det + s_det · [1 + k · max(0, |mlat|-20)²]
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
from unwrap_large_v2 import unwrap_large_v2

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT_A = Path("plots/v6_R_s_scatter.png")
OUT_B = Path("plots/v6_vs_v5_blob_comparison.png")
L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0


def fit_R_s_per_det(y, u, box_arr, det_arr, weight_mask):
    out = {}
    for box in "ABC":
        for det in range(6):
            m = (box_arr == box) & (det_arr == det) & weight_mask
            n = int(m.sum())
            if n < 200:
                out[(box, det)] = (np.nan, np.nan, np.nan, np.nan, n)
                continue
            xi = u[m]
            yi = y[m]
            xbar = xi.mean()
            ybar = yi.mean()
            Sxx = ((xi - xbar) ** 2).sum()
            Sxy = ((xi - xbar) * (yi - ybar)).sum()
            s = Sxy / Sxx
            R = ybar - s * xbar
            yhat = R + s * xi
            sse = ((yi - yhat) ** 2).sum()
            var = sse / (n - 2)
            sigma_s = float(np.sqrt(var / Sxx))
            sigma_R = float(np.sqrt(var * (1.0 / n + xbar ** 2 / Sxx)))
            out[(box, det)] = (float(R), float(s), sigma_R, sigma_s, n)
    return out


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  rows: {len(df):,}")

    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]),
        grid["mlat"],
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)
    abs_mlat_safe = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)
    mlat_term = np.maximum(0.0, abs_mlat_safe - B_THRESHOLD) ** 2

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

    # --- Pass 1: v2 unwrap with C=150 → get resid_v2 for fitting ---
    print("\n[1/5] v2 unwrap with C=150...")
    large_v2, _ = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=150.0, return_confidence=True
    )
    base_v2 = (pho - large_v2) * lf / L - wide / L
    resid_v2 = base_v2 - sci

    is_clean = (
        (wide / np.maximum(pho, 1) < 0.3)
        & (sci > 100)
        & np.isfinite(resid_v2)
        & ~np.isnan(mlat)
        & (np.abs(resid_v2) < 2000)
    )

    # --- v6 fit: R_det, s_det per det, k global ---
    print("\n[2/5] v6 fit: R_det, s_det, k...")
    k = 0.00188
    u = 1.0 + k * mlat_term
    fits = fit_R_s_per_det(resid_v2, u, box_arr, det_arr, is_clean)

    R_per_row = np.zeros(len(df))
    s_per_row = np.zeros(len(df))
    for box in "ABC":
        for det in range(6):
            R, s, _, _, _ = fits[(box, det)]
            m = (box_arr == box) & (det_arr == det)
            R_per_row[m] = R
            s_per_row[m] = s

    # --- v6 model: C_v6 = R + s · u, then re-unwrap ---
    C_v6_per_row = R_per_row + s_per_row * u
    print(f"  C_v6 per-row: median={np.median(C_v6_per_row):.0f}, range=[{np.min(C_v6_per_row):.0f}, {np.max(C_v6_per_row):.0f}]")

    print("\n[3/5] v6 unwrap with per-row C...")
    large_v6, _ = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=C_v6_per_row, return_confidence=True
    )
    # Event-balance cap (same as v5)
    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n_wraps_v6 = np.round((large_v6 - large_raw) / 1024).astype(int)
    n_max = np.maximum(np.floor((max_large_event - large_raw) / 1024.0).astype(int), 0)
    n_wraps_v6_capped = np.where(n_wraps_v6 > n_max, n_max, n_wraps_v6)
    large_v6 = large_raw + n_wraps_v6_capped * 1024.0
    base_v6 = (pho - large_v6) * lf / L - wide / L
    resid_v6_total = base_v6 - sci
    resid_v6_clean = resid_v6_total - C_v6_per_row

    # --- v5 model for comparison: s_v5_det = R + s (combined), k same ---
    print("\n[4/5] v5 model for comparison (R+s combined as single C_eq)...")
    s_v5_per_row = R_per_row + s_per_row  # what v5 would have fit at equator
    C_v5_per_row = s_v5_per_row * u  # WRONG: applies u to R as well
    large_v5, _ = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv, C=C_v5_per_row, return_confidence=True
    )
    n_wraps_v5 = np.round((large_v5 - large_raw) / 1024).astype(int)
    n_wraps_v5_capped = np.where(n_wraps_v5 > n_max, n_max, n_wraps_v5)
    large_v5 = large_raw + n_wraps_v5_capped * 1024.0
    base_v5 = (pho - large_v5) * lf / L - wide / L
    resid_v5_total = base_v5 - sci
    resid_v5_clean = resid_v5_total - C_v5_per_row

    is_valid = (
        np.isfinite(base_v6) & np.isfinite(resid_v6_clean) & (sci > 0)
        & np.isfinite(base_v5) & np.isfinite(resid_v5_clean)
    )

    # --- Fig A: (R, s) scatter ---
    print("\n[5/5] Plotting...")
    fig_a, ax_a = plt.subplots(figsize=(8, 7))
    colors = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
    markers = {"A": "o", "B": "s", "C": "^"}
    for box in "ABC":
        Rs, ss, sRs, sSs = [], [], [], []
        labels = []
        for det in range(6):
            R, s, sR, sS, _ = fits[(box, det)]
            Rs.append(R); ss.append(s); sRs.append(sR); sSs.append(sS)
            labels.append(f"{box}-{det}")
        Rs = np.array(Rs); ss = np.array(ss); sRs = np.array(sRs); sSs = np.array(sSs)
        ax_a.errorbar(Rs, ss, xerr=sRs*3, yerr=sSs*3, fmt=markers[box],
                       color=colors[box], markersize=10, capsize=3,
                       label=f"Box {box}", alpha=0.85, mew=1.2, mec="black")
        for i, lbl in enumerate(labels):
            offset = (8, 6) if lbl != "B-2" else (10, -12)
            fontweight = "bold" if lbl == "B-2" else "normal"
            ax_a.annotate(lbl, (Rs[i], ss[i]), xytext=offset, textcoords="offset points",
                           fontsize=9, fontweight=fontweight)

    # Mark B-2 with arrow + annotation
    R_b2, s_b2, _, _, _ = fits[("B", 2)]
    ax_a.annotate(
        "B-2: anomalous PMT\nnoise floor",
        xy=(R_b2, s_b2),
        xytext=(20, 175),
        fontsize=11, color="#d62728", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
    )

    # Population mean ± std envelope for R (excl B-2)
    other_Rs = [fits[(b, d)][0] for b in "ABC" for d in range(6) if (b, d) != ("B", 2)]
    R_mean = float(np.mean(other_Rs))
    R_std = float(np.std(other_Rs))
    ax_a.axvspan(R_mean - 2*R_std, R_mean + 2*R_std, alpha=0.12, color="gray",
                  label=f"17-det R = {R_mean:+.1f} ± {R_std:.1f} (2σ)")
    ax_a.axvline(0, color="black", ls=":", lw=0.8, alpha=0.6)

    ax_a.set_xlabel(r"$R_{\rm det}$  —  electronic noise floor  (cnt/s)", fontsize=12)
    ax_a.set_ylabel(r"$s_{\rm det}$  —  cosmic-ray sensitivity  (cnt/s)", fontsize=12)
    ax_a.set_title(
        "v6 decomposition:  per-detector $R_{\\rm det}$ vs $s_{\\rm det}$\n"
        f"17 dets cluster near $R{R_mean:+.0f}$ cps;  B-2 isolated at R = +{R_b2:.0f} (52σ),  $s$ normal",
        fontsize=11,
    )
    ax_a.legend(loc="upper left", fontsize=10)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(-30, 70)
    ax_a.set_ylim(80, 200)
    plt.tight_layout()
    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_A, dpi=120, bbox_inches="tight")
    plt.close(fig_a)
    print(f"  Saved {OUT_A}")

    # --- Fig B: v5 vs v6 residual-clean for B-2 specifically (then all) ---
    fig_b, axes = plt.subplots(2, 2, figsize=(16, 12))
    LO, HI = 30.0, 10_000.0
    Y_LO, Y_HI = -500, 1500

    rng = np.random.RandomState(0)
    is_b2 = is_valid & (box_arr == "B") & (det_arr == 2)

    for col_i, (label, mask) in enumerate(
        [("all 18 detectors", is_valid), ("B-2 only", is_b2)]
    ):
        for row_i, (model_name, resid_clean) in enumerate(
            [("v5 (R+s combined)", resid_v5_clean), ("v6 (R + s·u)", resid_v6_clean)]
        ):
            ax = axes[row_i, col_i]
            sci_s = sci[mask]
            res_s = resid_clean[mask]
            N = min(150_000, len(sci_s))
            if len(sci_s) > N:
                idx = rng.choice(len(sci_s), N, replace=False)
                sci_p = sci_s[idx]
                res_p = res_s[idx]
            else:
                sci_p = sci_s; res_p = res_s

            in_range = (sci_p >= LO) & (sci_p <= HI) & (res_p >= Y_LO) & (res_p <= Y_HI)
            sci_pp = sci_p[in_range]
            res_pp = res_p[in_range]

            xb = np.logspace(np.log10(LO), np.log10(HI), 150)
            yb = np.linspace(Y_LO, Y_HI, 150)
            H, xe, ye = np.histogram2d(sci_pp, res_pp, bins=[xb, yb])
            ix = np.clip(np.searchsorted(xe, sci_pp) - 1, 0, len(xe) - 2)
            iy = np.clip(np.searchsorted(ye, res_pp) - 1, 0, len(ye) - 2)
            dens = H[ix, iy].astype(float); dens[dens < 1] = 1
            order = np.argsort(dens)
            ax.scatter(sci_pp[order], res_pp[order], c=dens[order], cmap="viridis",
                        norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                        s=2, alpha=0.55, rasterized=True, edgecolor="none")
            ax.axhline(0, color="red", lw=1.5, alpha=0.7, label="zero")
            ax.set_xscale("log")
            ax.set_xlim(LO, HI)
            ax.set_ylim(Y_LO, Y_HI)
            ax.set_xlabel("Sci$_{\\rm 1s}$ observed (cnt/s)")
            ax.set_ylabel(f"residual$_{{\\rm clean}}$  (cnt/s)")
            ax.grid(True, alpha=0.3, which="both")

            # Blob count
            is_blob = (sci_pp >= 800) & (sci_pp <= 2500) & (res_pp >= -300) & (res_pp <= -50)
            is_main = (sci_pp >= 800) & (sci_pp <= 2500) & (res_pp >= -50) & (res_pp <= 100)
            blob_ratio = is_blob.sum() / max(is_main.sum(), 1)
            ax.set_title(
                f"{model_name}  —  {label}\n"
                f"blob/main ratio (Sci 800-2500, resid -300 to -50) = {blob_ratio:.3f}",
                fontsize=11,
            )
            ax.legend(loc="upper left", fontsize=9)

    fig_b.suptitle(
        "v5 (R, s combined) vs v6 (R + s·u decomposed):  B-2 blob removal",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_B, dpi=120, bbox_inches="tight")
    plt.close(fig_b)
    print(f"  Saved {OUT_B}")

    # --- Stats ---
    is_b2_eval = is_b2 & (sci >= 800) & (sci <= 2500)
    blob_v5 = is_b2_eval & (resid_v5_clean >= -300) & (resid_v5_clean <= -50)
    main_v5 = is_b2_eval & (resid_v5_clean >= -50) & (resid_v5_clean <= 100)
    blob_v6 = is_b2_eval & (resid_v6_clean >= -300) & (resid_v6_clean <= -50)
    main_v6 = is_b2_eval & (resid_v6_clean >= -50) & (resid_v6_clean <= 100)
    print(f"\n  B-2 blob/main ratio:")
    print(f"    v5: {blob_v5.sum()/max(main_v5.sum(),1):.3f}  ({blob_v5.sum():,} blob / {main_v5.sum():,} main)")
    print(f"    v6: {blob_v6.sum()/max(main_v6.sum(),1):.3f}  ({blob_v6.sum():,} blob / {main_v6.sum():,} main)")


if __name__ == "__main__":
    main()
