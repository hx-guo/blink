#!/usr/bin/env python3
"""Real scatter plot of the v5 model on full-data, via row-group sampling.

35.8 billion rows can't be scatter-plotted directly. This reads a handful of
evenly-spaced row groups per yearly cache file (covering the whole mission),
reconstructs the v5 pipeline per row using the *per-day s_det already stored*
in v5_agg_full.npz, and scatter-plots a manageable subsample (~1M points).

Same three panels as the sample05 v5 figure, but points are drawn from the
entire 8.9-year dataset rather than one half-year.

Usage:
    python3 plot_v5_scatter_sampled.py \
        --cache-dir /Volumes/Graphite/work/blink/blink_clean_relaxed \
        --full-npz n_below_study/v5_npz/v5_agg_full.npz \
        --aacgm-grid n_below_study/aacgm_grid_2020.npz \
        --rowgroups-per-file 10 --max-points 1000000 \
        --output plots/v5_final_scatter.png
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0
NEEDED = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s", "L_cycles", "Dt", "Lat", "Lon"]


def unwrap_large_v2(pho, large, wide, sci, l_cycles, dt, C):
    pho = np.asarray(pho, dtype=np.float64)
    large = np.asarray(large, dtype=np.float64)
    wide = np.asarray(wide, dtype=np.float64)
    sci = np.asarray(sci, dtype=np.float64)
    L = np.asarray(l_cycles, dtype=np.float64) * L_CYCLES_TO_SEC
    lf = 1.0 - np.asarray(dt, dtype=np.float64) / np.asarray(l_cycles, dtype=np.float64)
    predicted = pho - (wide + (sci + C) * L) / lf
    n_raw = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    max_allowed = pho - wide
    large_corr = large + n_raw * 1024.0
    over = large_corr > max_allowed
    if over.any():
        n_max = np.maximum(np.floor((max_allowed - large) / 1024.0).astype(int), 0)
        large_corr = large + np.where(over, n_max, n_raw) * 1024.0
    return large_corr


def process_rows(df, interp, calib):
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    abs_mlat = np.abs(interp(pts))
    abs_mlat = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)
    mlat_term = np.maximum(0.0, abs_mlat - B_THRESHOLD) ** 2

    pho = df["PHO"].astype("float64").values
    large_raw = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc

    # FIXED closed-form v5t: C = s0_det*g(t)*[1+k(t)*mlat_term] + C0, zero refit
    ty = (pd.to_datetime(df["date"]).values.astype("datetime64[D]") - calib["t0"]).astype("timedelta64[D]").astype(float) / 365.25
    g = 1.0 - calib["beta"] * ty
    w = calib["w"]; kc = calib["k_coeffs"]
    k_t = (kc[0] + kc[1]*np.cos(w*ty) + kc[2]*np.sin(w*ty)
           + kc[3]*np.cos(2*w*ty) + kc[4]*np.sin(2*w*ty))
    box_idx = np.select([df["box"].values == b for b in "ABC"], [0, 1, 2], default=0)
    detid = box_idx * 6 + df["det"].values
    s_det_per_row = calib["s0_det"][detid] * g
    C_per_row = s_det_per_row * (1.0 + k_t * mlat_term) + calib["C0"]
    large_v3 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C_per_row)
    max_large_event = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n_wraps_v3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    n_max = np.maximum(np.floor((max_large_event - large_raw) / 1024.0).astype(int), 0)
    large_v5 = large_raw + np.where(n_wraps_v3 > n_max, n_max, n_wraps_v3) * 1024.0

    base_v5 = (pho - large_v5) * lf / L - wide / L
    resid_v5 = base_v5 - sci
    resid_clean = resid_v5 - C_per_row
    return sci, base_v5, resid_v5, resid_clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    p.add_argument("--calib", default="n_below_study/v5_npz/v5t_calib.npz")
    p.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    p.add_argument("--rowgroups-per-file", type=int, default=10)
    p.add_argument("--max-points", type=int, default=1_000_000)
    p.add_argument("--output", default="plots/v5t_final_scatter.png")
    args = p.parse_args()

    cz = np.load(args.calib)
    calib = {"s0_det": cz["s0_det"], "beta": float(cz["beta"]),
             "w": float(cz["w"]), "k_coeffs": cz["k_coeffs"], "C0": float(cz["C0"]),
             "t0": np.datetime64(str(cz["t0"]))}
    print(f"calib: g=1-{calib['beta']:.4f}t (linear), C0={calib['C0']:+.1f}")

    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)

    files = sorted(glob.glob(os.path.join(args.cache_dir, "clean_relaxed_20*.parquet")))
    files = [f for f in files if "sample" not in f]
    print(f"{len(files)} cache files")

    sci_all, base_all, resid_all, residc_all = [], [], [], []
    for f in files:
        pf = pq.ParquetFile(f)
        n_rg = pf.num_row_groups
        rg_idx = np.unique(np.linspace(0, n_rg - 1, args.rowgroups_per_file).astype(int))
        for rg in rg_idx:
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            sci, base_v5, resid_v5, resid_clean = process_rows(df, interp, calib)
            ok = np.isfinite(base_v5) & np.isfinite(resid_v5) & (sci > 0) & (base_v5 > 0)
            sci_all.append(sci[ok]); base_all.append(base_v5[ok])
            resid_all.append(resid_v5[ok]); residc_all.append(resid_clean[ok])
        print(f"  {os.path.basename(f)}: {sum(len(a) for a in sci_all):,} pts so far")

    sci = np.concatenate(sci_all); base = np.concatenate(base_all)
    resid = np.concatenate(resid_all); residc = np.concatenate(residc_all)
    print(f"Total collected: {len(sci):,}")

    if len(sci) > args.max_points:
        rng = np.random.RandomState(0)
        sel = rng.choice(len(sci), args.max_points, replace=False)
        sci, base, resid, residc = sci[sel], base[sel], resid[sel], residc[sel]
    print(f"Plotting {len(sci):,} points")

    LO, HI = 30.0, 10_000.0
    Y_LO, Y_HI = -500, 1500

    def density_color(x, y, xbins, ybins):
        H, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins])
        ix = np.clip(np.searchsorted(xe, x) - 1, 0, len(xe) - 2)
        iy = np.clip(np.searchsorted(ye, y) - 1, 0, len(ye) - 2)
        d = H[ix, iy].astype(float); d[d < 1] = 1
        return d

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7.5))

    # Panel 1: log-log
    xb = np.logspace(np.log10(LO), np.log10(HI), 200)
    d1 = density_color(sci, base, xb, xb)
    o = np.argsort(d1)
    ax1.scatter(sci[o], base[o], c=d1[o], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d1.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    xx = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax1.plot(xx, xx, "k--", lw=1.5, label="y = x")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci_1s observed (cnt/s)")
    ax1.set_ylabel("Sci_pred_base v5 (cnt/s)")
    ax1.set_title("log-log Sci_pred vs Sci_obs — v5, sampled scatter")
    ax1.legend(loc="lower right", fontsize=9); ax1.grid(True, alpha=0.3, which="both")

    # Panel 2: residual BEFORE
    yb = np.linspace(Y_LO, Y_HI, 200)
    in2 = (sci >= LO) & (sci <= HI) & (resid >= Y_LO) & (resid <= Y_HI)
    d2 = density_color(sci[in2], resid[in2], xb, yb)
    o2 = np.argsort(d2)
    ax2.scatter(sci[in2][o2], resid[in2][o2], c=d2[o2], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d2.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    ax2.axhline(0, color="k", ls=":", lw=0.7)
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci_1s observed (cnt/s)")
    ax2.set_ylabel("residual = base − Sci_obs (BEFORE model)")
    ax2.set_title("residual BEFORE model (v5)"); ax2.grid(True, alpha=0.3, which="both")

    # Panel 3: residual AFTER
    in3 = (sci >= LO) & (sci <= HI) & (residc >= Y_LO) & (residc <= Y_HI)
    d3 = density_color(sci[in3], residc[in3], xb, yb)
    o3 = np.argsort(d3)
    ax3.scatter(sci[in3][o3], residc[in3][o3], c=d3[o3], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d3.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    ax3.axhline(0, color="r", lw=2.0, label="zero (perfect model)")
    ax3.set_xscale("log"); ax3.set_xlim(LO, HI); ax3.set_ylim(Y_LO, Y_HI)
    ax3.set_xlabel("Sci_1s observed (cnt/s)")
    ax3.set_ylabel("residual_clean (AFTER unified model)")
    ax3.set_title("residual AFTER unified model (v5)")
    ax3.legend(loc="upper left", fontsize=10); ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"v5t fixed formula: C = s0_det·g(t)·[1+k(t)·max(0,|mlat|-20)²] + C0  (zero refit)\n"
        f"g(t)=1-{calib['beta']:.4f}t (linear) [outgassing], "
        f"k(t)=harmonic [solar cycle];  {len(sci):,} pts sampled across full 8.9yr (2017-2026)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
