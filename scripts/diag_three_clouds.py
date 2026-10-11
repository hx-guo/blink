#!/usr/bin/env python3
"""Diagnose the three clouds in the full-data v5 scatter.

Samples row groups across all years (same as plot_v5_scatter_sampled.py) but
records per-point year / detector / |mlat| / s_det, then colors the
residual-BEFORE panel (base - Sci_obs vs Sci_obs) by each dimension. Whichever
dimension separates the clouds is the cause.

Output: plots/diag_three_clouds.png
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
    pho = np.asarray(pho, float); large = np.asarray(large, float)
    wide = np.asarray(wide, float); sci = np.asarray(sci, float)
    L = np.asarray(l_cycles, float) * L_CYCLES_TO_SEC
    lf = 1.0 - np.asarray(dt, float) / np.asarray(l_cycles, float)
    predicted = pho - (wide + (sci + C) * L) / lf
    n_raw = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    max_allowed = pho - wide
    lc = large + n_raw * 1024.0
    over = lc > max_allowed
    if over.any():
        n_max = np.maximum(np.floor((max_allowed - large) / 1024.0).astype(int), 0)
        lc = large + np.where(over, n_max, n_raw) * 1024.0
    return lc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    p.add_argument("--full-npz", default="n_below_study/v5_npz/v5_agg_full.npz")
    p.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    p.add_argument("--rowgroups-per-file", type=int, default=8)
    p.add_argument("--max-points", type=int, default=400000)
    p.add_argument("--output", default="plots/diag_three_clouds.png")
    args = p.parse_args()

    z = np.load(args.full_npz)
    dates = z["dates"]; s_det_daily = z["s_det_daily"]; k_global = float(z["k_global"])
    pop_median = float(np.nanmedian(s_det_daily))
    date_to_i = {d: i for i, d in enumerate(dates)}

    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)

    files = sorted(glob.glob(os.path.join(args.cache_dir, "clean_relaxed_20*.parquet")))
    files = [f for f in files if "sample" not in f]

    cols = {k: [] for k in ["sci", "resid", "year", "detid", "absmlat", "sdet"]}
    for f in files:
        pf = pq.ParquetFile(f)
        n_rg = pf.num_row_groups
        for rg in np.unique(np.linspace(0, n_rg - 1, args.rowgroups_per_file).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            abs_mlat = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
            abs_mlat = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)
            mlat_term = np.maximum(0.0, abs_mlat - B_THRESHOLD) ** 2

            s_det_per_row = np.full(len(df), pop_median)
            for date, idx in df.groupby("date").groups.items():
                di = date_to_i.get(date)
                if di is None:
                    continue
                sd = s_det_daily[di]
                sub = df.loc[idx]
                for bi, box in enumerate("ABC"):
                    for det in range(6):
                        m = (sub["box"].values == box) & (sub["det"].values == det)
                        if np.isfinite(sd[bi, det]):
                            s_det_per_row[np.asarray(idx)[m]] = sd[bi, det]

            pho = df["PHO"].astype(float).values; large_raw = df["Large"].astype(float).values
            wide = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
            lc = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
            L = lc * L_CYCLES_TO_SEC; lf = 1.0 - dtv / lc

            C_per_row = s_det_per_row * (1.0 + k_global * mlat_term)
            large_v3 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C_per_row)
            max_le = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
            n3 = np.round((large_v3 - large_raw) / 1024).astype(int)
            nmax = np.maximum(np.floor((max_le - large_raw) / 1024.0).astype(int), 0)
            large_v5 = large_raw + np.where(n3 > nmax, nmax, n3) * 1024.0
            base_v5 = (pho - large_v5) * lf / L - wide / L
            resid = base_v5 - sci

            ok = np.isfinite(base_v5) & np.isfinite(resid) & (sci > 0) & (base_v5 > 0)
            box_idx = np.select([df["box"].values == b for b in "ABC"], [0, 1, 2], default=0)
            detid = box_idx * 6 + df["det"].values
            yr = df["date"].str[:4].astype(int).values
            cols["sci"].append(sci[ok]); cols["resid"].append(resid[ok])
            cols["year"].append(yr[ok]); cols["detid"].append(detid[ok])
            cols["absmlat"].append(abs_mlat[ok]); cols["sdet"].append(s_det_per_row[ok])
        print(f"  {os.path.basename(f)}: {sum(len(a) for a in cols['sci']):,} pts")

    for k in cols:
        cols[k] = np.concatenate(cols[k])
    n = len(cols["sci"])
    if n > args.max_points:
        sel = np.random.RandomState(0).choice(n, args.max_points, replace=False)
        for k in cols:
            cols[k] = cols[k][sel]
    print(f"Plotting {len(cols['sci']):,} points")

    LO, HI = 30.0, 10_000.0
    Y_LO, Y_HI = -300, 1200
    inr = (cols["sci"] >= LO) & (cols["sci"] <= HI) & (cols["resid"] >= Y_LO) & (cols["resid"] <= Y_HI)
    sci = cols["sci"][inr]; resid = cols["resid"][inr]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    specs = [
        ("year", cols["year"][inr], "year", "turbo", None, None),
        ("|mlat|", cols["absmlat"][inr], "|mlat| (deg)", "plasma", 0, 60),
        ("detector (box*6+det)", cols["detid"][inr], "det id 0-17 (A0..C5)", "tab20", 0, 17),
        ("s_det (per-day)", cols["sdet"][inr], "s_det (cnt/s)", "viridis", 80, 210),
    ]
    for ax, (title, c, clabel, cmap, vmin, vmax) in zip(axes.flat, specs):
        order = np.argsort(np.random.RandomState(1).rand(len(sci)))  # shuffle for fair overdraw
        sca = ax.scatter(sci[order], resid[order], c=c[order], cmap=cmap, s=2, alpha=0.5,
                         vmin=vmin, vmax=vmax, rasterized=True, edgecolor="none")
        ax.set_xscale("log"); ax.set_xlim(LO, HI); ax.set_ylim(Y_LO, Y_HI)
        ax.axhline(0, color="k", ls=":", lw=0.7)
        ax.set_xlabel("Sci_1s observed (cnt/s)")
        ax.set_ylabel("residual = base − Sci_obs (BEFORE model)")
        ax.set_title(f"colored by {title}", fontsize=12)
        cb = fig.colorbar(sca, ax=ax); cb.set_label(clabel)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Three-cloud diagnosis: residual BEFORE model, colored 4 ways",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
