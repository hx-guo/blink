#!/usr/bin/env python3
"""Two diagnostics in one figure:
  (top)    residual-AFTER (base - Sci_obs - C_v5) vs Sci, colored by |mlat| and by year
           — does the v5 model flatten the high-mlat cloud and the recent-year cloud?
  (bottom) s_det daily time series (from npz) — do per-detector params drift over 8.9 yr?
           + high-|mlat| residual-AFTER median per year (proxy for whether global k drifts)

Output: plots/diag_after_and_drift.png
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


def collect(args, interp, dates, s_det_daily, k_global, pop_median):
    date_to_i = {d: i for i, d in enumerate(dates)}
    files = sorted(glob.glob(os.path.join(args.cache_dir, "clean_relaxed_20*.parquet")))
    files = [f for f in files if "sample" not in f]
    out = {k: [] for k in ["sci", "residc", "absmlat", "year"]}
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
            residc = base_v5 - sci - C_per_row
            ok = np.isfinite(base_v5) & (sci > 0) & (base_v5 > 0)
            out["sci"].append(sci[ok]); out["residc"].append(residc[ok])
            out["absmlat"].append(abs_mlat[ok]); out["year"].append(df["date"].str[:4].astype(int).values[ok])
        print(f"  {os.path.basename(f)}: {sum(len(a) for a in out['sci']):,} pts")
    for k in out:
        out[k] = np.concatenate(out[k])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    p.add_argument("--full-npz", default="n_below_study/v5_npz/v5_agg_full.npz")
    p.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    p.add_argument("--rowgroups-per-file", type=int, default=8)
    p.add_argument("--max-points", type=int, default=400000)
    p.add_argument("--output", default="plots/diag_after_and_drift.png")
    args = p.parse_args()

    z = np.load(args.full_npz)
    dates = z["dates"]; s_det_daily = z["s_det_daily"]; k_global = float(z["k_global"])
    pop_median = float(np.nanmedian(s_det_daily))

    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)

    d = collect(args, interp, dates, s_det_daily, k_global, pop_median)
    n = len(d["sci"])
    if n > args.max_points:
        sel = np.random.RandomState(0).choice(n, args.max_points, replace=False)
        for k in d:
            d[k] = d[k][sel]
    print(f"Plotting {len(d['sci']):,} points")

    LO, HI = 30.0, 10_000.0
    Y_LO, Y_HI = -400, 800
    inr = (d["sci"] >= LO) & (d["sci"] <= HI) & (d["residc"] >= Y_LO) & (d["residc"] <= Y_HI)
    sci = d["sci"][inr]; residc = d["residc"][inr]
    absmlat = d["absmlat"][inr]; year = d["year"][inr]
    shuf = np.random.RandomState(1).permutation(len(sci))

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Panel (0,0): residual-AFTER by |mlat|
    ax = axes[0, 0]
    sc = ax.scatter(sci[shuf], residc[shuf], c=absmlat[shuf], cmap="plasma", vmin=0, vmax=60,
                    s=2, alpha=0.5, rasterized=True, edgecolor="none")
    ax.axhline(0, color="k", lw=1.5)
    ax.set_xscale("log"); ax.set_xlim(LO, HI); ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("Sci_1s observed (cnt/s)"); ax.set_ylabel("residual AFTER model (cnt/s)")
    ax.set_title("residual-AFTER colored by |mlat| — is high-mlat cloud flattened?", fontsize=12)
    fig.colorbar(sc, ax=ax).set_label("|mlat| (deg)")
    ax.grid(True, alpha=0.3, which="both")

    # Panel (0,1): residual-AFTER by year
    ax = axes[0, 1]
    sc = ax.scatter(sci[shuf], residc[shuf], c=year[shuf], cmap="turbo", s=2, alpha=0.5,
                    rasterized=True, edgecolor="none")
    ax.axhline(0, color="k", lw=1.5)
    ax.set_xscale("log"); ax.set_xlim(LO, HI); ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("Sci_1s observed (cnt/s)"); ax.set_ylabel("residual AFTER model (cnt/s)")
    ax.set_title("residual-AFTER colored by year — is recent-year cloud flattened?", fontsize=12)
    fig.colorbar(sc, ax=ax).set_label("year")
    ax.grid(True, alpha=0.3, which="both")

    # Panel (1,0): s_det daily time series
    ax = axes[1, 0]
    dt = np.array([np.datetime64(s) for s in dates])
    median_all = np.nanmedian(s_det_daily.reshape(len(dates), -1), axis=1)
    for bi in range(3):
        for det in range(6):
            if (bi, det) == (1, 2):  # B-2 highlighted separately
                continue
            ax.plot(dt, s_det_daily[:, bi, det], lw=0.3, alpha=0.25, color="gray")
    ax.plot(dt, s_det_daily[:, 1, 2], lw=1.0, color="red", label="B-2 (anomalous)")
    ax.plot(dt, median_all, lw=1.5, color="black", label="median of 18 dets")
    ax.set_xlabel("date"); ax.set_ylabel("s_det per-day (cnt/s)")
    ax.set_title("s_det daily time series — PMT outgassing drift?", fontsize=12)
    ax.set_ylim(50, 230)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (1,1): high-|mlat| residual-AFTER median per year (k drift proxy)
    ax = axes[1, 1]
    hi_mlat = absmlat >= 35
    years_sorted = np.array(sorted(np.unique(year)))
    med_hi, med_lo = [], []
    for y in years_sorted:
        m_hi = hi_mlat & (year == y)
        m_lo = (absmlat < 10) & (year == y)
        med_hi.append(np.median(residc[m_hi]) if m_hi.sum() > 50 else np.nan)
        med_lo.append(np.median(residc[m_lo]) if m_lo.sum() > 50 else np.nan)
    ax.plot(years_sorted, med_hi, "o-", color="darkorange", label="|mlat|≥35° (tests k)")
    ax.plot(years_sorted, med_lo, "s-", color="navy", label="|mlat|<10° (tests s_det)")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("year"); ax.set_ylabel("median residual-AFTER (cnt/s)")
    ax.set_title("residual-AFTER median per year — param drift check\n(flat at 0 = no drift)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"v5 residual-AFTER + parameter drift  (k={k_global:.5f} global)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")

    # Print numeric drift summary
    print("\n=== s_det drift (per-det, first 6 months vs last 6 months median) ===")
    half = 183
    for bi, box in enumerate("ABC"):
        for det in range(6):
            early = np.nanmedian(s_det_daily[:half, bi, det])
            late = np.nanmedian(s_det_daily[-half:, bi, det])
            print(f"  {box}-{det}: {early:6.1f} → {late:6.1f}  (Δ={late-early:+.1f})")


if __name__ == "__main__":
    main()
