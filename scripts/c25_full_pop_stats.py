#!/usr/bin/env python3
"""Streaming C25 self-validation on the full HXMT/HE cache.

Computes residual statistics (mean, std, median, P5, P95) over every row
in the locked 2017-06-22 to 2026-05-31 mission window, without downsampling.

For the 2D-histogram-based figure panels, accumulates the histograms
incrementally so the script never holds more than one row group in memory.
Output:
  - stdout: full-population residual stats
  - plots/c25_conservation_fullpop.png: regenerated 4-panel figure
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_c25_conservation import (  # noqa
    NEEDED, BOX_ID, L_CYCLES_TO_SEC,
    sigm, unwrap_v2, C_model_c25, process_rows,
)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# Histogram bin edges (must match plot_c25_conservation.py panels)
LO, HI = 30.0, 10_000.0
XBIN = np.logspace(np.log10(LO), np.log10(HI), 200)
RES_LO, RES_HI = -400, 400
YBIN_RES = np.linspace(RES_LO, RES_HI, 200)
FRAC_LO, FRAC_HI = -40, 40
YBIN_FRAC = np.linspace(FRAC_LO, FRAC_HI, 200)


class TDigest:
    """Minimal streaming quantile via fixed grid histogram, good enough for
    paper-scale (resid range bounded; bin density tracks the distribution)."""

    def __init__(self, lo: float, hi: float, nbins: int = 4001):
        self.lo, self.hi, self.nbins = lo, hi, nbins
        self.edges = np.linspace(lo, hi, nbins + 1)
        self.counts = np.zeros(nbins, dtype=np.int64)
        self.under = 0
        self.over = 0

    def update(self, values: np.ndarray) -> None:
        below = values < self.lo
        above = values > self.hi
        self.under += int(below.sum())
        self.over += int(above.sum())
        m = ~(below | above)
        if m.any():
            h, _ = np.histogram(values[m], bins=self.edges)
            self.counts += h

    def quantile(self, q: float) -> float:
        total = self.under + int(self.counts.sum()) + self.over
        target = q * total
        cum = self.under
        if cum >= target:
            return float(self.lo)
        cumulative = cum + np.cumsum(self.counts)
        idx = int(np.searchsorted(cumulative, target))
        if idx >= self.nbins:
            return float(self.hi)
        # Linear interp inside the chosen bin
        prev = cum if idx == 0 else cumulative[idx - 1]
        bin_count = self.counts[idx]
        if bin_count == 0:
            return float((self.edges[idx] + self.edges[idx + 1]) * 0.5)
        frac = (target - prev) / bin_count
        return float(self.edges[idx] + frac * (self.edges[idx + 1] - self.edges[idx]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    ap.add_argument("--c25-json", default="/tmp/per_det_25param.json")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    ap.add_argument("--output", default="plots/c25_conservation_fullpop.png")
    ap.add_argument("--stats-out", default="plots/c25_fullpop_stats.json")
    ap.add_argument("--window-end", default="2026-05-31",
                    help="Hard cap on date (inclusive). Default 2026-05-31.")
    args = ap.parse_args()

    P = json.loads(Path(args.c25_json).read_text())
    print(f"C25 params: alpha_m={P['alpha']:.3f}, mu_m={P['mu_m']:.2f}, "
          f"alpha_t={P['amp0']:.3f}, mu_t={P['mu_t']:.2f}, C_0={P['C0']:+.1f}")

    t_ref = np.datetime64("2017-06-22")
    win_end = np.datetime64(args.window_end)

    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan,
    )

    files = [
        f for f in sorted(glob.glob(os.path.join(args.cache_dir, "clean_relaxed_20*.parquet")))
        if "sample" not in f
    ]
    print(f"Streaming over {len(files)} yearly parquet files...")

    n = 0
    sum_r = 0.0
    sum_r2 = 0.0
    resid_digest = TDigest(-500.0, 500.0, 5001)
    H1 = np.zeros((len(XBIN) - 1, len(XBIN) - 1), dtype=np.int64)
    H2 = np.zeros((len(XBIN) - 1, len(YBIN_RES) - 1), dtype=np.int64)
    H3 = np.zeros((len(XBIN) - 1, len(YBIN_FRAC) - 1), dtype=np.int64)
    # Panel 4 needs conditional quantiles per sci bin
    nbin4 = len(XBIN) - 1
    digests4 = [TDigest(-500.0, 500.0, 2001) for _ in range(nbin4)]

    for f in files:
        pf = pq.ParquetFile(f)
        n_rg = pf.num_row_groups
        for rg in range(n_rg):
            df = pf.read_row_group(rg, columns=NEEDED).to_pandas()
            # Window cap on date
            dates = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
            df = df.loc[dates <= win_end]
            if df.empty:
                continue
            sci, rec = process_rows(df, interp, P, t_ref)
            ok = np.isfinite(sci) & np.isfinite(rec) & (sci > 0)
            sci = sci[ok]
            rec = rec[ok]
            if sci.size == 0:
                continue
            resid = rec - sci

            n += sci.size
            sum_r += float(resid.sum())
            sum_r2 += float((resid * resid).sum())
            resid_digest.update(resid)

            H1 += np.histogram2d(sci, rec, bins=[XBIN, XBIN])[0].astype(np.int64)
            H2 += np.histogram2d(sci, resid, bins=[XBIN, YBIN_RES])[0].astype(np.int64)
            frac = (sci - rec) / sci * 100.0
            H3 += np.histogram2d(sci, frac, bins=[XBIN, YBIN_FRAC])[0].astype(np.int64)

            # Conditional digests
            sci_bin = np.searchsorted(XBIN, sci) - 1
            sci_bin = np.clip(sci_bin, 0, nbin4 - 1)
            for b in range(nbin4):
                m_b = sci_bin == b
                if m_b.any():
                    digests4[b].update(resid[m_b])
        print(f"  {os.path.basename(f)}: cumulative N={n:,}")

    mean_r = sum_r / n
    var_r = sum_r2 / n - mean_r * mean_r
    std_r = float(np.sqrt(max(var_r, 0)))
    quantile_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles = {f"P{int(q*100):02d}": resid_digest.quantile(q) for q in quantile_levels}
    p25 = quantiles["P25"]
    p50 = quantiles["P50"]
    p75 = quantiles["P75"]
    p5 = quantiles["P05"]
    p95 = quantiles["P95"]
    iqr = p75 - p25
    sigma_iqr = iqr / 1.349  # Gaussian-equivalent robust sigma

    print()
    print("=== Full-population C25 residual statistics ===")
    print(f"  N        = {n:,}")
    print(f"  mean     = {mean_r:+.3f} cnt/s")
    print(f"  median   = {p50:+.3f} cnt/s")
    print(f"  std      = {std_r:.3f} cnt/s  (tail-inflated)")
    for q in quantile_levels:
        key = f"P{int(q*100):02d}"
        print(f"  {key:<8} = {quantiles[key]:+.3f} cnt/s")
    print(f"  IQR (P75-P25)        = {iqr:.3f} cnt/s")
    print(f"  robust sigma IQR/1.349 = {sigma_iqr:.3f} cnt/s")

    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps({
        "N": n,
        "mean": mean_r,
        "median": p50,
        "std": std_r,
        "P05": p5, "P95": p95,
        "quantiles": quantiles,
        "IQR": iqr,
        "sigma_iqr": sigma_iqr,
        "window_start": "2017-06-22",
        "window_end": args.window_end,
    }, indent=2))
    print(f"Stats JSON: {args.stats_out}")

    # Persist the residual digest so future post-processing can query any quantile
    digest_npz = Path(args.stats_out).with_suffix(".digest.npz")
    np.savez(digest_npz,
             edges=resid_digest.edges, counts=resid_digest.counts,
             under=resid_digest.under, over=resid_digest.over,
             N=n, mean=mean_r, std=std_r)
    print(f"Residual digest: {digest_npz}")

    # 4-panel figure from accumulated 2D histograms.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(13, 23), gridspec_kw={"height_ratios": [1, 1, 1, 0.75]},
    )

    def _imshow_hist(ax, H, xbins, ybins):
        Hp = np.clip(H.T.astype(float), 1, None)
        ax.imshow(
            Hp,
            origin="lower",
            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(2, Hp.max())),
            interpolation="nearest",
        )

    # Panel 1
    _imshow_hist(ax1, H1, XBIN, XBIN)
    xx = np.logspace(np.log10(LO), np.log10(HI), 100)
    ax1.plot(xx, xx, "r-", lw=1.8, label=r"$y=x$ (perfect recovery)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$ observed (cnt/s)")
    ax1.set_ylabel(r"$\mathrm{Sci}_\mathrm{rec}$ recovered (cnt/s)")
    ax1.set_title(r"conservation-recovered Sci vs observed Sci (full mission window)", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel 2
    _imshow_hist(ax2, H2, XBIN, YBIN_RES)
    ax2.axhline(0, color="r", lw=1.8, label="zero")
    ax2.axhline(p50, color="orange", ls="--", lw=1.2, label=f"median = ${p50:+.1f}$")
    ax2.set_xscale("log")
    ax2.set_xlim(LO, HI); ax2.set_ylim(RES_LO, RES_HI)
    ax2.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$ observed (cnt/s)")
    ax2.set_ylabel(r"residual $\mathrm{Sci}_\mathrm{rec}-\mathrm{Sci}_\mathrm{obs}$ (cnt/s)")
    ax2.set_title("recovery residual (full population)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3 (fractional residual)
    _imshow_hist(ax3, H3, XBIN, YBIN_FRAC)
    ax3.axhline(0, color="r", lw=1.8, label="zero")
    ax3.set_xscale("log")
    ax3.set_xlim(LO, HI); ax3.set_ylim(FRAC_LO, FRAC_HI)
    ax3.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$ observed (cnt/s)")
    ax3.set_ylabel(r"fractional residual (\%)")
    ax3.set_title("fractional residual (full population)", fontsize=12)
    ax3.legend(loc="upper left", fontsize=11)
    ax3.grid(True, alpha=0.3, which="both")

    # Panel 4: conditional percentiles per sci bin
    sci_centers = np.sqrt(XBIN[:-1] * XBIN[1:])
    qs = {"P5": 0.05, "P50": 0.50, "P95": 0.95}
    for label, q in qs.items():
        vals = []
        for b in range(nbin4):
            d = digests4[b]
            total = d.under + int(d.counts.sum()) + d.over
            vals.append(d.quantile(q) if total > 0 else np.nan)
        ax4.plot(sci_centers, vals, label=label)
    ax4.axhline(0, color="r", lw=1.2)
    ax4.set_xscale("log")
    ax4.set_xlim(LO, HI); ax4.set_ylim(-200, 200)
    ax4.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$ observed (cnt/s)")
    ax4.set_ylabel("conditional residual (cnt/s)")
    ax4.set_title("conditional percentiles vs Sci_obs", fontsize=12)
    ax4.legend(loc="upper right", fontsize=11)
    ax4.grid(True, alpha=0.3, which="both")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Figure: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
