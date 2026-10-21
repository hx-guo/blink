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

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib", "../saturation",)]  # sibling and shared script modules
import pubstyle  # noqa: E402
pubstyle.apply()
from scipy.interpolate import RegularGridInterpolator

pass  # was a second sys.path entry for the scripts directory
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


def plot_scatter(sci_s, rec_s, output, plot_lo=100.0, plot_hi=5000.0):
    """Publication figure: recovered-vs-observed scatter of sampled frames.

    sci_s/rec_s are a uniform random sample of the full population;
    plot_lo crops the sparsely populated lower-left corner.
    """
    fig, ax1 = plt.subplots(figsize=(pubstyle.COL_W, 2.85))
    m = np.isfinite(sci_s) & np.isfinite(rec_s) & (sci_s > 0) & (rec_s > 0)
    sci_s, rec_s = sci_s[m], rec_s[m]
    # Local point density from a log-log 2D histogram lookup; draw
    # low-density points first so dense cores sit on top.
    lx, ly = np.log10(sci_s), np.log10(rec_s)
    H, xe, ye = np.histogram2d(lx, ly, bins=240)
    ix = np.clip(np.searchsorted(xe, lx) - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(ye, ly) - 1, 0, H.shape[1] - 1)
    dens = H[ix, iy]
    order = np.argsort(dens)
    ax1.scatter(sci_s[order], rec_s[order], c=dens[order], s=1.2,
                cmap="viridis", norm=LogNorm(vmin=1, vmax=dens.max()),
                linewidths=0, rasterized=True, zorder=1)
    xx = np.logspace(np.log10(plot_lo), np.log10(plot_hi), 100)
    ax1.plot(xx, xx, "r-", lw=1.0, zorder=3,
             label=r"$y=x$ (perfect recovery)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(plot_lo, plot_hi); ax1.set_ylim(plot_lo, plot_hi)
    ax1.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$ observed (cnt/s)")
    ax1.set_ylabel(r"$\mathrm{Sci}_\mathrm{rec}$ recovered (cnt/s)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.25, which="both", zorder=0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    ap.add_argument("--c25-json", default="/tmp/per_det_25param.json")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    ap.add_argument("--output", default="plots/c25_conservation_fullpop.png")
    ap.add_argument("--stats-out", default="plots/c25_fullpop_stats.json")
    ap.add_argument("--window-end", default="2026-05-31",
                    help="Hard cap on date (inclusive). Default 2026-05-31.")
    ap.add_argument("--from-sample", default=None,
                    help="Path to a saved sample npz; replot without streaming.")
    ap.add_argument("--sample-size", type=int, default=200_000,
                    help="Reservoir-sample size for the scatter figure.")
    ap.add_argument("--plot-lo", type=float, default=100.0,
                    help="Lower axis bound of the scatter plot (crop empty corner).")
    ap.add_argument("--plot-hi", type=float, default=5000.0,
                    help="Upper axis bound of the scatter plot.")
    args = ap.parse_args()

    if args.from_sample:
        z = np.load(args.from_sample)
        plot_scatter(z["sci"], z["rec"], args.output, args.plot_lo, args.plot_hi)
        return 0

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
    # Reservoir sample of (sci, rec) pairs for the scatter figure
    rng = np.random.default_rng(42)
    K = args.sample_size
    S_sci = np.empty(K); S_rec = np.empty(K)
    n_seen = 0
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

            # Vectorized Algorithm-R reservoir update
            m = sci.size
            g = n_seen + np.arange(m)
            if n_seen < K:
                take = min(K - n_seen, m)
                S_sci[n_seen:n_seen + take] = sci[:take]
                S_rec[n_seen:n_seen + take] = rec[:take]
                sel = rng.random(m - take) < K / (g[take:] + 1.0)
                pos = rng.integers(0, K, size=int(sel.sum()))
                S_sci[pos] = sci[take:][sel]; S_rec[pos] = rec[take:][sel]
            else:
                sel = rng.random(m) < K / (g + 1.0)
                pos = rng.integers(0, K, size=int(sel.sum()))
                S_sci[pos] = sci[sel]; S_rec[pos] = rec[sel]
            n_seen += m

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
    hist_npz = Path(args.stats_out).with_suffix(".hists.npz")
    np.savez(hist_npz, H1=H1, H2=H2, H3=H3, XBIN=XBIN)
    print(f"2D histograms: {hist_npz}")
    K_eff = min(K, n_seen)
    sample_npz = Path(args.stats_out).with_suffix(".sample.npz")
    np.savez(sample_npz, sci=S_sci[:K_eff], rec=S_rec[:K_eff], n_total=n_seen)
    print(f"Scatter sample ({K_eff:,} of {n_seen:,}): {sample_npz}")
    np.savez(digest_npz,
             edges=resid_digest.edges, counts=resid_digest.counts,
             under=resid_digest.under, over=resid_digest.over,
             N=n, mean=mean_r, std=std_r)
    print(f"Residual digest: {digest_npz}")

    plot_scatter(S_sci[:K_eff], S_rec[:K_eff], args.output, args.plot_lo, args.plot_hi)
    print(f"Figure: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
