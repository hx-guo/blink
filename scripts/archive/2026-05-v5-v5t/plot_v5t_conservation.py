#!/usr/bin/env python3
"""Conservation-recovery figure: Sci recovered from the conservation equation
(+ v5t closed-form C) vs observed Sci. Ideal recovery lies on y=x.

  Sci_rec = (PHO - Large_real)(1-d)/L - Wide/L - C(det,|mlat|,t)

Left:  Sci_rec vs Sci_obs (log-log), y=x line = perfect conservation recovery.
Right: residual = Sci_rec - Sci_obs.

Title carries the conservation equation itself. Sampled across full 8.9 yr.

Usage:
    python3 plot_v5t_conservation.py [--rowgroups-per-file 8] [--max-points 800000]
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

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
    n = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    maxa = pho - wide; lc = large + n * 1024.0; over = lc > maxa
    if over.any():
        nmax = np.maximum(np.floor((maxa - large) / 1024.0).astype(int), 0)
        lc = large + np.where(over, nmax, n) * 1024.0
    return lc


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

    ty = (pd.to_datetime(df["date"]).values.astype("datetime64[D]") - calib["t0"]).astype("timedelta64[D]").astype(float) / 365.25
    g = 1.0 - calib["beta"] * ty
    w = calib["w"]; kc = calib["k_coeffs"]
    k_t = (kc[0] + kc[1]*np.cos(w*ty) + kc[2]*np.sin(w*ty)
           + kc[3]*np.cos(2*w*ty) + kc[4]*np.sin(2*w*ty))
    box_idx = np.select([df["box"].values == b for b in "ABC"], [0, 1, 2], default=0)
    detid = box_idx * 6 + df["det"].values
    C_per_row = calib["s0_det"][detid] * g * (1.0 + k_t * mlat_term) + calib["C0"]

    large_v3 = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C_per_row)
    max_le = pho - ((sci + MIN_C_SLACK) * L + wide) / lf
    n3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    nmax = np.maximum(np.floor((max_le - large_raw) / 1024.0).astype(int), 0)
    large_v5 = large_raw + np.where(n3 > nmax, nmax, n3) * 1024.0
    base_v5 = (pho - large_v5) * lf / L - wide / L
    sci_rec = base_v5 - C_per_row          # conservation-recovered Sci
    return sci, sci_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    ap.add_argument("--calib", default="n_below_study/v5_npz/v5t_calib.npz")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    ap.add_argument("--rowgroups-per-file", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=800000)
    ap.add_argument("--output", default="plots/v5t_conservation.png")
    args = ap.parse_args()

    cz = np.load(args.calib)
    calib = {"s0_det": cz["s0_det"], "beta": float(cz["beta"]),
             "w": float(cz["w"]), "k_coeffs": cz["k_coeffs"], "C0": float(cz["C0"]),
             "t0": np.datetime64(str(cz["t0"]))}
    print(f"calib: g=1-{calib['beta']:.4f}t (linear), C0={calib['C0']:+.1f}")

    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)
    files = [f for f in sorted(glob.glob(os.path.join(args.cache_dir, "clean_relaxed_20*.parquet"))) if "sample" not in f]

    sci_all, rec_all = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        n_rg = pf.num_row_groups
        for rg in np.unique(np.linspace(0, n_rg - 1, args.rowgroups_per_file).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            sci, rec = process_rows(df, interp, calib)
            ok = np.isfinite(sci) & np.isfinite(rec) & (sci > 0)
            sci_all.append(sci[ok]); rec_all.append(rec[ok])
        print(f"  {os.path.basename(f)}: {sum(len(a) for a in sci_all):,} pts")

    sci = np.concatenate(sci_all); rec = np.concatenate(rec_all)
    if len(sci) > args.max_points:
        sel = np.random.RandomState(0).choice(len(sci), args.max_points, replace=False)
        sci, rec = sci[sel], rec[sel]
    resid = rec - sci
    print(f"Plotting {len(sci):,} points")
    print(f"residual: median={np.median(resid):+.2f}, "
          f"MAD={np.median(np.abs(resid-np.median(resid))):.2f}, "
          f"|resid|<30: {np.mean(np.abs(resid)<30)*100:.1f}%")

    LO, HI = 30.0, 10_000.0

    def dens(x, y, xb, yb):
        H, xe, ye = np.histogram2d(x, y, bins=[xb, yb])
        ix = np.clip(np.searchsorted(xe, x) - 1, 0, len(xe) - 2)
        iy = np.clip(np.searchsorted(ye, y) - 1, 0, len(ye) - 2)
        d = H[ix, iy].astype(float); d[d < 1] = 1
        return d

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(13, 23), gridspec_kw={"height_ratios": [1, 1, 1, 0.75]})

    # Left: Sci_rec vs Sci_obs, y=x
    m1 = (sci >= LO) & (sci <= HI) & (rec >= LO) & (rec <= HI)
    xb = np.logspace(np.log10(LO), np.log10(HI), 200)
    d1 = dens(sci[m1], rec[m1], xb, xb)
    o = np.argsort(d1)
    ax1.scatter(sci[m1][o], rec[m1][o], c=d1[o], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d1.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    xx = np.logspace(np.log10(LO), np.log10(HI), 100)
    ax1.plot(xx, xx, "r-", lw=1.8, label="y = x  (perfect recovery)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci$_{\\rm obs}$  observed (cnt/s)")
    ax1.set_ylabel("Sci$_{\\rm rec}$  recovered from conservation (cnt/s)")
    ax1.set_title("conservation-recovered Sci  vs  observed Sci", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11); ax1.grid(True, alpha=0.3, which="both")

    # Right: residual
    Y_LO, Y_HI = -400, 400
    m2 = (sci >= LO) & (sci <= HI) & (resid >= Y_LO) & (resid <= Y_HI)
    yb = np.linspace(Y_LO, Y_HI, 200)
    d2 = dens(sci[m2], resid[m2], xb, yb)
    o2 = np.argsort(d2)
    ax2.scatter(sci[m2][o2], resid[m2][o2], c=d2[o2], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d2.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    ax2.axhline(0, color="r", lw=1.8, label="zero")
    med = np.median(resid[m2])
    ax2.axhline(med, color="orange", ls="--", lw=1.2, label=f"median = {med:+.1f}")
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci$_{\\rm obs}$  observed (cnt/s)")
    ax2.set_ylabel("residual  Sci$_{\\rm rec}$ $-$ Sci$_{\\rm obs}$  (cnt/s)")
    ax2.set_title("recovery residual", fontsize=12)
    ax2.legend(loc="upper left", fontsize=11); ax2.grid(True, alpha=0.3, which="both")

    # ax3: fractional residual (data - pred) / data, in percent
    frac = (sci - rec) / sci * 100.0
    P_LO, P_HI = -40, 40
    m3 = (sci >= LO) & (sci <= HI) & (frac >= P_LO) & (frac <= P_HI)
    yb3 = np.linspace(P_LO, P_HI, 200)
    d3 = dens(sci[m3], frac[m3], xb, yb3)
    o3 = np.argsort(d3)
    ax3.scatter(sci[m3][o3], frac[m3][o3], c=d3[o3], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d3.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    ax3.axhline(0, color="r", lw=1.8, label="zero")
    medf = np.median(frac[m3])
    ax3.axhline(medf, color="orange", ls="--", lw=1.2, label=f"median = {medf:+.1f}\\%")
    ax3.set_xscale("log"); ax3.set_xlim(LO, HI); ax3.set_ylim(P_LO, P_HI)
    ax3.set_xlabel("Sci$_{\\rm obs}$  observed (cnt/s)")
    ax3.set_ylabel(r"$(\mathrm{Sci_{obs}}-\mathrm{Sci_{rec}})\,/\,\mathrm{Sci_{obs}}$  (\%)")
    ax3.set_title("fractional residual (data $-$ prediction) / data", fontsize=12)
    ax3.legend(loc="upper left", fontsize=11); ax3.grid(True, alpha=0.3, which="both")

    # ax4: per-rate CONDITIONAL percentiles — for each count-rate bin (discrete),
    # the bands hold 50/68/90/95% of the samples AT THAT rate.
    fin = np.isfinite(frac) & (sci > 0)
    sci_f, frac_f = sci[fin], frac[fin]
    xbc = np.logspace(np.log10(LO), np.log10(HI), 34)   # discrete count-rate bins
    xcen = np.sqrt(xbc[:-1] * xbc[1:])
    qdef = [(50, 25, 75), (68, 16, 84), (90, 5, 95), (95, 2.5, 97.5)]
    med = np.full(len(xcen), np.nan)
    band = {pct: [np.full(len(xcen), np.nan), np.full(len(xcen), np.nan)] for pct, _, _ in qdef}
    for i in range(len(xcen)):
        m = (sci_f >= xbc[i]) & (sci_f < xbc[i + 1])
        if m.sum() < 50:
            continue
        fv = frac_f[m]
        med[i] = np.median(fv)
        for pct, lo, hi in qdef:
            band[pct][0][i] = np.percentile(fv, lo)
            band[pct][1][i] = np.percentile(fv, hi)
    colors = {50: "#d62728", 68: "#ff7f0e", 90: "#2ca02c", 95: "#1f77b4"}
    for pct, _, _ in qdef:
        lo_line, hi_line = band[pct]
        ax4.plot(xcen, lo_line, "o-", color=colors[pct], lw=1.5, ms=3, label=f"{pct}\\%")
        ax4.plot(xcen, hi_line, "o-", color=colors[pct], lw=1.5, ms=3)
    ax4.plot(xcen, med, "k-", lw=2.0, label="median")
    ax4.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax4.set_xscale("log"); ax4.set_xlim(LO, HI); ax4.set_ylim(P_LO, P_HI)
    ax4.set_xlabel("Sci$_{\\rm obs}$  observed (cnt/s)")
    ax4.set_ylabel(r"fractional residual  (\%)")
    ax4.set_title(r"per-rate conditional percentiles (each band holds 50/68/90/95\% of samples at that rate)", fontsize=12)
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend(loc="upper right", fontsize=9.5, framealpha=0.92, ncol=3)

    fig.text(
        0.5, 0.985,
        r"$\mathrm{Sci}_\mathrm{rec} = \dfrac{1}{L_\mathrm{cyc}\cdot 16\,\mu\mathrm{s}}"
        r"\left[\,(\mathrm{PHO}-\mathrm{Large})\,\dfrac{L_\mathrm{cyc}-\mathrm{Dt}}{L_\mathrm{cyc}} - \mathrm{Wide}\,\right] - C$"
        "\n"
        r"$C(\mathrm{det},|\mathrm{mlat}|,t) = s_{0,\mathrm{det}}\,g(t)\,"
        r"\left[\,1+k(t)\max(0,\,|\mathrm{mlat}|-20^\circ)^2\,\right] + C_0$"
        "\n"
        r"$g(t)=1-\beta t$  (PMT outgassing, linear)          "
        r"$k(t)=c_0+a_1\cos\omega t+b_1\sin\omega t,\ \ \omega=2\pi/11\,\mathrm{yr}$  (solar cycle, single sinusoid)",
        ha="center", va="top", fontsize=14, fontweight="bold", linespacing=2.2)

    fig.text(
        0.5, 0.905,
        "Telemetry (per-second, per-detector):  PHO, Large, Wide, Sci = photo / large-event / wide-PSD / science counts;  "
        "Dt = dead-time counts;  $L_{\\rm cyc}$ = integration cycles ($\\times16\\,\\mu$s = frame time)\n"
        "Model parameters:  $s_{0,\\rm det}$ = per-detector sensitivity (18 constants);  $|$mlat$|$ = geomagnetic latitude;  "
        "$C_0=-5.9$ = common-mode offset.    Sampled 800k points across full 8.9 yr (2017$-$2026)",
        ha="center", va="top", fontsize=10, linespacing=2.0)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.845, bottom=0.04, hspace=0.30)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
