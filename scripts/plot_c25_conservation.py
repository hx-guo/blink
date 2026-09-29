#!/usr/bin/env python3
"""C25 conservation-recovery figure: Sci_rec from conservation + C25 model
vs observed Sci. Ideal recovery lies on y=x.

  Sci_rec = (PHO - Large_unwrap) (L_cyc - Dt) / L_cyc / (L_cyc * 16us)
            - Wide / (L_cyc * 16us)
            - C(i, |m|, t)

4 panels:
  1. Sci_rec vs Sci_obs (log-log, density-colored, y=x line)
  2. Residual Sci_rec - Sci_obs vs Sci_obs (cnt/s)
  3. Fractional residual (Sci_obs - Sci_rec) / Sci_obs in %
  4. Per-rate conditional percentiles 50/68/90/95
"""
from __future__ import annotations
import argparse, glob, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

L_CYCLES_TO_SEC = 16e-6
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
BOX_ID = {"a":0,"b":1,"c":2,"A":0,"B":1,"C":2}


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    pho = np.asarray(pho, float); large = np.asarray(large, float)
    wide = np.asarray(wide, float); sci = np.asarray(sci, float)
    L = np.asarray(lc, float) * L_CYCLES_TO_SEC
    lf = 1.0 - np.asarray(dt, float) / np.asarray(lc, float)
    predicted = pho - (wide + (sci + C) * L) / lf
    n = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    maxa = pho - wide; lcr = large + n * 1024.0; over = lcr > maxa
    if over.any():
        nmax = np.maximum(np.floor((maxa - large) / 1024.0).astype(int), 0)
        lcr = large + np.where(over, nmax, n) * 1024.0
    return lcr


def C_model_c25(mlat, t, box, det, P):
    A_i = np.array(P["a_det"])
    alpha_m = P["alpha"]; mu_m = P["mu_m"]; w_m = P["k_m"]
    alpha_t = P["amp0"]; mu_t = P["mu_t"]; w_t = P["k_t"]
    C_0 = P["C0"]
    bi = np.array([BOX_ID[b] for b in box])
    di = np.asarray(det, dtype=int)
    A = A_i[bi*6 + di]
    sm = sigm((np.abs(mlat) - mu_m) / w_m)
    st = sigm((t - mu_t) / w_t)
    g = 1.0 + alpha_m * sm
    return A * g * (1.0 - alpha_t * g * st) + C_0


def process_rows(df, interp, P, t_ref):
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    abs_mlat = np.abs(interp(pts))
    abs_mlat = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)
    pho = df["PHO"].astype("float64").values
    large_raw = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc

    ty = ((pd.to_datetime(df["date"]).values.astype("datetime64[D]") - t_ref)
          .astype("timedelta64[D]").astype(float) / 365.25)
    C_per_row = C_model_c25(abs_mlat, ty, df["box"].values, df["det"].values, P)

    large_v3 = unwrap_v2(pho, large_raw, wide, sci, lc, dtv, C_per_row)
    max_le = pho - (sci * L + wide) / lf
    n3 = np.round((large_v3 - large_raw) / 1024).astype(int)
    nmax = np.maximum(np.floor((max_le - large_raw) / 1024.0).astype(int), 0)
    large_final = large_raw + np.where(n3 > nmax, nmax, n3) * 1024.0
    sci_rec = (pho - large_final) * lf / L - wide / L - C_per_row
    return sci, sci_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    ap.add_argument("--c25-json", default="/tmp/per_det_25param.json")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    ap.add_argument("--rowgroups-per-file", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=800000)
    ap.add_argument("--output", default="plots/c25_conservation.png")
    args = ap.parse_args()

    P = json.loads(Path(args.c25_json).read_text())
    print(f"C25 params loaded: alpha_m={P['alpha']:.3f}, mu_m={P['mu_m']:.2f}, "
          f"alpha_t={P['amp0']:.3f}, mu_t={P['mu_t']:.2f}, C_0={P['C0']:+.1f}")
    t_ref = np.datetime64("2017-06-22")

    grid = np.load(args.aacgm_grid)
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)
    files = [f for f in sorted(glob.glob(os.path.join(args.cache_dir, "clean_relaxed_20*.parquet")))
             if "sample" not in f]

    sci_all, rec_all = [], []
    for f in files:
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        for rg in np.unique(np.linspace(0, n_rg - 1, args.rowgroups_per_file).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            sci, rec = process_rows(df, interp, P, t_ref)
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

    # Panel 1: Sci_rec vs Sci_obs
    m1 = (sci >= LO) & (sci <= HI) & (rec >= LO) & (rec <= HI)
    xb = np.logspace(np.log10(LO), np.log10(HI), 200)
    d1 = dens(sci[m1], rec[m1], xb, xb); o = np.argsort(d1)
    ax1.scatter(sci[m1][o], rec[m1][o], c=d1[o], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d1.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    xx = np.logspace(np.log10(LO), np.log10(HI), 100)
    ax1.plot(xx, xx, "r-", lw=1.8, label=r"$y=x$  (perfect recovery)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$  observed (cnt/s)")
    ax1.set_ylabel(r"$\mathrm{Sci}_\mathrm{rec}$  recovered from conservation (cnt/s)")
    ax1.set_title(r"conservation-recovered Sci  vs  observed Sci", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11); ax1.grid(True, alpha=0.3, which="both")

    # Panel 2: residual
    Y_LO, Y_HI = -400, 400
    m2 = (sci >= LO) & (sci <= HI) & (resid >= Y_LO) & (resid <= Y_HI)
    yb = np.linspace(Y_LO, Y_HI, 200)
    d2 = dens(sci[m2], resid[m2], xb, yb); o2 = np.argsort(d2)
    ax2.scatter(sci[m2][o2], resid[m2][o2], c=d2[o2], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d2.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    ax2.axhline(0, color="r", lw=1.8, label="zero")
    med = np.median(resid[m2])
    ax2.axhline(med, color="orange", ls="--", lw=1.2, label=f"median = ${med:+.1f}$")
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$  observed (cnt/s)")
    ax2.set_ylabel(r"residual  $\mathrm{Sci}_\mathrm{rec}-\mathrm{Sci}_\mathrm{obs}$  (cnt/s)")
    ax2.set_title("recovery residual", fontsize=12)
    ax2.legend(loc="upper left", fontsize=11); ax2.grid(True, alpha=0.3, which="both")

    # Panel 3: fractional residual
    frac = (sci - rec) / sci * 100.0
    P_LO, P_HI = -40, 40
    m3 = (sci >= LO) & (sci <= HI) & (frac >= P_LO) & (frac <= P_HI)
    yb3 = np.linspace(P_LO, P_HI, 200)
    d3 = dens(sci[m3], frac[m3], xb, yb3); o3 = np.argsort(d3)
    ax3.scatter(sci[m3][o3], frac[m3][o3], c=d3[o3], cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(d3.max(), 2)), s=2, alpha=0.5,
                rasterized=True, edgecolor="none")
    ax3.axhline(0, color="r", lw=1.8, label="zero")
    medf = np.median(frac[m3])
    ax3.axhline(medf, color="orange", ls="--", lw=1.2,
                label=f"median = ${medf:+.1f}$\\%")
    ax3.set_xscale("log"); ax3.set_xlim(LO, HI); ax3.set_ylim(P_LO, P_HI)
    ax3.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$  observed (cnt/s)")
    ax3.set_ylabel(r"$(\mathrm{Sci_{obs}}-\mathrm{Sci_{rec}})\,/\,\mathrm{Sci_{obs}}$  (\%)")
    ax3.set_title(r"fractional residual  (data $-$ prediction) / data", fontsize=12)
    ax3.legend(loc="upper left", fontsize=11); ax3.grid(True, alpha=0.3, which="both")

    # Panel 4: per-rate conditional percentiles
    fin = np.isfinite(frac) & (sci > 0)
    sci_f, frac_f = sci[fin], frac[fin]
    xbc = np.logspace(np.log10(LO), np.log10(HI), 34)
    xcen = np.sqrt(xbc[:-1] * xbc[1:])
    qdef = [(50, 25, 75), (68, 16, 84), (90, 5, 95), (95, 2.5, 97.5)]
    medp = np.full(len(xcen), np.nan)
    band = {pct: [np.full(len(xcen), np.nan), np.full(len(xcen), np.nan)] for pct, _, _ in qdef}
    for i in range(len(xcen)):
        m = (sci_f >= xbc[i]) & (sci_f < xbc[i + 1])
        if m.sum() < 50: continue
        fv = frac_f[m]
        medp[i] = np.median(fv)
        for pct, lo, hi in qdef:
            band[pct][0][i] = np.percentile(fv, lo)
            band[pct][1][i] = np.percentile(fv, hi)
    colors = {50: "#d62728", 68: "#ff7f0e", 90: "#2ca02c", 95: "#1f77b4"}
    for pct, _, _ in qdef:
        lo_l, hi_l = band[pct]
        ax4.plot(xcen, lo_l, "o-", color=colors[pct], lw=1.5, ms=3, label=f"{pct}\\%")
        ax4.plot(xcen, hi_l, "o-", color=colors[pct], lw=1.5, ms=3)
    ax4.plot(xcen, medp, "k-", lw=2.0, label="median")
    ax4.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax4.set_xscale("log"); ax4.set_xlim(LO, HI); ax4.set_ylim(P_LO, P_HI)
    ax4.set_xlabel(r"$\mathrm{Sci}_\mathrm{obs}$  observed (cnt/s)")
    ax4.set_ylabel(r"fractional residual  (\%)")
    ax4.set_title(r"per-rate conditional percentiles  (each band holds 50/68/90/95\% at that rate)",
                  fontsize=12)
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend(loc="upper right", fontsize=9.5, framealpha=0.92, ncol=3)

    # Header with C25 formula
    A_i = np.array(P["a_det"])
    fig.text(
        0.5, 0.992,
        r"$\mathrm{Sci}_\mathrm{rec}\,=\,\dfrac{1}{L_\mathrm{cyc}\cdot 16\,\mu\mathrm{s}}"
        r"\left[\,(\mathrm{PHO}-\mathrm{Large})\,\dfrac{L_\mathrm{cyc}-\mathrm{Dt}}{L_\mathrm{cyc}}-\mathrm{Wide}\,\right]\,-\,C$"
        "\n"
        r"$S(x;\mu,w)\,=\,\dfrac{1}{1+e^{-(x-\mu)/w}}$ \quad\quad "
        r"$g(|m|)\,=\,1+\alpha_m\,S(|m|;\mu_m,w_m)$"
        "\n"
        r"$C(i,|m|,t)\,=\,A_i\,g(|m|)\,\left[\,1-\alpha_t\,g(|m|)\,S(t;\mu_t,w_t)\,\right]\,+\,C_0$"
        "\n"
        fr"$\alpha_m={P['alpha']:.2f},\ \mu_m={P['mu_m']:.1f}^\circ,\ w_m={P['k_m']:.1f}^\circ;\ "
        fr"\alpha_t={P['amp0']:.3f},\ \mu_t={P['mu_t']:.2f}\,\mathrm{{yr}},\ "
        fr"w_t={P['k_t']:.2f}\,\mathrm{{yr}};\ "
        fr"C_0={P['C0']:+.1f},\ \{{A_i\}}\in[{A_i.min():.0f},\,{A_i.max():.0f}]$",
        ha="center", va="top", fontsize=14, fontweight="bold", linespacing=2.0)

    fig.text(
        0.5, 0.895,
        r"Telemetry (per-second, per-detector):  PHO, Large, Wide, Sci "
        r"$=$ photo / large-event / wide-PSD / science counts;  "
        r"Dt $=$ dead-time counts;  $L_\mathrm{cyc}\,\times\,16\,\mu\mathrm{s}$ $=$ frame time."
        "\n"
        r"Model: 25 parameters $=$ 18 per-det amplitudes $\{A_i\}$ + 7 shared shape/decay constants. "
        r"Purely phenomenological --- no PMT-outgassing or solar-cycle narrative.    "
        f"Sampled {len(sci):,} points across full 8.9\\,yr (2017$-$2026).",
        ha="center", va="top", fontsize=10, linespacing=1.8)

    fig.subplots_adjust(left=0.10, right=0.95, top=0.840, bottom=0.04, hspace=0.30)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
