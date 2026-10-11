#!/usr/bin/env python3
"""Talk-20 result figure 3: GRB 260226A, HXMT/HE vs Fermi/GBM, four panels.

Three traces per panel only (the engineering-counter prediction of
plot_260226_combo.py is dropped):
  navy  记到的        sky  补回后        red  Fermi/GBM 独立看到的

Usage: python3 scripts/talk20_fig_260226a.py -o <dir>    (run from blink root)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
import talk20_style as ts
from plot_hxmt_vs_gbm import load_hxmt_reconstruct, load_gbm_tte
from plot_hxmt_csi_multi import (load_hxmt_csi, load_gbm as load_gbm_csi,
                                 CONFIGS as CSI_CONFIGS)
from plot_hxmt_vs_ibis_bands import fit_background

BIN = 0.5
BEFORE, AFTER = 10.0, 80.0
XLIM = (-4.5, 50.0)
BB_BKG = (-6.5, -2.0, 60.0, 80.0)
NAI_BKG = (-4.5, -0.5, 65.0, 80.0)
SCALE_RANGE = (20.0, 40.0)
GBM_BB_DETS = ["n0", "n3", "b0"]
CSI_BANDS = [(70, 150), (150, 300), (300, 700)]
T0_UTC = "2026-02-26T10:37:50"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="/tmp")
    a = ap.parse_args()
    ts.apply(12)

    edges = np.arange(-BEFORE, AFTER + BIN, BIN)
    x = edges[:-1] + BIN / 2

    obs_t, fill_t = load_hxmt_reconstruct(BEFORE, AFTER)
    all_t = np.concatenate([obs_t, fill_t]) if len(fill_t) else obs_t
    gbm_bb = np.concatenate([load_gbm_tte(d, BEFORE, AFTER, None, None)
                             for d in GBM_BB_DETS])

    t1, t2, t3, t4 = BB_BKG
    r_obs = np.histogram(obs_t, bins=edges)[0] / BIN
    r_all = np.histogram(all_t, bins=edges)[0] / BIN
    r_gbm = np.histogram(gbm_bb, bins=edges)[0] / BIN
    bkgm = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    net_obs = r_obs - np.mean(r_all[bkgm])
    net_all = r_all - np.mean(r_all[bkgm])
    net_gbm = r_gbm - np.mean(r_gbm[bkgm])
    fill_bins = np.histogram(fill_t, bins=edges)[0] > 0
    sm = (x >= SCALE_RANGE[0]) & (x < SCALE_RANGE[1]) & ~fill_bins
    scale_bb = net_all[sm].sum() / net_gbm[sm].sum()
    net_gbm_s = net_gbm * scale_bb
    print(f"  HXMT: {len(obs_t):,} obs + {len(fill_t):,} fill; "
          f"GBM scale x{scale_bb:.2f}", file=sys.stderr)

    cfg = CSI_CONFIGS["260226A"]
    lut = np.load(cfg["csi_lut"])
    nobs_t, nobs_e, nfill_t, nfill_e = load_hxmt_csi(cfg["recon"], T0_UTC, lut)
    nall_t = np.concatenate([nobs_t, nfill_t])
    nall_e = np.concatenate([nobs_e, nfill_e])
    ext = cfg["externals"][0]
    gbm_t, gbm_e = load_gbm_csi(dir=ext["dir"], trig=ext["trig"],
                                dets=ext["dets"], tmet=ext["tmet"],
                                tutc=ext["tutc"], t0=T0_UTC)
    nx = x
    n1, n2, n3, n4 = NAI_BKG
    nbkgm = ((nx >= n1) & (nx < n2)) | ((nx >= n3) & (nx < n4))

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.0), sharex=True,
                             gridspec_kw={"hspace": 0.0})

    for k, (ax, (elo, ehi)) in enumerate(zip(axes, CSI_BANDS)):
        def rate(t, e):
            m = (e >= elo) & (e < ehi)
            return np.histogram(t[m], bins=edges)[0] / BIN
        r_o, r_a, r_g = rate(nobs_t, nobs_e), rate(nall_t, nall_e), rate(gbm_t, gbm_e)
        n_o = r_o - fit_background(nx, r_o, nbkgm, 1)
        n_a = r_a - fit_background(nx, r_a, nbkgm, 1)
        n_g = r_g - fit_background(nx, r_g, nbkgm, 1)
        fb = (r_a - r_o) > 1e-9
        smc = (nx >= SCALE_RANGE[0]) & (nx < SCALE_RANGE[1]) & ~fb
        sc = n_a[smc].sum() / n_g[smc].sum()
        print(f"  band {elo}-{ehi} keV: scale x{sc:.3f}", file=sys.stderr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = n_g * sc
            ok = ref > 0.05 * np.nanmax(ref)
            rrb = np.where(ok, n_a / ref, np.nan)
        for tag, m in (("有填充的道", ok & fb), ("没填充的道", ok & ~fb)):
            v = rrb[m & np.isfinite(rrb)]
            if len(v):
                q1, q2, q3 = np.percentile(v, [25, 50, 75])
                print(f"    {elo}-{ehi} keV [{tag}] 补回后/GBM = {q2:.2f} "
                      f"± {(q3 - q1) / 1.349:.2f} ({len(v)} bins)", file=sys.stderr)

        ax.step(nx, ref, where="mid", color=ts.RED, lw=1.0,
                label="Fermi/GBM 独立看到的（按面积归一）", zorder=3)
        ax.step(nx, n_o, where="mid", color=ts.NAVY, lw=1.0,
                label="慧眼 HE 记到的", zorder=4)
        ax.step(nx, n_a, where="mid", color=ts.SKY, lw=1.2,
                label="补回后", zorder=5)
        ax.axhline(0, color=ts.AXIS, lw=0.6, zorder=1)
        vb = (nx >= XLIM[0]) & (nx <= XLIM[1])
        ymx = np.nanmax(n_a[vb])
        ax.set_ylim(min(-0.05 * ymx, 1.2 * n_a[vb].min()),
                    ymx * (1.42 if k == 0 else 1.15))
        ax.text(0.015, 0.93, f"慧眼 HE 的 CsI，沉积能量 {elo}–{ehi} keV，{BIN * 1e3:.0f} 毫秒并道",
                transform=ax.transAxes, fontsize=12, va="top", color=ts.TICK)
        if k == 1:
            ipk2 = int(np.nanargmax(np.where(vb, n_a, -np.inf)))
            ts.mark(ax, 3, nx[ipk2] + 1.3, n_a[ipk2])
        if k == 0:
            ipk = int(np.nanargmax(np.where(vb, n_a, -np.inf)))
            ts.mark(ax, 1, nx[ipk] - 1.15, n_o[ipk])
            ts.mark(ax, 2, nx[ipk] + 1.15, n_a[ipk])
            h, l = ax.get_legend_handles_labels()
            ax.legend([h[1], h[2], h[0]], [l[1], l[2], l[0]],
                      loc="upper right", fontsize=12, handlelength=1.4,
                      labelspacing=0.4, borderaxespad=0.4)

    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ts.clean(ax)
        ax.spines["bottom"].set_visible(ax is axes[-1])
    fig.supylabel("净计数率（事件 / 秒）", fontsize=12.5, x=0.015, color=ts.TICK)
    axes[-1].set_xlim(*XLIM)
    axes[-1].set_xlabel("2026-02-26 触发后的秒数")
    fig.subplots_adjust(left=0.108, right=0.99, top=0.995, bottom=0.078)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext_ in ("pdf", "png"):
        fig.savefig(out / f"talk20_260226a.{ext_}",
                    dpi=150 if ext_ == "png" else None)
        print(f"saved {out}/talk20_260226a.{ext_}")


if __name__ == "__main__":
    main()
