#!/usr/bin/env python3
"""Talk-20 result figure 2: FRB 200428 / SGR 1935+2154, HXMT/HE vs INTEGRAL/IBIS.

Three traces only (the engineering-counter channel of
plot_hxmt_vs_ibis_bands.py is dropped):
  navy 记到的     sky 补回后     red INTEGRAL/IBIS 独立看到的
plus the two CHIME radio pulses as grey dotted time markers.

Usage: python3 scripts/talks/talk20_fig_200428.py -o <dir>    (run from blink root)
"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("C25_JSON", "data/hxmt_aux/per_det_25param.json")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from astropy.time import Time

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts", "../lib",)]  # sibling and shared script modules
import talk20_style as ts
from plot_hxmt_vs_ibis import (compute_hxmt_light_travel,
                               compute_integral_light_travel, load_ibis_events,
                               HXMT_TRIGGER_UTC_STR, IBIS_FILE,
                               INTEGRAL_ORBIT_FILE, HXMT_ORBIT_FILE)
from plot_hxmt_vs_ibis_bands import (CACHE_PW, BANDS, RADIO_PEAKS,
                                     channel_to_kev_lut, load_hxmt_nai,
                                     load_hxmt_all, fit_background)

BIN = 0.005
BEFORE, AFTER = 0.3, 0.9
BKG = (-0.3, -0.05, 0.7, 0.9)
SCALE_RANGE = (0.35, 0.65)
XLIM = (0.33, 0.63)
IBIS_BAND = (20.0, 200.0)   # match HE/NaI's deposited band for the reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="/tmp")
    a = ap.parse_args()
    ts.apply(12)

    trig = Time(HXMT_TRIGGER_UTC_STR, scale="utc")
    trig_met = (trig - Time("2012-01-01T00:00:00", scale="utc")).sec
    trig_ijd = (trig.tt - Time("2000-01-01T00:00:00", scale="tt")).to("day").value
    hlt = compute_hxmt_light_travel(trig_met, trig, HXMT_ORBIT_FILE)
    ilt = compute_integral_light_travel(trig_ijd, IBIS_FILE, INTEGRAL_ORBIT_FILE)

    E_lut = channel_to_kev_lut()
    obs_t, obs_e, fill_t, fill_e = load_hxmt_nai(CACHE_PW, trig_met, E_lut, hlt)
    all_t = np.concatenate([obs_t, fill_t]); all_e = np.concatenate([obs_e, fill_e])
    ibis_t, ibis_e = load_ibis_events(IBIS_FILE, trig_ijd, ilt)
    ibm = (ibis_e >= IBIS_BAND[0]) & (ibis_e < IBIS_BAND[1])
    ibis_t = ibis_t[ibm]
    print(f"  HXMT NaI: {len(obs_t):,} obs + {len(fill_t):,} fill; "
          f"IBIS {IBIS_BAND[0]:.0f}-{IBIS_BAND[1]:.0f} keV: {len(ibis_t):,}",
          file=sys.stderr)

    edges = np.arange(-BEFORE, AFTER + BIN, BIN)
    x = edges[:-1] + BIN / 2
    t1, t2, t3, t4 = BKG
    bkgm = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    sm = (x >= SCALE_RANGE[0]) & (x < SCALE_RANGE[1])
    vis = (x >= XLIM[0]) & (x < XLIM[1])

    fig, ax = plt.subplots(figsize=(10.4, 7.0))

    r_obs = np.histogram(obs_t, bins=edges)[0] / BIN
    r_all = np.histogram(all_t, bins=edges)[0] / BIN
    r_ib = np.histogram(ibis_t, bins=edges)[0] / BIN
    n_obs = r_obs - fit_background(x, r_obs, bkgm, 1)
    n_all = r_all - fit_background(x, r_all, bkgm, 1)
    n_ib = r_ib - fit_background(x, r_ib, bkgm, 1)
    sc = n_all[sm].sum() / n_ib[sm].sum() if n_ib[sm].sum() > 0 else 1.0

    for rp, _ in RADIO_PEAKS:
        ax.axvline(rp, color="#3A3A3A", ls=":", lw=1.2, alpha=0.85, zorder=1)
    ax.step(x, n_ib * sc, where="mid", color=ts.RED, lw=1.1,
            label=f"INTEGRAL/IBIS 独立看到的（20–200 keV，×{sc:.0f}）", zorder=3)
    ax.step(x, n_obs, where="mid", color=ts.NAVY, lw=1.1,
            label="慧眼 HE 记到的", zorder=6)
    ax.step(x, n_all, where="mid", color=ts.SKY, lw=1.4,
            label="补回后", zorder=7)
    ax.axhline(0, color=ts.AXIS, lw=0.6)
    ax.set_ylim(min(0, n_all[vis].min() * 1.1),
                max(n_all[vis].max(), (n_ib * sc)[vis].max()) * 1.30)
    ax.text(0.014, 0.975, f"慧眼 HE 的 NaI，{BIN * 1e3:.0f} 毫秒并道",
            transform=ax.transAxes, fontsize=12.5, va="top", color=ts.TICK)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
    ts.clean(ax)
    h, l = ax.get_legend_handles_labels()
    h = [h[1], h[2], h[0], Line2D([0], [0], color="#3A3A3A", ls=":", lw=1.2)]
    l = [l[1], l[2], l[0], "CHIME 的两个射电脉冲"]
    ax.legend(h, l, loc="upper right", fontsize=12.5, handlelength=1.5,
              labelspacing=0.45, borderaxespad=0.4)
    ax.set_ylabel("净计数率（事件 / 秒）", fontsize=12.5)
    top = max(n_all[vis].max(), (n_ib * sc)[vis].max())
    pk = x[vis][int(np.nanargmax(n_all[vis]))]
    ts.mark(ax, 1, 0.503, 0.30 * top)
    ts.mark(ax, 2, pk + 0.004, 1.06 * top)
    ts.mark(ax, 3, RADIO_PEAKS[0][0], 1.20 * top)
    axes = [ax]
    axes[-1].set_xlim(*XLIM)
    axes[-1].set_xlabel("2020-04-28 触发后的秒数")
    fig.subplots_adjust(left=0.112, right=0.99, top=0.99, bottom=0.10)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"talk20_200428.{ext}", dpi=150 if ext == "png" else None)
        print(f"saved {out}/talk20_200428.{ext}")


if __name__ == "__main__":
    main()
