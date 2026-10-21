#!/usr/bin/env python3
"""Slides versions of paper f15 (250919A) and f7 (260226A), simplified to
2 panels each for the conference talk.

Top panel: all HXMT/HE events vs externals + engineering channel.
Bottom panel: HE-CsI 300-700 keV (the cleanest-matched band).
Wide 16:9-friendly aspect, deck palette, Chinese labels, big fonts.

Data: same local pipelines as plot_250919_combo.py / plot_260226_combo.py.

Outputs:
  talk-hxmt-saturation/he_f15_250919_2panel.pdf
  talk-hxmt-saturation/he_f7_260226_2panel.pdf
"""
import sys
import os
from pathlib import Path

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts", "../lib",)]  # sibling and shared script modules
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_hxmt_csi_multi import CONFIGS, load_hxmt_csi, LOADERS, hxmt_met
from engineering_prediction import load_engineering_prediction, T_REF

OUT_DIR = "/Users/skyair/Developer/ihep/talk-hxmt-saturation"
OBS_C = "#1B3454"   # hxdeep observed
REC_C = "#5b9bd5"   # light blue reconstructed
ENG_C = "#2e8b57"   # green engineering

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.linewidth": 0.8,
})

BAND = (300, 700)

RUNS = {
    "250919A": dict(
        bin=0.1, before=30.0, after=140.0, xlim=(-3.0, 18.0),
        eng=dict(date_str="20250919", hour_str="000000",
                 date64="2025-09-19",
                 orbit_path="data/hxmt_aux/HXMT_20250919T00_Orbit_FFFFFF_V1_1K.FITS"),
        out="he_f15_250919_2panel.pdf",
        binlabel="0.1 s", toptag="全部事件",
    ),
    "260226A": dict(
        bin=0.5, before=10.0, after=80.0, xlim=(-4.5, 62.0),
        eng=dict(date_str="20260226", hour_str="100000",
                 date64="2026-02-26",
                 orbit_path="data/hxmt_aux/HXMT_20260226T10_Orbit_FFFFFF_V1_1K.FITS"),
        out="he_f7_260226_2panel.pdf",
        binlabel="0.5 s", toptag="全部事件", t0_override="2026-02-26T10:37:50",
    ),
}

EXT_LABEL = {
    "gbm": "Fermi/GBM",
    "svom": "SVOM/GRM",
}


def load_hxmt_all(recon, t0):
    trig = hxmt_met(t0)
    ot, ft = [], []
    with open(recon) as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 5 or p[0] == "box":
                continue
            typ, met = p[1], float(p[2])
            (ot if typ == "EVT" else ft).append(met - trig)
    return np.array(ot), np.array(ft)


def run(burst):
    R = RUNS[burst]
    BIN, BEFORE, AFTER = R["bin"], R["before"], R["after"]
    xl = R["xlim"]

    cfg = CONFIGS[burst]
    t0 = R.get("t0_override", cfg["t0"])
    lut = np.load(cfg["csi_lut"])
    ot, oe, ft, fe = load_hxmt_csi(cfg["recon"], t0, lut)
    at, ae = np.concatenate([ot, ft]), np.concatenate([oe, fe])
    all_ot, all_ft = load_hxmt_all(cfg["recon"], t0)
    all_at = np.concatenate([all_ot, all_ft])

    exts = []
    for e in cfg["externals"]:
        t, en = LOADERS[e["type"]](t0=t0, **{k: v for k, v in e.items()
                                             if k not in ("type", "color", "label")})
        exts.append((e, t, en))

    trig_met = hxmt_met(t0)
    ty = (np.datetime64(R["eng"]["date64"]) - T_REF).astype(
        "timedelta64[D]").astype(float) / 365.25
    eng_t, eng_rate = load_engineering_prediction(
        date_str=R["eng"]["date_str"], hour_str=R["eng"]["hour_str"],
        trigger_met=trig_met, before=BEFORE, after=AFTER,
        t_years_const=ty, orbit_path=R["eng"]["orbit_path"])

    edges = np.arange(-BEFORE, AFTER + BIN, BIN)
    x = edges[:-1] + BIN / 2
    b = cfg["bkg"]
    bkgm = ((x >= b[0]) & (x < b[1])) | ((x >= b[2]) & (x < b[3]))
    s = cfg["scale_range"]
    fillbin = np.histogram(ft, bins=edges)[0] > 0
    sm = (x >= s[0]) & (x < s[1]) & (~fillbin)
    vis = (x >= xl[0]) & (x < xl[1])

    def net(t, e, lo, hi):
        m = (e >= lo) & (e < hi)
        r = np.histogram(t[m], bins=edges)[0] / BIN
        return r - np.polyval(np.polyfit(x[bkgm], r[bkgm], 1), x)

    def net_all(t):
        r = np.histogram(t, bins=edges)[0] / BIN
        return r - np.polyval(np.polyfit(x[bkgm], r[bkgm], 1), x)

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(6.9, 3.9), sharex=True,
        gridspec_kw={"hspace": 0.0})

    # ---- top: all events + engineering ----
    nHo, nHa = net_all(all_ot), net_all(all_at)
    axT.fill_between(x, nHo, nHa, step="mid", color=REC_C, alpha=0.30, zorder=2)
    axT.step(x, nHo, where="mid", color=OBS_C, lw=1.1, label="HXMT/HE 观测", zorder=4)
    axT.step(x, nHa, where="mid", color=REC_C, lw=1.3, label="HXMT/HE 重建", zorder=5)
    ymax = nHa[vis].max()
    for e, t, en in exts:
        nE = net_all(t)
        sc = nHa[sm].sum() / nE[sm].sum() if nE[sm].sum() > 0 else 1.0
        lab = EXT_LABEL.get(e["type"], e["label"])
        axT.step(x, nE * sc, where="mid", color=e["color"], lw=1.0,
                 label=f"{lab} $\\times${sc:.2f}", zorder=3)
    if eng_t is not None:
        ebm = ((eng_t >= b[0]) & (eng_t < b[1])) | ((eng_t >= b[2]) & (eng_t < b[3]))
        net_eng = eng_rate - np.mean(eng_rate[ebm])
        axT.step(eng_t, net_eng, where="post", color=ENG_C, lw=1.3,
                 label="工程通道 $\\widehat{S}_{\\rm rec}^{\\rm eng}$ (1 Hz)", zorder=3)
        evis = (eng_t >= xl[0]) & (eng_t < xl[1])
        ymax = max(ymax, net_eng[evis].max())
    axT.axhline(0, color="grey", lw=0.5)
    axT.set_ylabel("净计数率 (c/s)", fontsize=10)
    axT.text(0.015, 0.90, f"{R['toptag']} · {R['binlabel']}",
             transform=axT.transAxes, fontweight="bold", va="top", fontsize=10)
    axT.legend(loc="upper right", ncol=2, fontsize=8, frameon=True,
               framealpha=0.9, borderaxespad=0.25, columnspacing=0.9,
               handlelength=1.4)
    axT.set_ylim(min(0, nHo[vis].min() * 1.1), ymax * 1.30)
    axT.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    # ---- bottom: CsI 300-700 keV ----
    lo, hi = BAND
    nHo, nHa = net(ot, oe, lo, hi), net(at, ae, lo, hi)
    axB.fill_between(x, nHo, nHa, step="mid", color=REC_C, alpha=0.30, zorder=2)
    axB.step(x, nHo, where="mid", color=OBS_C, lw=1.1, zorder=4)
    axB.step(x, nHa, where="mid", color=REC_C, lw=1.3, zorder=5)
    for e, t, en in exts:
        nE = net(t, en, lo, hi)
        sc = nHa[sm].sum() / nE[sm].sum() if nE[sm].sum() > 0 else 1.0
        axB.step(x, nE * sc, where="mid", color=e["color"], lw=1.0, zorder=3)
    axB.axhline(0, color="grey", lw=0.5)
    axB.set_ylabel("净计数率 (c/s)", fontsize=10)
    axB.text(0.015, 0.90, f"CsI {lo}–{hi} keV（沉积能）· {R['binlabel']}",
             transform=axB.transAxes, fontweight="bold", va="top", fontsize=10)
    axB.set_ylim(min(0, nHo[vis].min() * 1.1), nHa[vis].max() * 1.18)
    axB.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    axB.set_xlim(*xl)
    axB.set_xlabel(f"相对 $T_0$ 时间 (s)　[$T_0$ = {t0} UTC]", fontsize=10)
    axB.tick_params(labelsize=9)
    axT.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.13)
    out = os.path.join(OUT_DIR, R["out"])
    fig.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    for burst in (sys.argv[1:] or ["250919A", "260226A"]):
        run(burst)
