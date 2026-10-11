#!/usr/bin/env python3
"""Talk-20 result figure 4: is the recovered fraction the same in every
energy channel?

What this figure does NOT do: overlay HE's and GBM's count spectra. The two
instruments redistribute the same photons very differently (measured on this
segment, HE-CsI / GBM-NaI runs from 0.20 at 78 keV to 7.95 at 716 keV), so a
direct overlay of deposited-energy spectra shows detector physics, not ours.
An honest spectrum-vs-spectrum comparison needs the response matrices.

What it does test, response-free and model-free: the ratio recovered/observed
per energy channel, ours against the one Fermi/GBM predicts independently.
  blue  HE: (observed + filled) / observed, from the three-box reconstruction
  red   GBM: 3 G_seg(E) / sum_box G_alive,box(E), from GBM's own gap-free curve
The two come from completely different data. Flat and coincident means the
recovery put back the same fraction at every energy, i.e. it did not distort
the spectrum.

Usage: python3 scripts/talk20_fig_segspec.py -o <dir>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).parent))
import talk20_style as ts

NUMBERS = Path("/Users/skyair/Developer/ihep/talk-20/figs/spectrum/"
               "tier2_260226A_numbers.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="/tmp")
    ap.add_argument("--numbers", default=str(NUMBERS))
    a = ap.parse_args()
    ts.apply(13)

    d = json.load(open(a.numbers))
    seg = d["segment"]
    s = d["views"]["csi"]["spectrum"]
    eb = np.array(s["ebins"]); ctr = np.sqrt(eb[:-1] * eb[1:])
    good = np.array(s["good"], bool)
    R_hx, S_hx = np.array(s["ratio_hxmt"]), np.array(s["sigma_ratio_hxmt"])
    R_g, S_g = np.array(s["ratio_gbm_pred"]), np.array(s["sigma_ratio_gbm_pred"])
    tot = s["total_ratio_hxmt"]

    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.axhline(tot, color=ts.AXIS, lw=1.4, ls=(0, (6, 4)), zorder=1)
    ax.text(eb[0] * 0.99, tot - 0.014, f"整段平均 {tot:.2f}", ha="left",
            va="top", fontsize=13, color=ts.TICK)
    ax.errorbar(ctr[good] / 1.012, R_g[good], yerr=S_g[good], fmt="s", ms=7,
                color=ts.RED, ecolor=ts.RED, capsize=3, lw=1.5, zorder=3,
                label="Fermi/GBM 独立算出来的同一个比值")
    ax.errorbar(ctr[good] * 1.012, R_hx[good], yerr=S_hx[good], fmt="o", ms=7,
                color=ts.SKY, ecolor=ts.SKY, capsize=3, lw=1.5, zorder=4,
                label="慧眼 HE：补回后 ÷ 记到的")

    ax.set_xscale("log")
    ax.set_xlim(eb[0] * 0.97, eb[-1] * 1.05)
    lo = np.nanmin(np.r_[R_g[good] - S_g[good], R_hx[good] - S_hx[good]])
    hi = np.nanmax(np.r_[R_g[good] + S_g[good], R_hx[good] + S_hx[good]])
    ax.set_ylim(lo - 0.10, hi + 0.16)
    ax.set_xticks([70, 100, 200, 400, 800])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("沉积能量（keV）")
    ax.set_ylabel("补回后 ÷ 记到的")
    ax.text(0.014, 0.05,
            f"慧眼 HE 的 CsI；丢数据的那一段：触发后 {seg[0]:.1f}–{seg[1]:.1f} 秒",
            transform=ax.transAxes, fontsize=12.5, color=ts.TICK)

    g = np.where(good)[0]
    for n, k in ((1, g[0]), (2, g[len(g) // 2]), (3, g[-1])):
        ts.mark(ax, n, ctr[k], max(R_hx[k] + S_hx[k], R_g[k] + S_g[k]) + 0.065)

    ax.legend(loc="upper left", fontsize=13.5, handlelength=1.4,
              labelspacing=0.5, borderaxespad=0.6)
    ts.clean(ax)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.105)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"talk20_segspec.{ext}", dpi=150 if ext == "png" else None)
        print(f"saved {out}/talk20_segspec.{ext}")
    print(f"  整段平均比值 慧眼 {tot:.2f} / GBM {s['total_ratio_gbm_pred']:.2f}；"
          f"逐道一致性 χ²/dof = {s['chi2_hxmt_vs_gbm_pred']:.1f}/{s['dof']}")


if __name__ == "__main__":
    main()
