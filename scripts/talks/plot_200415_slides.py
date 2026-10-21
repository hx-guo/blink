#!/usr/bin/env python3
"""Slides version of paper f11 (GRB 200415A too-short failure) for the talk.

Same data pipeline as plot_200415_1ms_failure.py (local cache +
data/asim_mxgs + SPI-ACS time system), restyled for the deck: Chinese
labels, deck palette, big fonts, y axis in 1e6 c/s.

Output: talk-hxmt-saturation/he_f11_200415_failure.pdf
"""
import sys
import os
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts",)]  # sibling and shared script modules
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
import plot_hxmt_vs_spiacs as P

BW = 0.001
BKG = (-1.5, -0.1)
TAIL = (0.05, 0.15)
XLIM = (-0.010, 0.150)

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
})

C_OBS = "#1B3454"   # hxdeep observed
C_REC = "#5b9bd5"   # light blue reconstructed
C_ASIM = "#7d4fd0"  # purple external


def binned(t, edges):
    return np.histogram(t, bins=edges)[0] / (edges[1] - edges[0])


def main():
    tz, ltt = P.compute_spiacs_time_system()
    t0 = tz + TimeDelta(5.925 + ltt, format="sec")

    obs, fill, rng = P.load_hxmt(t0, 2, 2)
    asim = P.load_asim(t0, 2, 2)

    _e = np.arange(-2, 2 + BW, BW)
    peak_shift = (_e[:-1] + BW / 2)[np.histogram(asim, bins=_e)[0].argmax()]
    obs, fill, asim = obs - peak_shift, fill - peak_shift, asim - peak_shift
    t0 = t0 + TimeDelta(peak_shift, format="sec")

    all_t = np.concatenate([obs, fill])
    edges = np.arange(-2, 2 + BW, BW)
    x = edges[:-1] + BW / 2
    r_obs, r_all, r_asim = binned(obs, edges), binned(all_t, edges), binned(asim, edges)

    bm = (x >= BKG[0]) & (x < BKG[1])
    n_obs = r_obs - r_obs[bm].mean()
    n_all = r_all - r_all[bm].mean()
    n_asim = r_asim - r_asim[bm].mean()

    tm = (x >= TAIL[0]) & (x < TAIL[1])
    scale = n_all[tm].sum() / n_asim[tm].sum()
    n_asim_s = n_asim * scale

    lo_met, hi_met = P.load_hxmt_resets()
    t0_met = P.hxmt_t0_met(t0)
    sat_lo, sat_hi = lo_met - t0_met, hi_met - t0_met

    # ── plot (units: ms, 1e6 c/s) ──
    ms, M = 1000.0, 1e-6
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    xs = x * ms

    ax.axvspan(sat_lo * ms, sat_hi * ms, color="#C6362C", alpha=0.10, zorder=0)
    ax.step(xs, n_obs * M, where="mid", color=C_OBS, lw=1.2, label="HXMT/HE 观测")
    ax.step(xs, n_all * M, where="mid", color=C_REC, lw=1.4,
            label=f"HXMT/HE 重建 (+{len(fill):,})")
    ax.fill_between(xs, n_obs * M, n_all * M, step="mid", color=C_REC, alpha=0.30)
    ax.step(xs, n_asim_s * M, where="mid", color=C_ASIM, lw=1.4,
            label=f"ASIM/MXGS LED ($\\times${scale:.1f})")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    ax.text(0.5 * (sat_lo + sat_hi) * ms + 1, 0.86, "HXMT FIFO 饱和\n（三机箱共饱和）",
            ha="center", va="top", fontsize=9, color="#A02030",
            style="italic", fontweight="bold")
    ax.annotate("ASIM 实测\n~5 ms 尖峰", xy=(3.5, 1.22), xytext=(12, 1.34),
                fontsize=9, color=C_ASIM, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_ASIM, lw=1.2))
    ax.annotate("重建：线性斜坡平台", xy=(30, 0.13), xytext=(62, 0.45),
                fontsize=9, color="#2A6099", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#2A6099", lw=1.2))

    ax.set_ylabel("净计数率 ($10^6$ c/s)", fontsize=9)
    ax.set_xlabel("相对 $T_0$ 时间 (ms)　[bin = 1 ms]", fontsize=9)
    ax.set_ylim(-0.1, 1.8)
    ax.set_xlim(XLIM[0] * ms, XLIM[1] * ms)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92, borderaxespad=0.3)
    ax.grid(alpha=0.15)

    plt.tight_layout(pad=0.4)
    out = "/Users/skyair/Developer/ihep/talk-hxmt-saturation/he_f11_200415_failure.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
