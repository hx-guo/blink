#!/usr/bin/env python3
"""GRB 200415A — 1 ms HXMT/HE reconstruction vs ASIM/MXGS LED (too-short failure).

Single-panel figure for the paper's Section 7.1 (three-box co-saturation +
sub-millisecond substructure = worst case). Rebuilt from scratch.

Reuses the time-system / data loaders of plot_hxmt_vs_spiacs.py so the T0
derivation (SPI-ACS TIMEZERO + light-travel) is identical.

Run from the blink/ directory with the project venv:
    .venv/bin/python scripts/plot_200415_1ms_failure.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
import plot_hxmt_vs_spiacs as P
import pubstyle
pubstyle.apply()

# ── config ──
BW = 0.001                     # 1 ms bins
BKG = (-1.5, -0.1)             # pre-burst background window
TAIL = (0.05, 0.15)           # post-saturation tail for ASIM cross-normalisation
XLIM = (-0.010, 0.150)        # display window (s)
OUT = "200415A_hxmt_vs_asim_1ms.pdf"


def binned(t, edges):
    return np.histogram(t, bins=edges)[0] / (edges[1] - edges[0])


def main():
    tz, ltt = P.compute_spiacs_time_system()
    t0 = tz + TimeDelta(5.925 + ltt, format="sec")
    print("T0:", t0.iso, file=sys.stderr)

    obs, fill, rng = P.load_hxmt(t0, 2, 2)
    asim = P.load_asim(t0, 2, 2)

    # align T0 to the burst peak (ASIM), so the figure matches the paper's T0
    # (= 2020-04-15T08:48:05.56, the peak); all windows below are peak-relative.
    _e = np.arange(-2, 2 + BW, BW)
    peak_shift = (_e[:-1] + BW / 2)[np.histogram(asim, bins=_e)[0].argmax()]
    obs, fill, asim = obs - peak_shift, fill - peak_shift, asim - peak_shift
    t0 = t0 + TimeDelta(peak_shift, format="sec")
    print(f"peak shift {peak_shift*1000:.1f} ms -> T0 {t0.iso}", file=sys.stderr)

    all_t = np.concatenate([obs, fill])
    edges = np.arange(-2, 2 + BW, BW)
    x = edges[:-1] + BW / 2
    r_obs = binned(obs, edges)
    r_all = binned(all_t, edges)
    r_asim = binned(asim, edges)

    # background (mean rate over the pre-burst window that both instruments cover)
    bm = (x >= BKG[0]) & (x < BKG[1])
    bkg_h = r_all[bm].mean()
    bkg_a = r_asim[bm].mean()
    n_obs = r_obs - bkg_h
    n_all = r_all - bkg_h
    n_asim = r_asim - bkg_a

    # ASIM cross-normalisation on the post-saturation tail
    tm = (x >= TAIL[0]) & (x < TAIL[1])
    scale = n_all[tm].sum() / n_asim[tm].sum()
    n_asim_s = n_asim * scale
    print(f"HXMT bkg={bkg_h:.0f}  ASIM bkg={bkg_a:.0f}  scale(H/A)={scale:.2f}", file=sys.stderr)
    print(f"ASIM peak(scaled)={n_asim_s.max():.3g}  HXMT+rec peak={n_all.max():.3g}", file=sys.stderr)

    # FIFO-saturation interval (union of reconstructed reset gaps)
    lo_met, hi_met = P.load_hxmt_resets()
    t0_met = P.hxmt_t0_met(t0)
    sat_lo, sat_hi = lo_met - t0_met, hi_met - t0_met

    # HXMT/ASIM ratio over the saturation window, where ASIM is significant (>5% peak)
    sm = (x >= sat_lo) & (x < sat_hi) & (n_asim_s > 0.05 * n_asim_s.max())
    ratio = n_all[sm] / n_asim_s[sm]
    mean, std = ratio.mean(), ratio.std()
    print(f"scale={scale:.2f}  ratio mean={mean:.2f} std={std:.2f}  "
          f"median={np.median(ratio):.2f}  min={ratio.min():.2f}  N={sm.sum()}  "
          f"ASIMpeak={n_asim_s.max():.2e}  HXMTpeak={n_all.max():.2e}", file=sys.stderr)

    # ── plot ──
    ms = 1000.0
    fig, ax = plt.subplots(figsize=(pubstyle.COL_W, 1.9))

    xs = x * ms
    ax.axvspan(sat_lo * ms, sat_hi * ms, color="#D62728", alpha=0.08, zorder=0)
    ax.step(xs, n_obs, where="mid", color="navy", lw=0.9, label="HXMT/HE observed")
    ax.step(xs, n_all, where="mid", color="#5b9bd5", lw=1.0,
            label=f"HXMT/HE reconstructed (+{len(fill):,})")
    ax.fill_between(xs, n_obs, n_all, step="mid", color="#5b9bd5", alpha=0.30)
    ax.step(xs, n_asim_s, where="mid", color="#7d4fd0", lw=1.0,
            label=f"ASIM/MXGS LED 50–400 keV ($\\times${scale:.1f})")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("Net count rate (counts s$^{-1}$)")
    ax.legend(loc="center right")
    ax.set_ylim(-1e5, 1.8e6)
    ax.text(0.5 * (sat_lo + sat_hi) * ms, 1.62e6, "HXMT FIFO\nsaturation",
            ha="center", va="top", fontsize=8, color="#A02030", style="italic")
    ax.set_xlabel("Time since $T_0$ (ms)")
    ax.set_xlim(XLIM[0] * ms, XLIM[1] * ms)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.19)
    fig.savefig(OUT, dpi=200)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
    print("saved", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
