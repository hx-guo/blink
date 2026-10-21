#!/usr/bin/env python3
"""Figure for the energy-band gap-fill prototype validation.

Three panels:
  top row  — one injected 300 ms gap on the rising edge of 260226A:
             per-band 10 ms light curves, ground truth vs M1 recovery
  bottom L — per-band relative-error mean±std for M1/M2/M3 (D=100 ms)
             against the ground-truth Poisson floor
  bottom R — at the 223 real reset gaps: ECDF of the in-gap spectral
             mismatch |p_ref - q_cal|/p_ref that a time-independent-
             spectrum fill (M3) would commit

Usage: python3 scripts/saturation/plot_proto_eband.py [-o out.png]
Requires data/proto_eband_results{,_real_resets}.csv from
scripts/saturation/proto_eband_gapfill.py and the data/pack_260226a pack.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from proto_eband_gapfill import (  # noqa: E402
    BAND_EDGES, BOXES, CALIB_HALF, N_BANDS, T0, band_counts, load_box,
    load_resets,
)

# palette: bands = categorical slots 1-4, methods = slots 5/6/8
BAND_C = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
METHOD_C = {"M1": "#4a3aa7", "M2": "#e34948", "M3": "#eb6834"}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

BAND_LABEL = [f"ch {BAND_EDGES[j]}–{BAND_EDGES[j+1]}"
              for j in range(N_BANDS)]

EX_T0_REL, EX_DUR = 23.5, 0.300   # example injection (clean, rising edge)
PLOT_BIN = 0.010


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def panel_example(axs, pack: Path):
    boxes = {b: load_box(pack, b) for b in BOXES}
    resets = [iv for b in BOXES for iv in load_resets(pack, b)]
    g_lo, g_hi = T0 + EX_T0_REL, T0 + EX_T0_REL + EX_DUR
    assert all(not (s < g_hi + CALIB_HALF and e > g_lo - CALIB_HALF)
               for s, e in resets), "example window not clean"

    tgt, refs = boxes["a"], [boxes["b"], boxes["c"]]
    windows = ((g_lo - CALIB_HALF, g_lo), (g_hi, g_hi + CALIB_HALF))
    t_cal = sum(band_counts(*tgt, lo, hi) for lo, hi in windows)

    p_lo, p_hi = g_lo - 0.35, g_hi + 0.35
    edges = np.arange(p_lo, p_hi + 1e-9, PLOT_BIN)
    mids = (edges[:-1] + edges[1:]) / 2

    for j, ax in enumerate(axs):
        m_t, c_t = tgt
        sel = (c_t >= BAND_EDGES[j]) & (c_t < BAND_EDGES[j + 1])
        truth = np.histogram(m_t[sel], bins=edges)[0] / PLOT_BIN

        rec = np.zeros(len(mids))
        for r_met, r_ch in refs:
            r_cal = sum(band_counts(r_met, r_ch, lo, hi) for lo, hi in windows)
            k = t_cal[j] / r_cal[j]
            rs = (r_ch >= BAND_EDGES[j]) & (r_ch < BAND_EDGES[j + 1])
            rec += k * np.histogram(r_met[rs], bins=edges)[0] / PLOT_BIN
        rec /= len(refs)

        in_gap = (mids >= g_lo) & (mids < g_hi)
        ax.axvspan(g_lo - T0, g_hi - T0, color=GRID, alpha=0.45, lw=0)
        ax.step(mids - T0, truth, where="mid", color=MUTED, lw=1.0)
        ax.step(mids[in_gap] - T0, rec[in_gap], where="mid",
                color=BAND_C[j], lw=2.0)
        style(ax)
        ax.text(0.03, 0.94, BAND_LABEL[j], transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="top", fontweight="bold")
        ax.set_xlim(p_lo - T0, p_hi - T0)
        ax.set_ylim(0, None)
        if j == 0:
            ax.set_ylabel("rate [evt/s]", fontsize=8.5, color=INK2)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(f"$t - T_0$ [s]", fontsize=8.5, color=INK2)


def panel_bias(ax, results_csv: Path):
    df = pd.read_csv(results_csv)
    df = df[(df.dur_ms == 100) & (df.truth > 0)]
    xs = np.arange(N_BANDS)
    off = {"M1": -0.22, "M2": 0.0, "M3": 0.22}

    for j in xs:
        s = df[df.band == j]
        floor = (1.0 / np.sqrt(s.truth)).mean() * 100
        ax.plot([j - 0.36, j + 0.36], [floor, floor], color=AXIS, lw=1.0)
        ax.plot([j - 0.36, j + 0.36], [-floor, -floor], color=AXIS, lw=1.0)

    ax.axhline(0, color=AXIS, lw=0.8)
    for m, dx in off.items():
        e = [((df[(df.band == j)][m] - df[df.band == j].truth)
              / df[df.band == j].truth * 100) for j in xs]
        ax.errorbar(xs + dx, [v.mean() for v in e],
                    yerr=[v.std() for v in e], fmt="o", ms=4.5,
                    color=METHOD_C[m], lw=1.8, capsize=2.5, label=m)
    style(ax)
    ax.set_xticks(xs, BAND_LABEL, fontsize=8)
    ax.set_ylabel("per-band $N$ rel. error [%]", fontsize=8.5, color=INK2)
    ax.text(0.02, 0.03,
            "gray bars: ground-truth Poisson floor  (D = 100 ms, "
            "450 injections/method)",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    leg = ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3,
                    handletextpad=0.4, columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(INK2)


def panel_real_resets(ax, rr_csv: Path):
    rr = pd.read_csv(rr_csv)
    rr["mis"] = (rr.p_ref - rr.q_cal).abs() / rr.p_ref * 100
    for j in range(N_BANDS):
        v = np.sort(rr[rr.band == j].mis.values)
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=BAND_C[j], lw=2.0,
                label=BAND_LABEL[j])
        med = np.median(v)
        ax.plot(med, 0.5, "o", ms=5, color=BAND_C[j],
                mec=SURFACE, mew=1.2, zorder=5)
    ax.axhline(0.5, color=AXIS, lw=0.8, ls=":")
    style(ax)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1)
    ax.set_xlabel("in-gap spectral mismatch "
                  r"$|p_{\rm ref}-q_{\rm cal}|/p_{\rm ref}$ [%]",
                  fontsize=8.5, color=INK2)
    ax.set_ylabel("fraction of real resets", fontsize=8.5, color=INK2)
    med1 = np.median(rr[rr.band == 0].mis)
    ax.annotate(f"{BAND_LABEL[0]}: median {med1:.0f}%",
                xy=(med1, 0.5), xytext=(med1 + 6, 0.32),
                fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    leg = ax.legend(loc="lower right", fontsize=7.5, frameon=False,
                    handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=Path("data/pack_260226a"))
    ap.add_argument("--results", type=Path,
                    default=Path("data/proto_eband_results.csv"))
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("proto_eband_gapfill.png"))
    args = ap.parse_args()

    fig = plt.figure(figsize=(10.5, 7.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25],
                          hspace=0.42, wspace=0.14,
                          left=0.07, right=0.97, top=0.90, bottom=0.09)
    axs_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[1, 2:])

    panel_example(axs_top, args.pack)
    panel_bias(ax_b, args.results)
    panel_real_resets(
        ax_c, args.results.with_name(args.results.stem + "_real_resets.csv"))

    fig.suptitle("Energy-band gap-fill prototype — GRB 260226A synthetic-gap "
                 "validation", fontsize=11.5, color=INK, x=0.07, ha="left",
                 fontweight="bold")
    fig.text(0.07, 0.925,
             "top: injected 300 ms gap at $T_0$+23.5 s, Box A recovered "
             "from B+C — truth (gray) vs per-band recovery (color)",
             fontsize=8.5, color=INK2)

    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
