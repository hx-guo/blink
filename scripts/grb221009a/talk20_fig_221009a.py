#!/usr/bin/env python3
"""Talk-20 result figure 1: GRB 221009A full burst, three traces only.

  navy      HXMT/HE observed (event level)
  sky       HXMT/HE reconstructed AND dead-time corrected
  red       GECAM-C GRD01 LG x200 (independent reference)

The engineering D/tau traces of make_f10 are deliberately dropped. The
recovered trace MUST be the dead-time-corrected one: filling the lost events
alone only reaches 0.64 of GECAM on the main pulse, both together reach 0.99.

Usage: python3 scripts/grb221009a/talk20_fig_221009a.py -o <dir>     (run from blink root)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gecam", "../lib", "../talks",)]  # sibling and shared script modules
import talk20_style as ts
from make_f10 import (load_eng_dtau, to_grid, TRIGGER_MET_ASTROPY)
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime

CACHE_PW = "data/cache_221009a_recon_pw.csv"
CSI_PW, CSI_CHMIN = (90, 257), 30     # HE CsI selection (plot_hxmt_csi_multi)
BEFORE, AFTER, BIN = 50.0, 700.0, 1.0
SAA_END = 172.0          # GECAM-C leaves the South Atlantic Anomaly
# On-board firmware entered its SAA working mode: HE detectors off, zero
# events AND zero engineering counters in all three boxes (paper Sec. 5).
# These are not saturation and are not recoverable -- mark them, never fill.
HE_OFF = [(231.0, 248.0), (269.0, 276.0)]


def load_csi(path, before, after, bin_w):
    """HE CsI-selected observed / gap-filled rates on a bin grid.

    The CsI anti-coincidence crystal is what HE actually detected GRB 221009A
    with (the burst was far off-axis and hard), so the light curve is built
    from CsI events only: pulse width in [90, 257] and channel >= 30.
    """
    import pandas as pd
    edges = np.arange(-before, after + bin_w, bin_w)
    h_obs = np.zeros(len(edges) - 1)
    h_fill = np.zeros(len(edges) - 1)
    cols = ["type", "met", "channel", "pulse_width"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=4_000_000):
        pw = chunk["pulse_width"].to_numpy()
        ch = chunk["channel"].to_numpy()
        keep = (pw >= CSI_PW[0]) & (pw <= CSI_PW[1]) & (ch >= CSI_CHMIN)
        if not keep.any():
            continue
        t = chunk["met"].to_numpy()[keep] - TRIGGER_MET_ASTROPY
        isobs = (chunk["type"].to_numpy()[keep] == "EVT")
        h_obs += np.histogram(t[isobs], bins=edges)[0]
        h_fill += np.histogram(t[~isobs], bins=edges)[0]
    x = edges[:-1]
    r_obs, r_fill = h_obs / bin_w, h_fill / bin_w
    bkg = (r_obs + r_fill)[x < -10].mean()
    return x, r_obs - bkg, r_obs + r_fill - bkg, r_fill, int(h_fill.sum()), bkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="/tmp")
    a = ap.parse_args()
    ts.apply(13)

    x, obs, allr, r_fill, n_fill, B_evt = load_csi(CACHE_PW, BEFORE, AFTER, BIN)
    et, dtau, flive, tau_us, B_eng = load_eng_dtau(BEFORE, AFTER)
    flive_g = to_grid(x, et, flive)
    with np.errstate(divide="ignore", invalid="ignore"):
        all_dt = np.where(flive_g > 0.05, (allr + B_evt) / flive_g - B_evt, np.nan)

    # Detector-off seconds first: zero events, zero fillers AND zero recorded
    # dead time, so they pass every "clean" test while GECAM is at its peak.
    # Leaving them in the scale fit wrecks it (the denominator blows up).
    off = (allr + B_evt) < 10.0
    off = off | np.r_[off[1:], False] | np.r_[False, off[:-1]]

    g_met, _, _ = compute_time_offset()
    _, g_rate, _ = load_gecam_btime(g_met, BEFORE, AFTER, BIN, "lg", True)
    g_net = np.nan_to_num(g_rate, nan=0.0)

    # Effective-area scale for GECAM-C, fitted ONLY on seconds where neither
    # correction is active: no gap fillers and live fraction > 0.95, outside
    # the SAA. The agreement on the collapsed peak is then a real test, not a
    # consequence of the normalisation.
    clean = ((x > SAA_END) & (x < AFTER) & ~off & (r_fill == 0)
             & (flive_g > 0.95) & (g_net > 50.0) & np.isfinite(obs)
             & (obs > 0))
    # median of the per-second ratio, not a ratio of sums: robust to any one
    # second where the two instruments disagree
    scale = float(np.nanmedian(obs[clean] / g_net[clean]))
    print(f"  GECAM-C scale fitted on {int(clean.sum())} clean seconds: "
          f"x{scale:.0f}", file=sys.stderr)
    gecam = g_net * scale
    GECAM_SCALE = scale

    # blank the detector-off intervals so the traces break instead of
    # plunging to the axis, which would read as "still lost"
    obs = np.where(off, np.nan, obs)
    allr_m = np.where(off, np.nan, allr)
    all_dt = np.where(off, np.nan, all_dt)

    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    ax.axvspan(-BEFORE, SAA_END, color=ts.GREY, alpha=0.13, lw=0, zorder=0)
    for lo, hi in HE_OFF:
        ax.axvspan(lo, hi, color=ts.GREY, alpha=0.22, lw=0, zorder=0)
    # GECAM: non-positive bins are blanked (symlog would plunge them to the
    # axis); inside the SAA the rate is particle background, so it is drawn
    # faint to say "present but unusable".
    g_plot = np.where(gecam > 0, gecam, np.nan)
    in_saa = x < SAA_END
    ax.step(x, np.where(in_saa, g_plot, np.nan), where="post", color=ts.RED,
            lw=1.3, alpha=0.32, zorder=2)
    ax.step(x, np.where(in_saa, np.nan, g_plot), where="post", color=ts.RED,
            lw=1.4, label=f"GECAM-C 独立看到的（×{GECAM_SCALE:.0f}）", zorder=2)
    ax.step(x, obs, where="post", color=ts.NAVY, lw=1.7,
            label="慧眼 HE 记到的", zorder=4)
    ax.step(x, all_dt, where="post", color=ts.SKY, lw=1.7,
            label="补回丢掉的事件 + 死时间修正", zorder=5)

    ax.set_yscale("symlog", linthresh=1e3)
    ytop = np.nanmax([np.nanmax(all_dt), np.nanmax(g_plot)]) * 2.8
    ax.set_ylim(0, ytop)
    ax.set_xlim(-BEFORE, AFTER)
    ax.set_xlabel("2022-10-09 触发后的秒数")
    ax.set_ylabel("净计数率（事件 / 秒）")
    ax.text(0.015, 0.975, "慧眼 HE 的 CsI，1 秒并道",
            transform=ax.transAxes, va="top", fontsize=12.5, color=ts.TICK)
    ax.annotate("GECAM-C 过南大西洋异常区\n这一段没有可用数据",
                xy=(62, 0.60), xycoords=("data", "axes fraction"),
                ha="center", va="center", fontsize=12.5,
                color="#6B6B6B", linespacing=1.5)
    ax.annotate("慧眼按南大西洋异常区模式关机\n（不是饱和，补不回来）",
                xy=(273, 0.47), xytext=(310, 0.565),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                ha="left", va="center",
                fontsize=12, color="#6B6B6B", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    pw = (x >= 236) & (x < 268)
    ts.mark(ax, 1, 252, np.nanmedian(obs[pw]) if np.isfinite(
        np.nanmedian(obs[pw])) else 2.5e4)
    ipk = int(np.nanargmax(np.where((x > SAA_END) & (x < 310), all_dt, -np.inf)))
    ts.mark(ax, 2, x[ipk] - 13, all_dt[ipk] * 1.55)
    ts.mark(ax, 3, 120, 0.255, transform=ax.get_xaxis_transform())
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              loc="upper right", fontsize=14.5, handlelength=1.5,
              labelspacing=0.55, borderaxespad=0.4)
    ts.clean(ax)
    fig.tight_layout(pad=0.4)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"talk20_221009a.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out}/talk20_221009a.{ext}")

    # main-pulse agreement, detector-off bins excluded, GECAM above its floor
    w = ((x >= 170) & (x < 310) & ~off & np.isfinite(all_dt)
         & (gecam / GECAM_SCALE > 50.0)
         & (allr > 0.01 * np.nanmax(allr)))
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_dt = np.where(flive_g > 0.05, (obs + B_evt) / flive_g - B_evt, np.nan)
    print(f"  CsI fill events: {n_fill:,}   ({int(w.sum())} main-pulse bins)")
    for lbl, arr in (("只看记到的", obs), ("只做死时间修正", obs_dt),
                     ("只补回丢掉的事件", allr_m), ("两个都做", all_dt)):
        print(f"  {lbl:18s} / GECAM = {np.nanmedian(arr[w] / gecam[w]):.2f}")


if __name__ == "__main__":
    main()
