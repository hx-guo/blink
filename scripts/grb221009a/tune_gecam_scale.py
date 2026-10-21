#!/usr/bin/env python3
"""Interactive scale-tuning tool for GRB 221009A main-pulse figure.

Loads the same HXMT/GECAM data as plot_hxmt_vs_gecam.py and displays a
matplotlib window with a slider that controls the GECAM scale factor in
real time.  Useful for picking the visually-best scale when no single
fitting window is uncontroversial (e.g. when both instruments have
different effective area, energy band, dead-time).

Run:
    python3 scripts/grb221009a/tune_gecam_scale.py

Controls:
    - Drag the slider at the bottom to change the scale factor (1..1000).
    - Press 'p' to print the current scale to stdout.
    - Press 's' to save the figure at the current scale.
    - Close the window to exit.
"""
import sys
from pathlib import Path
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gecam",)]  # sibling and shared script modules
from plot_hxmt_vs_gecam import (
    HXMT_TRIGGER_UTC, GECAM_BTIME, GECAM_BTIME_BKG,
    compute_time_offset, load_gecam_btime,
)

CACHE = "data/cache_221009a_reconstruct.csv"
TRIGGER_MET = 339945423.0   # astropy MET for 2022-10-09T13:17:00 UTC
T_LO, T_HI = 170, 310       # visible window
SCALE_INIT = 369.7          # starting scale value
SCALE_MIN, SCALE_MAX = 10, 1000
SAVE_OUTPUT = "/Users/skyair/Developer/ihep/paper-hxmt-saturation/figures/f10_221009_too_bright.pdf"


def main():
    matplotlib.use("MacOSX")
    # Disable matplotlib's own 'p'/'s' so our keys work cleanly
    for k in list(matplotlib.rcParams.keys()):
        if k.startswith("keymap."):
            matplotlib.rcParams[k] = []

    # --- HXMT cache ---
    obs_t, fill_t = [], []
    with open(CACHE) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t_rel = float(row[2]) - TRIGGER_MET
            (obs_t if row[1] == "EVT" else fill_t).append(t_rel)
    obs_t = np.array(obs_t); fill_t = np.array(fill_t)
    all_t = np.concatenate([obs_t, fill_t])
    edges = np.arange(-50, 701, 1.0)
    x = edges[:-1]
    r_obs = np.histogram(obs_t, bins=edges)[0]
    r_all = np.histogram(all_t, bins=edges)[0]
    bkg_far = r_all[x < -10].mean()
    net_hxmt_obs = r_obs - bkg_far
    net_hxmt_all = r_all - bkg_far
    print(f"HXMT bkg (T<-10): {bkg_far:.0f} evt/s", file=sys.stderr)

    # --- GECAM LG ---
    g_met, _, _ = compute_time_offset()
    g_x, g_rate, ch_label = load_gecam_btime(
        g_met, before=50, after=700, bin_w=1.0,
        channels="lg", subtract_revisited=True)
    print(f"GECAM peak (raw): {np.nanmax(g_rate):.0f} cts/s", file=sys.stderr)

    # --- Figure ---
    fig, (ax_lc, ax_r) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05,
                     "left": 0.07, "right": 0.97, "bottom": 0.16, "top": 0.92})

    ax_lc.step(x, net_hxmt_obs, where="post", color="C0", lw=0.8,
               alpha=0.55, label="HXMT/HE observed")
    ax_lc.fill_between(x, 0, np.nan_to_num(net_hxmt_all), step="post",
                        color="C0", alpha=0.30, edgecolor="none")
    ax_lc.step(x, net_hxmt_all, where="post", color="C0", lw=1.0,
               label=f"HXMT/HE + reconstructed")
    g_line, = ax_lc.step(g_x, g_rate * SCALE_INIT, where="post",
                          color="C1", lw=1.2,
                          label=f"GECAM-C GRD01 LG (×{SCALE_INIT:.0f})")
    ax_lc.set_xlim(T_LO, T_HI); ax_lc.axhline(0, color="gray", lw=0.5, ls="--")
    ax_lc.set_ylabel("Net count rate (evt/s)")
    ax_lc.legend(loc="upper right", fontsize=9)

    # ratio panel
    sig = (net_hxmt_all > 1e4) & (g_rate > 50)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_init = np.where(sig, net_hxmt_all / (g_rate * SCALE_INIT), np.nan)
    r_line, = ax_r.step(x, ratio_init, where="post", color="C1", lw=1.0)
    ax_r.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax_r.set_ylim(0, 2.0)
    ax_r.set_ylabel("HXMT/GECAM"); ax_r.set_xlabel("Time since trigger (s)")
    ratio_txt = ax_r.text(0.99, 0.93, "", transform=ax_r.transAxes, ha="right",
                           va="top", fontsize=9, family="monospace",
                           bbox=dict(facecolor="white", alpha=0.9))

    # --- slider ---
    ax_slider = plt.axes([0.18, 0.04, 0.65, 0.03])
    sl = Slider(ax_slider, "GECAM ×", SCALE_MIN, SCALE_MAX,
                valinit=SCALE_INIT, valstep=1)

    def update(val):
        s = sl.val
        g_line.set_ydata(g_rate * s)
        g_line.set_label(f"GECAM-C GRD01 LG (×{s:.0f})")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sig, net_hxmt_all / (g_rate * s), np.nan)
        r_line.set_ydata(ratio)
        ratio_in_window = ratio[(x >= T_LO) & (x < T_HI)]
        valid = np.isfinite(ratio_in_window)
        if valid.any():
            med = float(np.nanmedian(ratio_in_window[valid]))
            q75, q25 = np.nanpercentile(ratio_in_window[valid], [75, 25])
            ratio_txt.set_text(
                f"scale = ×{s:.0f}\n"
                f"HXMT/GECAM = {med:.2f} ± {(q75-q25)/1.349:.2f} ({valid.sum()} bins)")
        ax_lc.legend(loc="upper right", fontsize=9)
        fig.canvas.draw_idle()

    sl.on_changed(update)
    update(SCALE_INIT)

    def on_key(event):
        if event.key == "p":
            print(f"current scale = {sl.val:.1f}")
        elif event.key == "s":
            fig.savefig(SAVE_OUTPUT, dpi=150)
            print(f"saved to {SAVE_OUTPUT}")
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    print(f"final scale = {sl.val:.1f}")


if __name__ == "__main__":
    main()
