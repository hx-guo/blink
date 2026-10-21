#!/usr/bin/env python3
"""Slides version of Fig 4 (LIS vs 1K, 3 boxes) for GRB 221009A.

Same data as plot_fig4_solve_vs_1k_221009a.py, restyled for the group-meeting
slides: wide 16:9-friendly aspect, deck colour scheme (1K = muted blue-grey
baseline, 1B-LIS = orange = our result), no suptitle (the slide has its own
title). Output goes to the talk directory, NOT the paper figures dir.
"""
import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

EPOCH    = "2022-10-09T13"
TRIGGER  = "2022-10-09T13:17:00"
BEFORE   = 10.0
AFTER    = 290.0
BIN      = 1.0
ZOOM_LO  = 250.0
ZOOM_HI  = 272.0
SHADE_C_LO = 263.0
SHADE_C_HI = 270.0
SHADE_A_LO = 251.0
SHADE_A_HI = 253.0
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

# ---- deck palette ----
C_1K_LINE = "#7C8A9E"   # muted blue-grey (1K standard = baseline)
C_1K_FILL = "#DCE2EA"
C_1B_LINE = "#D65D12"   # hxorange (1B LIS = our result)
C_1B_FILL = "#F5D6BC"
C_SHADE   = "#D65D12"   # failure-zone tint
C_ANNO    = "#8A3D0B"   # dark orange annotation text


def parse_met(s):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (dt - MET_EPOCH).total_seconds()


def run_cli(source):
    cmd = ["./target/release/blink", "sat", "extract",
           "--source", source,
           "--before", str(BEFORE), "--after", str(AFTER), TRIGGER]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    boxes = {}
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 3 or parts[0] == "box":
            continue
        box, typ, met = parts[0], parts[1], float(parts[2])
        if typ == "SEC":
            continue
        boxes.setdefault(box, []).append(met)
    return {b: np.array(v) for b, v in boxes.items()}


def main():
    t_ref = parse_met(TRIGGER)
    print("Loading 1B (LIS-reconstructed)...", file=sys.stderr)
    b1 = run_cli("1b")
    print("Loading 1K...", file=sys.stderr)
    k1 = run_cli("1k")

    edges = np.arange(t_ref + ZOOM_LO, t_ref + ZOOM_HI + BIN / 2, BIN)
    x = edges[:-1] - t_ref

    fig, axes = plt.subplots(3, 1, figsize=(11, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.14})

    for ax, box in zip(axes, ["A", "B", "C"]):
        evt_1b = b1.get(box, np.array([]))
        evt_1k = k1.get(box, np.array([]))
        win_1b = evt_1b[(evt_1b >= edges[0]) & (evt_1b < edges[-1])]
        win_1k = evt_1k[(evt_1k >= edges[0]) & (evt_1k < edges[-1])]
        r_1k = np.histogram(win_1k, bins=edges)[0] / BIN
        r_1b = np.histogram(win_1b, bins=edges)[0] / BIN

        if box == "C":
            ax.axvspan(SHADE_C_LO, SHADE_C_HI, color=C_SHADE, alpha=0.09, zorder=0)
        if box == "A":
            ax.axvspan(SHADE_A_LO, SHADE_A_HI, color=C_SHADE, alpha=0.09, zorder=0)

        ax.fill_between(x, r_1k, step="post", color=C_1K_FILL,
                        alpha=0.9, edgecolor="none", zorder=1)
        ax.step(x, r_1k, where="post", color=C_1K_LINE, lw=1.0,
                label=f"1K standard  ({len(win_1k):,})", zorder=2)
        ax.fill_between(x, r_1b, step="post", color=C_1B_FILL,
                        alpha=0.55, edgecolor="none", zorder=3)
        ax.step(x, r_1b, where="post", color=C_1B_LINE, lw=1.6,
                label=f"1B LIS, this work  ({len(win_1b):,})", zorder=4)

        ax.set_ylabel(f"Box {box}\nevt/s", fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
        ax.grid(alpha=0.13)
        ax.set_ylim(bottom=0, top=15500)
        ax.tick_params(labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if box == "C":
            ax.text((SHADE_C_LO + SHADE_C_HI) / 2, 14000,
                    r"1K $=$ 0 for ${\sim}7$ s",
                    ha="center", va="top", fontsize=10, color=C_ANNO,
                    fontweight="bold")
        if box == "A":
            ax.text((SHADE_A_LO + SHADE_A_HI) / 2 + 0.3, 14000,
                    r"1K wrap shift ${\sim}1.05$ s",
                    ha="left", va="top", fontsize=9, color=C_ANNO,
                    fontweight="bold")

    axes[-1].set_xlabel(
        r"Time $-$ $T_0$ (s)   "
        f"[$T_0=$ {TRIGGER} UTC, bin $=$ {BIN:.0f} s]", fontsize=10)
    axes[0].set_xlim(ZOOM_LO, ZOOM_HI)
    plt.tight_layout()
    out_pdf = "/Users/skyair/Developer/ihep/talk-groupmeeting/he_f4_lis_vs_1k.pdf"
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
