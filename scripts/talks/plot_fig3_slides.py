#!/usr/bin/env python3
"""Slides version of paper f5 (cross-box wrap uniqueness) for the talk.

Same data pipeline as plot_fig3_wrap_uniqueness.py (rust CLI extract on
local data/1B, GRB 221009A Box A/B), restyled for the deck: Chinese
labels, deck palette, big fonts, and an explicit +1.05 s arrow at the
burst onset so the misalignment is visible at a glance.

Output: talk-hxmt-saturation/he_f5_wrap_uniqueness.pdf
"""
import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

TRIGGER = "2022-10-09T13:17:00"
BEFORE, AFTER = 10.0, 280.0
ZOOM_LO, ZOOM_HI = 248.0, 271.5
BIN = 0.25
WRAP_PERIOD = 1.048576
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
})

C_B  = "#E89028"   # Box B reference (hxorange family)
C_J0 = "#1B3454"   # hxdeep, chosen
C_J1 = "#7FB0E0"   # light blue, alternative


def parse_met(s):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (dt - MET_EPOCH).total_seconds()


def run_cli(box):
    cmd = ["./target/release/blink", "sat", "extract",
           "--before", str(BEFORE), "--after", str(AFTER),
           "--box", box, "--source", "1b", TRIGGER]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    mets = []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 3 or parts[0] == "box":
            continue
        if parts[1] == "SEC":
            continue
        mets.append(float(parts[2]))
    return np.array(mets)


def main():
    t_ref = parse_met(TRIGGER)
    a_evts = run_cli("a")
    b_evts = run_cli("b")
    print(f"  Box A {len(a_evts):,} events, Box B {len(b_evts):,}",
          file=sys.stderr)

    edges = np.arange(t_ref + ZOOM_LO, t_ref + ZOOM_HI + BIN / 2, BIN)
    x = edges[:-1] - t_ref

    a_j0 = a_evts[(a_evts >= edges[0]) & (a_evts < edges[-1])]
    a_j1 = a_evts + WRAP_PERIOD
    a_j1 = a_j1[(a_j1 >= edges[0]) & (a_j1 < edges[-1])]
    b_win = b_evts[(b_evts >= edges[0]) & (b_evts < edges[-1])]

    rate_a0 = np.histogram(a_j0, bins=edges)[0] / BIN
    rate_a1 = np.histogram(a_j1, bins=edges)[0] / BIN
    rate_b = np.histogram(b_win, bins=edges)[0] / BIN

    fig, ax = plt.subplots(figsize=(3.5, 2.85))

    ax.step(x, rate_b, where="post", color=C_B, lw=1.3,
            label="Box B（跨机箱参考）", zorder=3)
    ax.step(x, rate_a0, where="post", color=C_J0, lw=1.5,
            label="Box A 绕圈 $j=0$（选定）", zorder=4)
    ax.step(x, rate_a1, where="post", color=C_J1, lw=1.3,
            label="Box A 绕圈 $j=1$（$+1.05$ s）", zorder=2)

    # onset misalignment arrow: j=0 rise vs j=1 rise
    thresh = 4000.0
    r0 = x[np.argmax(rate_a0 > thresh)]
    r1 = x[np.argmax(rate_a1 > thresh)]
    y_arrow = 6600
    ax.annotate("", xy=(r1, y_arrow), xytext=(r0, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="#C6362C", lw=2.2))
    ax.text(r1 + 0.55, y_arrow, "错一个绕圈周期：$\\mathbf{+1.05}$ s",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color="#C6362C")

    ax.set_xlabel(r"时间 $-\ T_0$ (s)　[bin = 250 ms]", fontsize=9)
    ax.set_ylabel("计数率 (c/s)", fontsize=9)
    ax.set_xlim(ZOOM_LO, ZOOM_HI)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower center", fontsize=8, frameon=True,
              framealpha=0.9, bbox_to_anchor=(0.56, 0.02))

    plt.tight_layout(pad=0.4)
    out = "/Users/skyair/Developer/ihep/talk-hxmt-saturation/he_f5_wrap_uniqueness.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
