#!/usr/bin/env python3
"""Slides version of paper f3 (LIS vs greedy) for the conference talk.

Same real data as plot_fig2_lis_vs_greedy.py (plots/fig2_real_ghost_window.json,
GRB 221009A Box A ds=1 window with one CRC-collision ghost), restyled for the
16:9 deck: deck palette, Chinese annotations, big fonts sized for a
0.56\\textwidth slide slot.

Output: talk-hxmt-saturation/he_f3_lis_vs_greedy.pdf
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# ---- deck palette ----
C_ACC   = "#3366A0"   # hxblue  accepted
C_ACC_E = "#1B3454"   # hxdeep  edge
C_REJ   = "#F5D0CB"   # pale red rejected fill
C_REJ_E = "#A93226"
C_GHOST = "#C6362C"   # hxred
C_GHOST_E = "#7A1212"

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
})

with open("plots/fig2_real_ghost_window.json") as f:
    DATA = json.load(f)
events = np.array(DATA["ef"], dtype=np.float64) / 1e5
accept_greedy = np.array(DATA["accept_greedy"], dtype=bool)
accept_lis = np.array(DATA["accept_lis"], dtype=bool)
GHOST_IDX = int(DATA["ghost_local_idx"])
CASCADE_FULL = int(DATA["cascade_full"])
n = len(events)
ghost_val = float(events[GHOST_IDX])
ymax = ghost_val * 1.30


def draw_panel(ax, accept, title, mode):
    for i, v in enumerate(events):
        if i == GHOST_IDX:
            color, edge, hatch = C_GHOST, C_GHOST_E, ""
        elif accept[i]:
            color, edge, hatch = C_ACC, C_ACC_E, ""
        else:
            color, edge, hatch = C_REJ, C_REJ_E, "////"
        ax.bar(i, v, width=0.78, color=color, edgecolor=edge,
               linewidth=0.9, hatch=hatch, zorder=3)

    ax.annotate("ghost\n$e\\approx 4.8\\times 10^5$",
                xy=(GHOST_IDX + 0.5, ghost_val * 0.86),
                xytext=(GHOST_IDX + 2.6, ghost_val * 0.88),
                ha="left", va="center", fontsize=8.5, fontweight="bold",
                color=C_GHOST_E,
                arrowprops=dict(arrowstyle="->", color=C_GHOST_E, lw=1.0))

    if mode == "greedy":
        ax.axhline(ghost_val, color=C_GHOST_E, ls="--", lw=1.3,
                   zorder=2, alpha=0.75)
        ax.text(n - 0.4, ghost_val + ymax * 0.015, "被 ghost 抬高的贪心阈值",
                fontsize=8.5, color=C_GHOST_E, va="bottom", ha="right",
                fontweight="bold")
        rej = np.where(~accept & (np.arange(n) != GHOST_IDX))[0]
        lo, hi = rej.min(), rej.max()
        ax.annotate("", xy=(hi + 0.4, ymax * 0.30),
                    xytext=(lo - 0.4, ymax * 0.30),
                    arrowprops=dict(arrowstyle="<->", color=C_REJ_E, lw=1.5))
        ax.text((lo + hi) / 2, ymax * 0.325,
                f"级联误杀：后续 {hi - lo + 1} 个全拒",
                ha="center", va="bottom", fontsize=9,
                color=C_REJ_E, fontweight="bold")
    else:
        ax.annotate("被孤立\n(长度-1)",
                    xy=(GHOST_IDX - 0.35, ghost_val - ymax * 0.05),
                    xytext=(GHOST_IDX - 2.9, ghost_val * 0.80),
                    ha="right", va="center", fontsize=8.5,
                    color=C_GHOST_E, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_GHOST_E, lw=1.1))
        acc = np.where(accept & (np.arange(n) != GHOST_IDX))[0]
        post = acc[acc > GHOST_IDX]
        lo, hi = post.min(), post.max()
        ax.annotate("", xy=(hi + 0.4, ymax * 0.30),
                    xytext=(lo - 0.4, ymax * 0.30),
                    arrowprops=dict(arrowstyle="<->", color=C_ACC_E, lw=1.5))
        ax.text((lo + hi) / 2, ymax * 0.325,
                f"{hi - lo + 1} 个全接受，零级联",
                ha="center", va="bottom", fontsize=9,
                color=C_ACC_E, fontweight="bold")

    ax.set_xticks(np.arange(0, n, 4))
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(r"$e_i$ [$10^5$ tick]", fontsize=9)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")
    ax.grid(alpha=0.18, axis="y", zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def main():
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(3.35, 2.95), sharex=True,
                                   gridspec_kw={"hspace": 0.14})
    draw_panel(axT, accept_greedy, "贪心", mode="greedy")
    draw_panel(axB, accept_lis, "LIS", mode="lis")
    axB.set_xlabel("窗口内事件序号（文件序）", fontsize=9)

    legend_elements = [
        Patch(facecolor=C_ACC, edgecolor=C_ACC_E, label="接受"),
        Patch(facecolor=C_REJ, edgecolor=C_REJ_E, hatch="////", label="误杀"),
        Patch(facecolor=C_GHOST, edgecolor=C_GHOST_E, label="ghost"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.54, 1.005),
               handlelength=1.2, columnspacing=1.2)

    fig.subplots_adjust(top=0.925, bottom=0.13, left=0.145, right=0.985)
    out = "/Users/skyair/Developer/ihep/talk-hxmt-saturation/he_f3_lis_vs_greedy.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}  (cascade_full={CASCADE_FULL})")


if __name__ == "__main__":
    main()
