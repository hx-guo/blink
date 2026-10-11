#!/usr/bin/env python3
"""GRB 221009A main pulse after recovery, in the talk-20 palette.
Same window as the 'before' slide so the two read as a pair.

Usage: python3 scripts/talk20_after_221009a.py -o <dir>
"""
import argparse, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, str(Path(__file__).parent))
from make_f10 import load_hxmt, GECAM_SCALE
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime

NAVY, SKY, SAT, MUTED, AXIS = "#1B3454", "#4B86C8", "#D62728", "#828282", "#C4C9D0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="/tmp")
    ap.add_argument("--bin", type=float, default=1.0)
    a = ap.parse_args()
    for f in ["Manrope-Light.ttf", "Manrope-Regular.ttf", "Manrope-Medium.ttf"]:
        fm.fontManager.addfont("/Users/skyair/Developer/ihep/talk-20/fonts/" + f)
    H = os.path.expanduser("~/Library/Fonts/")
    for f in ["NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Light.otf"]:
        fm.fontManager.addfont(H + f)
    plt.rcParams.update({"font.family": ["Manrope", "Noto Sans CJK SC"], "font.size": 13,
                         "axes.edgecolor": AXIS, "xtick.color": "#4A4F57", "ytick.color": "#4A4F57",
                         "axes.labelcolor": "#4A4F57", "axes.unicode_minus": False})
    before, after = 50.0, 700.0
    x, edges, obs, allr, n_fill, B_evt = load_hxmt(before, after, a.bin)
    g_met, _, _ = compute_time_offset()
    _, g_rate, _ = load_gecam_btime(g_met, before, after, a.bin, "lg", True)
    gecam = np.nan_to_num(g_rate, nan=0.0) * GECAM_SCALE

    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=300)
    ax.fill_between(x, 0, np.nan_to_num(allr), step="post", color=SKY, alpha=0.22, lw=0, zorder=1)
    ax.step(x, gecam, where="post", color=SAT, lw=2.0, zorder=2)
    ax.step(x, allr, where="post", color=SKY, lw=2.4, zorder=4)
    ax.step(x, obs, where="post", color=NAVY, lw=2.0, zorder=5)
    ax.set_xlim(172, 300); ax.set_ylim(-0.28e6, 5.45e6)
    ax.set_xlabel("2022-10-09 触发后的秒数"); ax.set_ylabel("净计数率 (事件/秒)")
    ax.text(236, 4.62e6, "GECAM-C 独立看到的", color=SAT, fontsize=16, fontweight="medium")
    ax.text(236, 4.18e6, "（按有效面积归一）", color=SAT, fontsize=13)
    ax.annotate("慧眼补回后", xy=(229, 4.2e6), xytext=(184, 4.3e6), color=SKY, fontsize=17,
                fontweight="medium", arrowprops=dict(arrowstyle="-", color=SKY, lw=1.3))
    ax.annotate("慧眼记到的", xy=(200, 6e4), xytext=(182, 1.0e6), color=NAVY, fontsize=15,
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.1))
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=3)
    fig.tight_layout(pad=0.5)
    out = Path(a.out_dir)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"after_221009a.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
    print(f"filled {n_fill:,} events")


if __name__ == "__main__":
    main()
