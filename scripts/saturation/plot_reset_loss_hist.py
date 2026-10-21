#!/usr/bin/env python3
"""每次 FIFO reset 的事件丢失分布直方图。"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRBS = [
    ("200415A", "/tmp/detect_200415a.csv", "#9467BD"),
    ("260226A", "/tmp/detect_260226a.csv", "#2CA02C"),
    ("221009A", "/tmp/detect_221009a.csv", "#D62728"),
]
FIFO_CAP = 455


def load(path):
    with open(path) as f:
        return [int(r["n_lost"]) for r in csv.DictReader(f)]


fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
xmax = 3200
bins = np.linspace(0, xmax, 65)

for ax, (name, path, color) in zip(axes, GRBS):
    data = np.array(load(path))
    if len(data) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        continue
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    med = np.median(data)
    ax.axvline(FIFO_CAP, color="#666", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(med, color="black", linestyle=":", linewidth=1.0)
    ax.text(med, ax.get_ylim()[1] * 0.9, f" median={int(med)}", fontsize=9, va="top")
    ax.text(0.99, 0.95,
            f"GRB {name}\nn={len(data)} resets\nmean={data.mean():.0f}, "
            f"max={data.max()}\ntotal lost={data.sum():,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"))
    ax.set_ylabel("# resets", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

axes[0].text(FIFO_CAP, axes[0].get_ylim()[1] * 0.55,
             f" FIFO cap\n  ({FIFO_CAP} evts)",
             fontsize=9, va="top", color="#444")
axes[-1].set_xlabel("Events lost per FIFO reset", fontsize=11)
axes[-1].set_xlim(0, xmax)
fig.suptitle("HXMT/HE FIFO reset: events lost per reset (all 3 boxes combined)",
             fontsize=12, y=0.995)
fig.tight_layout()
out = "/Users/skyair/Developer/ihep/blink/slides/fig_reset_loss_hist.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
