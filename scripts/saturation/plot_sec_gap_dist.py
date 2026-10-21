#!/usr/bin/env python3
"""Δstime distribution between consecutive SECs (= candidate wrap count per event)."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRBS = [
    ("221009A", "/tmp/sec_221009a.csv", "#D62728"),
    ("260226A", "/tmp/sec_260226a.csv", "#2CA02C"),
    ("200415A", "/tmp/sec_200415a.csv", "#9467BD"),
]


def load(path):
    """Returns dict box -> sorted list of SEC METs."""
    by_box = {"A": [], "B": [], "C": []}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            box, met = parts[0], float(parts[1])
            if box in by_box:
                by_box[box].append(met)
    for k in by_box:
        by_box[k].sort()
    return by_box


fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.6))

# Histogram of Δstime values across all GRBs
all_dstime = {name: [] for name, _, _ in GRBS}
worst = {}
for name, path, color in GRBS:
    by_box = load(path)
    for box, secs in by_box.items():
        if len(secs) < 2:
            continue
        diffs = np.diff(secs)
        dstime = np.round(diffs).astype(int)
        dstime = dstime[(dstime >= 1) & (dstime <= 50)]
        all_dstime[name].extend(dstime)
    if all_dstime[name]:
        worst[name] = max(all_dstime[name])

bins = np.arange(0.5, 22.5, 1)
for name, color in [(g[0], g[2]) for g in GRBS]:
    data = all_dstime[name]
    label = f"GRB {name} (n={len(data)} pairs, max Δ={worst.get(name, 0)})"
    ax1.hist(data, bins=bins, color=color, alpha=0.65, edgecolor="white",
             linewidth=0.6, label=label)

ax1.set_yscale("log")
ax1.set_xlabel("Δstime (s) between consecutive valid SECs  =  # candidate wraps per event", fontsize=10)
ax1.set_ylabel("# SEC pairs (log)", fontsize=10)
ax1.set_title("SEC gap distribution across 3 GRBs (all 3 Boxes)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 22))

fig.tight_layout()
out = "/Users/skyair/Developer/ihep/blink/slides/fig_sec_gap_dist.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
