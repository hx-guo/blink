#!/usr/bin/env python3
"""0x5A start-marker leakage rate vs saturation strength (SEE pointer corruption)."""
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRBS = [
    ("221009A", "/tmp/diag_221009a.csv", 339945422.0, "#D62728"),
    ("260226A", "/tmp/diag_260226a.csv", 446726273.0, "#2CA02C"),
    ("200415A", "/tmp/diag_200415a.csv", 261564488.0, "#9467BD"),
]
SLOTS_PER_PKT = 109


def load(path, t0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                lo = float(r["met_min"])
                hi = float(r["met_max"])
            except ValueError:
                continue
            if math.isnan(lo) or math.isnan(hi):
                continue
            n_5a = int(r["n_0x5a"])
            n_evt = int(r["n_evt"])
            span = hi - lo
            if span <= 0 or span > 1.5:
                continue
            rows.append({
                "box": r["box"],
                "t": (lo + hi) / 2 - t0,
                "rate": n_evt / span,
                "pct_5a": 100.0 * n_5a / SLOTS_PER_PKT,
                "n_5a": n_5a,
            })
    return rows


fig, axes = plt.subplots(2, 1, figsize=(10, 6.0))

# Top: time series for 221009A — event rate + 0x5A occurrence per 1s bin
ax1 = axes[0]
rows = [r for r in load(GRBS[0][1], GRBS[0][2]) if r["box"] == "A"]
t = np.array([r["t"] for r in rows])
rate = np.array([r["rate"] for r in rows])
pct5a = np.array([r["pct_5a"] for r in rows])
order = np.argsort(t)
t, rate, pct5a = t[order], rate[order], pct5a[order]

bin_w = 1.0
edges = np.arange(t.min(), t.max() + bin_w, bin_w)
idx = np.digitize(t, edges) - 1
n_bins = len(edges) - 1
rate_med = np.array([np.median(rate[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
pct5a_med = np.array([np.median(pct5a[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
pct5a_max = np.array([np.max(pct5a[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
centers = (edges[:-1] + edges[1:]) / 2

ax1.plot(centers, rate_med / 1000, color="#184E95", lw=1.0, label="Event rate (kHz)")
ax1.fill_between(centers, 0, rate_med / 1000, color="#184E95", alpha=0.15)
ax1.set_ylabel("Event rate (kHz)", fontsize=10, color="#184E95")
ax1.tick_params(axis="y", labelcolor="#184E95")
ax1.grid(True, alpha=0.3)

ax1b = ax1.twinx()
ax1b.fill_between(centers, 0, pct5a_max, color="#D97757", alpha=0.25, label="peak per pkt")
ax1b.plot(centers, pct5a_med, color="#C6613F", lw=1.2, label="median per pkt")
ax1b.set_ylabel("0x5A in byte[0] (%)", fontsize=10, color="#C6613F")
ax1b.tick_params(axis="y", labelcolor="#C6613F")
ax1b.set_ylim(0, max(8, np.nanmax(pct5a_max) * 1.05))
ax1b.legend(loc="upper right", fontsize=9, framealpha=0.9)

ax1.set_xlabel("T - T_trigger (s)", fontsize=10)
ax1.set_title("GRB 221009A Box A — event rate vs 0x5A misalignment rate (1s bins)", fontsize=11)

# Bottom: aggregate baseline vs saturated comparison for all 3 GRBs (per-Box)
ax2 = axes[1]
labels = []
baseline = []
saturated = []
colors = []

# Identify "saturation" packets as those where event rate > 30 kHz (heuristic)
RATE_HIGH = 30000.0
RATE_LOW = 10000.0
for name, path, t0, color in GRBS:
    rows = load(path, t0)
    for box in ["A", "B", "C"]:
        box_rows = [r for r in rows if r["box"] == box]
        if not box_rows:
            continue
        lo_pct = [r["pct_5a"] for r in box_rows if r["rate"] < RATE_LOW]
        hi_pct = [r["pct_5a"] for r in box_rows if r["rate"] > RATE_HIGH]
        if not lo_pct or not hi_pct:
            continue
        labels.append(f"{name}\nBox {box}")
        baseline.append(np.mean(lo_pct))
        saturated.append(np.mean(hi_pct))
        colors.append(color)

x = np.arange(len(labels))
w = 0.38
ax2.bar(x - w/2, baseline, w, color="#999", alpha=0.7, label=f"Baseline (rate < {int(RATE_LOW/1000)} kHz)")
ax2.bar(x + w/2, saturated, w, color=colors, alpha=0.85, label=f"Saturated (rate > {int(RATE_HIGH/1000)} kHz)")

# Annotate values on bars
for i, (b, s) in enumerate(zip(baseline, saturated)):
    ax2.text(i - w/2, b, f"{b:.2f}%", ha="center", va="bottom", fontsize=8)
    ax2.text(i + w/2, s, f"{s:.2f}%", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Mean 0x5A rate (%)", fontsize=10)
ax2.set_title("0x5A misalignment: baseline vs saturated (mean across packets)", fontsize=11)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
out = "/Users/skyair/Developer/ihep/blink/slides/fig_0x5a_misalignment.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
