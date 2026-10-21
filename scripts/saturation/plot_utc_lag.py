#!/usr/bin/env python3
"""UTC tail (MCU pack time) vs latest event MET — shows packing-time lag during saturation."""
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
SLOTS = 109


def load(path, t0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                hi = float(r["met_max"])
                utc = float(r["utc_tail"])
            except ValueError:
                continue
            if math.isnan(hi):
                continue
            n_evt = int(r["n_evt"])
            try:
                lo = float(r["met_min"])
            except ValueError:
                continue
            span = hi - lo
            if span <= 0 or span > 30:
                continue
            # MET_CORRECTION = 4.0s is added to event MET in the CLI output but not
            # to UTC tail. Subtract to get the true MCU pack-time lag.
            rows.append({
                "box": r["box"],
                "t": (lo + hi) / 2 - t0,
                "rate": n_evt / span,
                "lag": utc - (hi - 4.0),
            })
    return rows


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: time series for 221009A — event rate + UTC lag per 1s bin
ax1 = axes[0]
rows = [r for r in load(GRBS[0][1], GRBS[0][2]) if r["box"] == "A"]
t = np.array([r["t"] for r in rows])
rate = np.array([r["rate"] for r in rows])
lag = np.array([r["lag"] for r in rows])
order = np.argsort(t)
t, rate, lag = t[order], rate[order], lag[order]

bin_w = 1.0
edges = np.arange(t.min(), t.max() + bin_w, bin_w)
idx = np.digitize(t, edges) - 1
n_bins = len(edges) - 1
rate_med = np.array([np.median(rate[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
lag_max = np.array([np.max(lag[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
lag_med = np.array([np.median(lag[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
centers = (edges[:-1] + edges[1:]) / 2

ax1.plot(centers, rate_med / 1000, color="#184E95", lw=1.0)
ax1.fill_between(centers, 0, rate_med / 1000, color="#184E95", alpha=0.15)
ax1.set_ylabel("Event rate (kHz)", fontsize=10, color="#184E95")
ax1.tick_params(axis="y", labelcolor="#184E95")
ax1.grid(True, alpha=0.3)

ax1b = ax1.twinx()
ax1b.fill_between(centers, 0, lag_max, color="#D97757", alpha=0.25, label="peak lag")
ax1b.plot(centers, lag_med, color="#C6613F", lw=1.0, label="median lag")
ax1b.set_ylabel("UTC tail − event MET (s)", fontsize=10, color="#C6613F")
ax1b.tick_params(axis="y", labelcolor="#C6613F")
ax1b.set_ylim(0, max(2, np.nanmax(lag_max) * 1.05))
ax1b.legend(loc="upper right", fontsize=9, framealpha=0.9)

ax1.set_xlabel("T - T_trigger (s)", fontsize=10)
ax1.set_title("GRB 221009A Box A — event rate vs UTC tail lag (1s bins)", fontsize=11)

# Right: histogram of UTC lag values (all 3 GRBs, log-y)
ax2 = axes[1]
bin_edges = np.linspace(-1, 16, 60)
for name, path, t0, color in GRBS:
    rows = load(path, t0)
    lag_arr = np.array([r["lag"] for r in rows])
    ax2.hist(lag_arr, bins=bin_edges, color=color, alpha=0.55,
             edgecolor="white", linewidth=0.4, label=f"GRB {name} (n={len(lag_arr)})")

ax2.axvline(0, color="#888", linestyle="-", lw=0.8, alpha=0.5)
ax2.set_yscale("log")
ax2.set_xlabel("UTC tail − event MET (s)", fontsize=10)
ax2.set_ylabel("# packets (log)", fontsize=10)
ax2.set_title("UTC lag distribution (all 3 GRBs)", fontsize=11)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Stats annotation
total_lag_gt_1 = 0
total_pkts = 0
peak_lag = 0
for name, path, t0, color in GRBS:
    rows = load(path, t0)
    lag_arr = np.array([r["lag"] for r in rows])
    total_lag_gt_1 += np.sum(lag_arr > 1.0)
    total_pkts += len(lag_arr)
    peak_lag = max(peak_lag, lag_arr.max() if len(lag_arr) else 0)
ax2.text(0.65, 0.55,
         f"lag > 1s: {total_lag_gt_1:,}/{total_pkts:,}\n"
         f"  ({100*total_lag_gt_1/total_pkts:.1f}% of packets)\n"
         f"peak lag: {peak_lag:.1f}s",
         transform=ax2.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#ccc"))

fig.tight_layout()
out = "/Users/skyair/Developer/ihep/blink/slides/fig_utc_lag.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
