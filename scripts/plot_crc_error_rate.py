#!/usr/bin/env python3
"""CRC error rate vs saturation strength."""
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRBS = [
    ("221009A", "/tmp/diag_221009a.csv", 339945423.0, "#D62728"),
    ("260226A", "/tmp/diag_260226a.csv", 446726273.0, "#2CA02C"),
    ("200415A", "/tmp/diag_200415a.csv", 261564488.0, "#9467BD"),
]
SLOTS_PER_PKT = 109


def load(path, t0, box_filter=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if box_filter and r["box"] != box_filter:
                continue
            try:
                lo = float(r["met_min"])
                hi = float(r["met_max"])
            except ValueError:
                continue
            if math.isnan(lo) or math.isnan(hi):
                continue
            n_err = int(r["n_err"])
            n_evt = int(r["n_evt"])
            n_sec = int(r["n_sec"])
            span = hi - lo
            if span <= 0 or span > 1.5:
                continue
            rows.append({
                "box": r["box"],
                "t": (lo + hi) / 2 - t0,
                "rate": n_evt / span,
                "crc_pct": 100.0 * n_err / SLOTS_PER_PKT,
            })
    return rows


fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=False,
                          gridspec_kw={"height_ratios": [1, 1]})

# Panel 1: time series for 221009A (Box A only) — event rate vs peak CRC error rate per 1s bin
ax1 = axes[0]
rows = [r for r in load(GRBS[0][1], GRBS[0][2]) if r["box"] == "A"]
t = np.array([r["t"] for r in rows])
rate = np.array([r["rate"] for r in rows])
crc = np.array([r["crc_pct"] for r in rows])
order = np.argsort(t)
t, rate, crc = t[order], rate[order], crc[order]

bin_w = 1.0
edges = np.arange(t.min(), t.max() + bin_w, bin_w)
idx = np.digitize(t, edges) - 1
n_bins = len(edges) - 1
rate_med = np.array([np.median(rate[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
crc_max = np.array([np.max(crc[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
crc_p90 = np.array([np.percentile(crc[idx == i], 90) if (idx == i).any() else np.nan for i in range(n_bins)])
centers = (edges[:-1] + edges[1:]) / 2

ax1.plot(centers, rate_med / 1000, color="#184E95", lw=1.0, label="Event rate (kHz)")
ax1.fill_between(centers, 0, rate_med / 1000, color="#184E95", alpha=0.15)
ax1.axhline(15.6, color="#888", linestyle="--", lw=1, alpha=0.7)
ax1.text(t.max() * 0.99, 15.6, " MCU read ceiling ~15.6 kHz",
         ha="right", va="bottom", fontsize=9, color="#555")
ax1.set_ylabel("Event rate (kHz)", fontsize=10, color="#184E95")
ax1.tick_params(axis="y", labelcolor="#184E95")
ax1.grid(True, alpha=0.3)

# CRC on log scale — replace 0 with NaN to avoid log(0)
crc_max_log = np.where(crc_max > 0, crc_max, np.nan)
crc_p90_log = np.where(crc_p90 > 0, crc_p90, np.nan)

ax1b = ax1.twinx()
ax1b.plot(centers, crc_max_log, color="#C6613F", lw=1.0, alpha=0.85, label="Peak CRC fail %")
ax1b.scatter(centers, crc_max_log, color="#C6613F", s=5, alpha=0.6)
ax1b.set_ylabel("CRC error rate per packet (%, log)", fontsize=10, color="#C6613F")
ax1b.tick_params(axis="y", labelcolor="#C6613F")
ax1b.set_yscale("log")
ax1b.set_ylim(0.5, 100)

ax1.set_xlabel("T - T_trigger (s)", fontsize=10)
ax1.set_title("GRB 221009A Box A — event rate vs CRC error rate (1s bins)", fontsize=11)

# Panel 2: binned correlation — fraction of packets with CRC errors vs event rate
ax2 = axes[1]
edges_kHz = np.array([0, 2, 5, 10, 15, 20, 30, 50, 100, 200])
edges = edges_kHz * 1000
centers_kHz = (edges_kHz[:-1] + edges_kHz[1:]) / 2

for name, path, t0, color in GRBS:
    rows = load(path, t0)
    rate_arr = np.array([r["rate"] for r in rows])
    crc_arr = np.array([r["crc_pct"] for r in rows])
    fracs = []
    for i in range(len(edges) - 1):
        m = (rate_arr >= edges[i]) & (rate_arr < edges[i+1])
        if m.sum() < 10:
            fracs.append(np.nan)
        else:
            fracs.append((crc_arr[m] > 0).sum() / m.sum() * 100)
    fracs = np.array(fracs)
    # connect through NaN by plotting only finite values
    valid = np.isfinite(fracs)
    ax2.plot(centers_kHz[valid], fracs[valid], color=color, lw=1.8, marker="o", ms=6,
             label=f"GRB {name}", alpha=0.85)

ax2.axvline(15.6, color="#888", linestyle="--", lw=1, alpha=0.7)
ax2.text(15.6, 95, " MCU ceiling", ha="left", va="top", fontsize=9, color="#555")
ax2.set_xlabel("Event rate per packet (kHz)", fontsize=10)
ax2.set_ylabel("% of packets with ≥1 CRC error", fontsize=10)
ax2.set_xlim(0, 200)
ax2.set_ylim(-3, 105)
ax2.set_xscale("log")
ax2.set_xlim(1, 200)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.3, which="both")
ax2.set_title("Fraction of packets with CRC errors vs event rate (all 3 GRBs, Box A)", fontsize=11)

fig.tight_layout()
out = "/Users/skyair/Developer/ihep/blink/slides/fig_crc_rate.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
