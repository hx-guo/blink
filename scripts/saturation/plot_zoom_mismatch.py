#!/usr/bin/env python3
"""Plot zoomed-in views of 1B vs 1K events near mismatch regions.

1K = red crosses, 1B = blue circles. Yellow background = uncertain intervals.
"""

import subprocess
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime, timezone

WRAP_PERIOD = 1.048576
EPOCH = "2022-10-09T13"
TRIGGER = "2022-10-09T13:17:02"
BEFORE = 50.0
AFTER = 900.0
FITS_PATH = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"
BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

def parse_met(s):
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"]:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return (dt - MET_EPOCH).total_seconds()
        except ValueError:
            continue
    return float(s)

def met_to_utc(met):
    from datetime import timedelta
    dt = MET_EPOCH + timedelta(seconds=met)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def load_1b(box_name):
    cmd = [
        "./target/release/blink_cli", "sat", EPOCH, "--box", box_name.lower(),
        "solve", TRIGGER, "--before", str(BEFORE), "--after", str(AFTER),
    ]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  Loading 1B Box {box_name}...", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    mets, channels, sec_mets = [], [], []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 4 or parts[0] == "box":
            continue
        typ, met, ch = parts[1], float(parts[2]), int(parts[3])
        if typ == "SEC":
            sec_mets.append(met)
        else:
            mets.append(met)
            channels.append(ch)
    return np.array(mets), np.array(channels), np.sort(sec_mets)

def load_1k(box_name):
    print(f"  Loading 1K Box {box_name}...", file=sys.stderr)
    with fits.open(FITS_PATH, memmap=True) as hdul:
        data = hdul[1].data
        times = data["Time"]
        det_ids = data["Det_ID"]
        channels = data["Channel"]
    d_lo, d_hi = BOX_DET_RANGES[box_name]
    t_ref = parse_met(TRIGGER)
    mask = (det_ids >= d_lo) & (det_ids <= d_hi) & \
           (times >= t_ref - BEFORE) & (times <= t_ref + AFTER)
    return times[mask].copy(), channels[mask].copy(), det_ids[mask].copy()

def get_uncertain_intervals(sec_mets):
    intervals = []
    if len(sec_mets) >= 2:
        for i in range(len(sec_mets) - 1):
            if sec_mets[i + 1] - sec_mets[i] > WRAP_PERIOD:
                intervals.append((sec_mets[i], sec_mets[i + 1]))
    return intervals

def main():
    t_ref = parse_met(TRIGGER)

    # Mismatch regions to zoom into — narrow windows to see individual events
    # Each: (box, center_t_rel, half_width, description)
    zoom_regions = [
        ("A", 224.950, 0.010, "Box A T+224.94~224.96"),
        ("A", 259.50, 0.025, "Box A T+259.48~259.53"),
        ("B", 257.73, 0.020, "Box B T+257.71~257.75"),
        ("C", 109.15, 0.020, "Box C T+109.13~109.17"),
        ("C", 260.00, 0.025, "Box C T+259.98~260.03"),
        ("C", 515.00, 0.025, "Box C T+514.98~515.03"),
    ]

    # Pre-load data per box
    box_data = {}
    needed_boxes = set(r[0] for r in zoom_regions)
    for b in needed_boxes:
        m1b, c1b, sec = load_1b(b)
        m1k, c1k, d1k = load_1k(b)
        unc = get_uncertain_intervals(sec)
        box_data[b] = (m1b, c1b, m1k, c1k, d1k, unc)

    n_plots = len(zoom_regions)
    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for idx, (box, center, half_w, desc) in enumerate(zoom_regions):
        ax = axes[idx]
        m1b, c1b, m1k, c1k, d1k, unc = box_data[box]

        t_lo = t_ref + center - half_w
        t_hi = t_ref + center + half_w

        # Filter to window
        mask_1b = (m1b >= t_lo) & (m1b <= t_hi)
        mask_1k = (m1k >= t_lo) & (m1k <= t_hi)

        t1b = m1b[mask_1b] - t_ref
        ch1b = c1b[mask_1b]
        t1k = m1k[mask_1k] - t_ref
        ch1k = c1k[mask_1k]

        # Shade uncertain intervals
        for s, e in unc:
            if e - t_ref > center - half_w and s - t_ref < center + half_w:
                ax.axvspan(s - t_ref, e - t_ref, color="#FFF3E0", alpha=0.7, zorder=0)

        # Plot 1K as red crosses (bottom layer), 1B as blue circles (top)
        ax.scatter(t1k, ch1k, marker="x", s=80, c="red", alpha=0.9,
                   linewidths=1.5, zorder=2, label=f"1K ({len(t1k)})")
        ax.scatter(t1b, ch1b, marker="o", s=60, facecolors="none",
                   edgecolors="blue", alpha=0.8, linewidths=1.2,
                   zorder=3, label=f"1B ({len(t1b)})")

        ax.set_xlim(center - half_w, center + half_w)
        ax.set_ylabel("Channel", fontsize=12)
        ax.set_title(f"{desc}  (1B={len(t1b)}, 1K={len(t1k)}, diff={len(t1b)-len(t1k):+d})",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(f"Time - T₀ (s)    [T₀ = {met_to_utc(t_ref)} UTC]", fontsize=12)
    plt.tight_layout()
    out = "zoom_mismatch.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
