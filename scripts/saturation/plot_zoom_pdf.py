#!/usr/bin/env python3
"""Generate PDF with page-by-page zoomed 1B vs 1K comparison.

One page per 0.2s window, only pages containing mismatches.
1K = red crosses, 1B = blue circles. Yellow = uncertain intervals.
Each box gets its own PDF.
"""

import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from datetime import datetime, timezone, timedelta

WRAP_PERIOD = 1.048576
EPOCH = "2022-10-09T13"
TRIGGER = "2022-10-09T13:17:02"
BEFORE = 50.0
AFTER = 900.0
FITS_PATH = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"
BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

PAGE_WIDTH = 0.2  # seconds per page


def parse_met(s):
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"]:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return (dt - MET_EPOCH).total_seconds()
        except ValueError:
            continue
    return float(s)


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
        t = data["Time"]; d = data["Det_ID"]; c = data["Channel"]
    d_lo, d_hi = BOX_DET_RANGES[box_name]
    t_ref = parse_met(TRIGGER)
    mask = (d >= d_lo) & (d <= d_hi) & (t >= t_ref - BEFORE) & (t <= t_ref + AFTER)
    return t[mask].copy(), c[mask].copy(), d[mask].copy()


def get_uncertain_intervals(sec_mets):
    intervals = []
    if len(sec_mets) >= 2:
        for i in range(len(sec_mets) - 1):
            if sec_mets[i + 1] - sec_mets[i] > WRAP_PERIOD:
                intervals.append((sec_mets[i], sec_mets[i + 1]))
    return intervals


def is_confirmed(met, sec_mets, uncertain):
    if len(sec_mets) < 2:
        return False
    if met < sec_mets[0] or met > sec_mets[-1]:
        return False
    for s, e in uncertain:
        if s <= met <= e:
            return False
    return True


def find_mismatch_windows(m1b, c1b, m1k, c1k, sec_mets, uncertain, t_ref):
    """Find 0.2s windows that contain at least one unmatched event."""
    # Build sorted arrays in confirmed region
    def confirmed_mask(mets):
        mask = np.ones(len(mets), dtype=bool)
        if len(sec_mets) < 2:
            return np.zeros(len(mets), dtype=bool)
        mask &= (mets >= sec_mets[0]) & (mets <= sec_mets[-1])
        for s, e in uncertain:
            mask &= ~((mets >= s) & (mets <= e))
        return mask

    mask_1b = confirmed_mask(m1b)
    mask_1k = confirmed_mask(m1k)
    m1b_c = np.sort(m1b[mask_1b])
    c1b_c = c1b[np.argsort(m1b[mask_1b])] if mask_1b.any() else np.array([])
    m1k_c = np.sort(m1k[mask_1k])
    c1k_c = c1k[np.argsort(m1k[mask_1k])] if mask_1k.any() else np.array([])

    # Two-pointer match to find unmatched events
    TOL = 6e-6
    unmatched_times = []
    i, j = 0, 0
    n1b, n1k = len(m1b_c), len(m1k_c)

    while i < n1b and j < n1k:
        dt = m1b_c[i] - m1k_c[j]
        if abs(dt) < TOL:
            # Matched — check channel
            if c1b_c[i] != c1k_c[j]:
                unmatched_times.append(m1b_c[i])
            i += 1; j += 1
        elif dt < 0:
            unmatched_times.append(m1b_c[i])
            i += 1
        else:
            unmatched_times.append(m1k_c[j])
            j += 1
    while i < n1b:
        unmatched_times.append(m1b_c[i]); i += 1
    while j < n1k:
        unmatched_times.append(m1k_c[j]); j += 1

    if not unmatched_times:
        return []

    unmatched_times = np.array(unmatched_times)

    # Build set of PAGE_WIDTH windows that contain mismatches
    # Quantize to PAGE_WIDTH grid
    t_min = unmatched_times.min()
    t_max = unmatched_times.max()
    grid_start = np.floor((t_min - t_ref) / PAGE_WIDTH) * PAGE_WIDTH + t_ref
    grid_end = np.ceil((t_max - t_ref) / PAGE_WIDTH) * PAGE_WIDTH + t_ref

    windows = []
    t = grid_start
    while t < grid_end:
        t_lo, t_hi = t, t + PAGE_WIDTH
        n_in = np.sum((unmatched_times >= t_lo) & (unmatched_times < t_hi))
        if n_in > 0:
            windows.append((t_lo, t_hi))
        t += PAGE_WIDTH

    return windows


def generate_pdf(box_name):
    m1b, c1b, sec_mets = load_1b(box_name)
    m1k, c1k, d1k = load_1k(box_name)
    t_ref = parse_met(TRIGGER)
    uncertain = get_uncertain_intervals(sec_mets)

    windows = find_mismatch_windows(m1b, c1b, m1k, c1k, sec_mets, uncertain, t_ref)
    print(f"  Box {box_name}: {len(windows)} pages with mismatches")

    if not windows:
        print(f"  No mismatches — skipping PDF")
        return

    outfile = f"zoom_box_{box_name.lower()}.pdf"
    with PdfPages(outfile) as pdf:
        for page_idx, (t_lo, t_hi) in enumerate(windows):
            fig, ax = plt.subplots(figsize=(16, 8))

            # Shade uncertain intervals
            for s, e in uncertain:
                if e > t_lo and s < t_hi:
                    ax.axvspan(s - t_ref, e - t_ref, color="#FFF3E0", alpha=0.7, zorder=0)

            # Filter events
            mask_1k = (m1k >= t_lo) & (m1k < t_hi)
            mask_1b = (m1b >= t_lo) & (m1b < t_hi)
            tk = m1k[mask_1k] - t_ref
            ck = c1k[mask_1k]
            tb = m1b[mask_1b] - t_ref
            cb = c1b[mask_1b]

            ax.scatter(tk, ck, marker="x", s=60, c="red", alpha=0.9,
                       linewidths=1.2, zorder=2, label=f"1K ({len(tk)})")
            ax.scatter(tb, cb, marker="o", s=40, facecolors="none",
                       edgecolors="blue", alpha=0.8, linewidths=1.0,
                       zorder=3, label=f"1B ({len(tb)})")

            ax.set_xlim(t_lo - t_ref, t_hi - t_ref)
            ax.set_ylim(-5, 260)
            ax.set_ylabel("Channel", fontsize=12)
            ax.set_xlabel(f"Time − T₀ (s)", fontsize=11)
            ax.set_title(
                f"Box {box_name}  T+{t_lo-t_ref:.3f} ~ T+{t_hi-t_ref:.3f}  "
                f"(1B={len(tb)}, 1K={len(tk)}, Δ={len(tb)-len(tk):+d})  "
                f"[page {page_idx+1}/{len(windows)}]",
                fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
            ax.grid(alpha=0.2)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if (page_idx + 1) % 100 == 0:
                print(f"    ... {page_idx+1}/{len(windows)} pages", file=sys.stderr)

    print(f"  Saved: {outfile} ({len(windows)} pages)")


def main():
    for box in ["A", "B", "C"]:
        generate_pdf(box)


if __name__ == "__main__":
    main()
