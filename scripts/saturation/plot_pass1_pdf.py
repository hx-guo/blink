#!/usr/bin/env python3
"""Generate PDF: pass1-only 1B vs 1K, page by page, 1K=red crosses, 1B=blue circles.
Covers the full time range. One page per 0.2s window."""

import subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from datetime import datetime, timezone, timedelta

EPOCH = "2022-10-09T13"
TRIGGER = "2022-10-09T13:17:02"
BEFORE = 50.0
AFTER = 900.0
FITS_PATH = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"
BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
WRAP_PERIOD = 1.048576
PAGE_WIDTH = 0.2  # seconds per page

def parse_met(s):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (dt - MET_EPOCH).total_seconds()

def met_to_utc(met):
    return (MET_EPOCH + timedelta(seconds=met)).strftime("%Y-%m-%dT%H:%M:%S")

def load_1b_pass1(box_name):
    cmd = ["./target/release/blink_cli", "sat", EPOCH, "--box", box_name.lower(),
           "solve", TRIGGER, "--before", str(BEFORE), "--after", str(AFTER)]
    env = os.environ.copy()
    env["PASS1_ONLY"] = "1"
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  Loading 1B pass1 Box {box_name}...", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    mets, channels, sec_mets = [], [], []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 4 or parts[0] == "box": continue
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
        d = hdul[1].data
        t, det, ch = d["Time"], d["Det_ID"], d["Channel"]
    d_lo, d_hi = BOX_DET_RANGES[box_name]
    t_ref = parse_met(TRIGGER)
    mask = (det >= d_lo) & (det <= d_hi) & (t >= t_ref - BEFORE) & (t <= t_ref + AFTER)
    return t[mask].copy(), ch[mask].copy()

def get_uncertain_intervals(sec_mets):
    intervals = []
    if len(sec_mets) >= 2:
        for i in range(len(sec_mets) - 1):
            if sec_mets[i+1] - sec_mets[i] > WRAP_PERIOD:
                intervals.append((sec_mets[i], sec_mets[i+1]))
    return intervals

def generate_pdf(box_name):
    m1b, c1b, sec_mets = load_1b_pass1(box_name)
    m1k, c1k = load_1k(box_name)
    t_ref = parse_met(TRIGGER)
    # No yellow shading — pass1 confidence is more precise than SEC-gap intervals

    # Only generate pages that have ANY events (1B or 1K)
    all_mets = np.concatenate([m1b, m1k]) if len(m1k) > 0 else m1b
    t_start = np.floor((all_mets.min() - t_ref) / PAGE_WIDTH) * PAGE_WIDTH + t_ref
    t_end = np.ceil((all_mets.max() - t_ref) / PAGE_WIDTH) * PAGE_WIDTH + t_ref
    n_pages_total = int((t_end - t_start) / PAGE_WIDTH)

    outfile = f"pass1_box_{box_name.lower()}.pdf"
    print(f"  Generating {outfile} (~{n_pages_total} pages)...", file=sys.stderr)

    with PdfPages(outfile) as pdf:
        page_num = 0
        t = t_start
        while t < t_end:
            t_lo, t_hi = t, t + PAGE_WIDTH

            mask_1b = (m1b >= t_lo) & (m1b < t_hi)
            mask_1k = (m1k >= t_lo) & (m1k < t_hi)

            n1b = mask_1b.sum()
            n1k = mask_1k.sum()

            if n1b == 0 and n1k == 0:
                t += PAGE_WIDTH
                continue

            page_num += 1
            fig, ax = plt.subplots(figsize=(16, 8))

            tb = m1b[mask_1b] - t_ref
            cb = c1b[mask_1b]
            tk = m1k[mask_1k] - t_ref
            ck = c1k[mask_1k]

            ax.scatter(tk, ck, marker="x", s=60, c="red", alpha=0.9,
                       linewidths=1.2, zorder=2, label=f"1K ({n1k})")
            ax.scatter(tb, cb, marker="o", s=40, facecolors="none",
                       edgecolors="blue", alpha=0.8, linewidths=1.0,
                       zorder=3, label=f"1B pass1 ({n1b})")

            ax.set_xlim(t_lo - t_ref, t_hi - t_ref)
            ax.set_ylim(-5, 260)
            ax.set_ylabel("Channel", fontsize=12)
            ax.set_xlabel("Time − T₀ (s)", fontsize=11)

            status = ""
            if n1b == 0 and n1k > 0:
                status = "  [1B gap]"
            elif n1b > 0 and n1k == 0:
                status = "  [1K gap]"

            ax.set_title(
                f"Box {box_name}  T+{t_lo-t_ref:.3f}~{t_hi-t_ref:.3f}  "
                f"(1B={n1b}, 1K={n1k}, Δ={n1b-n1k:+d}){status}  "
                f"[page {page_num}]",
                fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
            ax.grid(alpha=0.2)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if page_num % 500 == 0:
                print(f"    ... {page_num} pages", file=sys.stderr)

            t += PAGE_WIDTH

    print(f"  Saved: {outfile} ({page_num} pages)")

def main():
    for box in ["A", "B", "C"]:
        generate_pdf(box)

if __name__ == "__main__":
    main()
