#!/usr/bin/env python3
"""Plot pass1-only 1B results vs 1K. Shows where pass1 has coverage (confident)
and where it leaves NaN gaps (uncertain)."""

import subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    with fits.open(FITS_PATH, memmap=True) as hdul:
        d = hdul[1].data
        t, det, ch = d["Time"], d["Det_ID"], d["Channel"]
    d_lo, d_hi = BOX_DET_RANGES[box_name]
    t_ref = parse_met(TRIGGER)
    mask = (det >= d_lo) & (det <= d_hi) & (t >= t_ref - BEFORE) & (t <= t_ref + AFTER)
    return t[mask].copy(), ch[mask].copy()

def main():
    t_ref = parse_met(TRIGGER)
    bin_width = 0.5

    for box in ["A", "B", "C"]:
        print(f"Processing Box {box}...", file=sys.stderr)
        m1b, c1b, sec_mets = load_1b_pass1(box)
        m1k, c1k = load_1k(box)

        # Compute uncertain intervals from SEC gaps
        uncertain = []
        for i in range(len(sec_mets) - 1):
            if sec_mets[i+1] - sec_mets[i] > WRAP_PERIOD:
                uncertain.append((sec_mets[i], sec_mets[i+1]))

        fig, (ax_lc, ax_res) = plt.subplots(2, 1, figsize=(24, 12), sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})

        # Shade uncertain
        for s, e in uncertain:
            for ax in (ax_lc, ax_res):
                ax.axvspan(s - t_ref, e - t_ref, color="#FFF3E0", alpha=0.5, zorder=0)

        # Light curves
        edges = np.arange(t_ref - BEFORE, t_ref + AFTER + bin_width, bin_width)
        x = edges[:-1] - t_ref

        rates_1k = np.histogram(m1k, bins=edges)[0] / bin_width
        rates_1b = np.histogram(m1b, bins=edges)[0] / bin_width

        ax_lc.fill_between(x, rates_1k, step="post", color="#DDDDDD", alpha=0.9, zorder=1)
        ax_lc.step(x, rates_1k, where="post", color="#AAAAAA", lw=0.8, label=f"1K ({len(m1k):,})", zorder=2)
        ax_lc.fill_between(x, rates_1b, step="post", color="#92C5DE", alpha=0.6, zorder=3)
        ax_lc.step(x, rates_1b, where="post", color="#2166AC", lw=0.8, label=f"1B pass1 ({len(m1b):,})", zorder=4)

        ax_lc.set_ylabel("Count rate (evt/s)", fontsize=14)
        ax_lc.legend(loc="upper right", fontsize=12)
        ax_lc.set_ylim(bottom=0)
        ax_lc.grid(alpha=0.15)

        # Residual
        residual = rates_1b - rates_1k
        ax_res.fill_between(x, residual, step="post",
                            where=residual >= 0, color="#2166AC", alpha=0.4, zorder=2)
        ax_res.fill_between(x, residual, step="post",
                            where=residual < 0, color="#D6604D", alpha=0.4, zorder=2)
        ax_res.step(x, residual, where="post", color="#333333", lw=0.6, zorder=3)
        ax_res.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax_res.set_ylabel("1B_pass1 − 1K (evt/s)", fontsize=14)
        ax_res.grid(alpha=0.15)

        ax_res.set_xlabel(f"Time − T₀ (s)    [T₀ = {met_to_utc(t_ref)} UTC]", fontsize=13)
        ax_lc.set_xlim(-BEFORE, AFTER)

        fig.suptitle(f"Box {box}  Pass 1 only vs 1K  ({bin_width}s bins)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        out = f"pass1_vs_1k_box_{box.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
