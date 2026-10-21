#!/usr/bin/env python3
"""Plot 1B pass1 vs 1K with two colors:
   - Blue: confident (1s SEC pairs)
   - Orange: greedy (gap > 1s SEC pairs)
All 3 GRBs × 3 Boxes = 9 plots.
"""

import subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.io import fits
from datetime import datetime, timezone, timedelta

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}

GRBS = [
    {
        "name": "GRB 200415A",
        "epoch": "2020-04-15T08",
        "trigger": "2020-04-15T08:48:08",
        "before": 0.5,
        "after": 2.0,
        "bin": 0.001,
        "fits": "data/1K/Y202004/20200415-1036/HXMT_20200415T08_HE-Evt_FFFFFF_V2_1K.FITS",
    },
    {
        "name": "GRB 221009A",
        "epoch": "2022-10-09T13",
        "trigger": "2022-10-09T13:17:02",
        "before": 50.0,
        "after": 700.0,
        "bin": 0.5,
        "fits": "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS",
    },
    {
        "name": "GRB 260226A",
        "epoch": "2026-02-26T10",
        "trigger": "2026-02-26T10:37:53",
        "before": 10.0,
        "after": 70.0,
        "bin": 0.2,
        "fits": "data/1K/Y202602/20260226-3179/HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS",
    },
]


def parse_met(s):
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"]:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return (dt - MET_EPOCH).total_seconds()
        except ValueError:
            continue
    return float(s)


def met_to_utc(met):
    return (MET_EPOCH + timedelta(seconds=met)).strftime("%Y-%m-%dT%H:%M:%S")


def load_1b(epoch, trigger, before, after, box, max_gap):
    cmd = ["./target/release/blink_cli", "sat", epoch, "--box", box.lower(),
           "solve", trigger, "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env["PASS1_ONLY"] = "1"
    env["HXMT_1B_DIR"] = "data/1B"
    env["HXMT_1K_DIR"] = "data/1K"
    env["MAX_SEC_GAP"] = str(max_gap)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    mets = []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 4 or parts[0] == "box" or parts[1] == "SEC":
            continue
        mets.append(float(parts[2]))
    return np.array(mets)


def load_1k(fits_path, box, t_ref, before, after):
    with fits.open(fits_path, memmap=True) as h:
        d = h[1].data
        lo, hi = BOX_DET_RANGES[box]
        mask = (d["Det_ID"] >= lo) & (d["Det_ID"] <= hi) & \
               (d["Time"] >= t_ref - before) & (d["Time"] <= t_ref + after)
        return d["Time"][mask].copy()


def plot_grb(grb):
    t_ref = parse_met(grb["trigger"])

    for box in ["A", "B", "C"]:
        print(f"  {grb['name']} Box {box}...", file=sys.stderr)

        m_1s = load_1b(grb["epoch"], grb["trigger"], grb["before"], grb["after"], box, 1)
        m_all = load_1b(grb["epoch"], grb["trigger"], grb["before"], grb["after"], box, 999)
        m_1k = load_1k(grb["fits"], box, t_ref, grb["before"], grb["after"])

        # Δstime>1 events (not in Δstime=1 set)
        set_1s = set(np.round(m_1s, 7))
        m_multi = np.array([m for m in m_all if round(m, 7) not in set_1s])

        edges = np.arange(t_ref - grb["before"], t_ref + grb["after"] + grb["bin"], grb["bin"])
        x = edges[:-1] - t_ref

        r_1k = np.histogram(m_1k, bins=edges)[0] / grb["bin"]
        r_1s = np.histogram(m_1s, bins=edges)[0] / grb["bin"]
        r_multi = np.histogram(m_multi, bins=edges)[0] / grb["bin"]
        r_total = r_1s + r_multi

        fig, (ax_lc, ax_res) = plt.subplots(2, 1, figsize=(24, 12), sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})

        # 1K
        ax_lc.fill_between(x, r_1k, step="post", color="#DDDDDD", alpha=0.9, zorder=1)
        ax_lc.step(x, r_1k, where="post", color="#AAAAAA", lw=0.8, zorder=2)

        # Δstime>1 (orange)
        ax_lc.fill_between(x, r_total, step="post", color="#F4A460", alpha=0.6, zorder=3)

        # Δstime=1 (blue, overwrites orange)
        ax_lc.fill_between(x, r_1s, step="post", color="#92C5DE", alpha=0.8, zorder=4)
        ax_lc.step(x, r_1s, where="post", color="#2166AC", lw=0.8, zorder=4)

        legend_elements = [
            Patch(facecolor="#DDDDDD", edgecolor="#AAAAAA", label=f"1K ({len(m_1k):,})"),
            Patch(facecolor="#92C5DE", edgecolor="#2166AC", label=f"1B \u0394t=1 ({len(m_1s):,})"),
            Patch(facecolor="#F4A460", edgecolor="#D2691E", label=f"1B \u0394t>1 ({len(m_multi):,})"),
        ]
        ax_lc.legend(handles=legend_elements, loc="upper right", fontsize=12)
        ax_lc.set_ylabel("Count rate (evt/s)", fontsize=14)
        ax_lc.set_ylim(bottom=0)
        ax_lc.grid(alpha=0.15)

        # Residual
        residual = r_total - r_1k
        res_pos = np.maximum(residual, 0)
        res_neg = np.minimum(residual, 0)
        ax_res.fill_between(x, res_pos, step="post", color="#2166AC", alpha=0.4, zorder=2)
        ax_res.fill_between(x, res_neg, step="post", color="#D6604D", alpha=0.4, zorder=2)
        ax_res.step(x, residual, where="post", color="#333333", lw=0.6, zorder=3)
        ax_res.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax_res.set_ylabel("1B − 1K (evt/s)", fontsize=14)
        ax_res.grid(alpha=0.15)

        ax_res.set_xlabel(f"Time − T₀ (s)    [T₀ = {met_to_utc(t_ref)} UTC]", fontsize=13)
        ax_lc.set_xlim(-grb["before"], grb["after"])

        fig.suptitle(f"{grb['name']}  Box {box}  \u0394t=1 (blue) vs \u0394t>1 (orange) vs 1K (gray)  ({grb['bin']}s bins)",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()

        safe_name = grb["name"].replace(" ", "_").lower()
        out = f"gap_boundary_{safe_name}_box_{box.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    for grb in GRBS:
        print(f"Processing {grb['name']}...", file=sys.stderr)
        plot_grb(grb)
