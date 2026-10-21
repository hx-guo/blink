#!/usr/bin/env python3
"""Plot pass1 1B vs 1K for all 3 GRBs × 3 Boxes = 9 plots."""

import subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime, timezone, timedelta

BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

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


def load_1b_pass1(epoch, trigger, before, after, box_name):
    cmd = ["./target/release/blink_cli", "sat", epoch, "--box", box_name.lower(),
           "solve", trigger, "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env["PASS1_ONLY"] = "1"
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    mets, channels = [], []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 4 or parts[0] == "box":
            continue
        if parts[1] == "SEC":
            continue
        mets.append(float(parts[2]))
        channels.append(int(parts[3]))
    return np.array(mets), np.array(channels)


def load_1k(fits_path, box_name, t_ref, before, after):
    with fits.open(fits_path, memmap=True) as hdul:
        d = hdul[1].data
        t, det, ch = d["Time"], d["Det_ID"], d["Channel"]
    d_lo, d_hi = BOX_DET_RANGES[box_name]
    mask = (det >= d_lo) & (det <= d_hi) & (t >= t_ref - before) & (t <= t_ref + after)
    return t[mask].copy(), ch[mask].copy()


def load_resets(epoch, box_name):
    cmd = ["./target/release/blink_cli", "sat", epoch, "--box", box_name.lower(), "detect"]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    env["MAX_SEC_GAP"] = "999"
    env["PASS1_ONLY"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    resets = []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 5 or p[0] == "box" or p[1] != "FifoReset":
            continue
        resets.append((float(p[2]), float(p[3])))  # (start_met, stop_met)
    return resets


def plot_grb(grb):
    t_ref = parse_met(grb["trigger"])
    bin_width = grb["bin"]

    for box in ["A", "B", "C"]:
        print(f"  {grb['name']} Box {box}...", file=sys.stderr)

        m1b, c1b = load_1b_pass1(grb["epoch"], grb["trigger"],
                                  grb["before"], grb["after"], box)
        m1k, c1k = load_1k(grb["fits"], box, t_ref, grb["before"], grb["after"])
        resets = load_resets(grb["epoch"], box)
        print(f"    {len(m1b):,} 1B events, {len(m1k):,} 1K events, {len(resets)} FIFO resets",
              file=sys.stderr)

        fig, (ax_lc, ax_res) = plt.subplots(
            2, 1, figsize=(24, 12), sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})

        all_mets = np.concatenate([m1b, m1k]) if len(m1k) > 0 else m1b
        if len(all_mets) == 0:
            plt.close()
            continue

        t_min = min(t_ref - grb["before"], all_mets.min())
        t_max = max(t_ref + grb["after"], all_mets.max())
        edges = np.arange(t_min, t_max + bin_width, bin_width)
        x = edges[:-1] - t_ref

        # Shade FIFO reset regions
        for r_start, r_stop in resets:
            for ax in (ax_lc, ax_res):
                ax.axvspan(r_start - t_ref, r_stop - t_ref,
                           color="#F4A460", alpha=0.25, zorder=0, linewidth=0)

        rates_1k = np.histogram(m1k, bins=edges)[0] / bin_width if len(m1k) > 0 else np.zeros(len(x))
        rates_1b = np.histogram(m1b, bins=edges)[0] / bin_width if len(m1b) > 0 else np.zeros(len(x))

        ax_lc.fill_between(x, rates_1k, step="post", color="#DDDDDD", alpha=0.9, zorder=1)
        ax_lc.step(x, rates_1k, where="post", color="#AAAAAA", lw=0.8,
                   label=f"1K ({len(m1k):,})", zorder=2)
        ax_lc.fill_between(x, rates_1b, step="post", color="#92C5DE", alpha=0.6, zorder=3)
        ax_lc.step(x, rates_1b, where="post", color="#2166AC", lw=0.8,
                   label=f"1B pass1 ({len(m1b):,})", zorder=4)

        if resets:
            # Invisible span just for the legend entry
            ax_lc.axvspan(0, 0, color="#F4A460", alpha=0.25,
                          label=f"FIFO reset ({len(resets)})")
        ax_lc.set_ylabel("Count rate (evt/s)", fontsize=14)
        ax_lc.legend(loc="upper right", fontsize=12)
        ax_lc.set_ylim(bottom=0)
        ax_lc.grid(alpha=0.15)

        residual = rates_1b - rates_1k
        res_pos = np.maximum(residual, 0)
        res_neg = np.minimum(residual, 0)
        ax_res.fill_between(x, res_pos, step="post", color="#2166AC", alpha=0.4, zorder=2)
        ax_res.fill_between(x, res_neg, step="post", color="#D6604D", alpha=0.4, zorder=2)
        ax_res.step(x, residual, where="post", color="#333333", lw=0.6, zorder=3)
        ax_res.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax_res.set_ylabel("1B_pass1 − 1K (evt/s)", fontsize=14)
        ax_res.grid(alpha=0.15)

        utc_str = met_to_utc(t_ref)
        ax_res.set_xlabel(f"Time − T₀ (s)    [T₀ = {utc_str} UTC]", fontsize=13)
        ax_lc.set_xlim(t_min - t_ref, t_max - t_ref)

        fig.suptitle(f"{grb['name']}  Box {box}  1B pass1 vs 1K  ({bin_width}s bins)",
                     fontsize=16, fontweight="bold")
        plt.tight_layout()

        safe_name = grb["name"].replace(" ", "_").lower()
        out = f"pass1_{safe_name}_box_{box.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out}", file=sys.stderr)


def main():
    for grb in GRBS:
        print(f"Processing {grb['name']}...", file=sys.stderr)
        plot_grb(grb)


if __name__ == "__main__":
    main()
