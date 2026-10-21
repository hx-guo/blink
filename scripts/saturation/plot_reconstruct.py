#!/usr/bin/env python3
"""Plot reconstruction results: observed + filled vs 1K.

Usage:
    python3 scripts/saturation/plot_reconstruct.py 2026-02-26T10 --trigger 2026-02-26T10:37:53 --before 10 --after 70 --bin 0.2
    python3 scripts/saturation/plot_reconstruct.py 2022-10-09T13 --trigger 2022-10-09T13:17:02 --before 50 --after 700 --bin 0.5
"""

import argparse, subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

GRBS = {
    "2020-04-15T08": {"name": "GRB 200415A", "fits": "data/1K/Y202004/20200415-1036/HXMT_20200415T08_HE-Evt_FFFFFF_V2_1K.FITS"},
    "2022-10-09T13": {"name": "GRB 221009A", "fits": "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"},
    "2026-02-26T10": {"name": "GRB 260226A", "fits": "data/1K/Y202602/20260226-3179/HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS"},
}

BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}


def parse_met(s):
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"]:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return (dt - MET_EPOCH).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s}")


def met_to_utc(met):
    return (MET_EPOCH + timedelta(seconds=met)).strftime("%Y-%m-%dT%H:%M:%S")


def run_reconstruct(epoch, trigger, before, after, box_name):
    cmd = ["./target/release/blink_cli", "sat", epoch, "--box", box_name.lower(),
           "reconstruct", trigger, "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.stderr:
        for line in proc.stderr.strip().split("\n"):
            print(f"    {line}", file=sys.stderr)

    obs_mets, fill_mets = [], []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 3 or parts[0] == "box":
            continue
        typ, met = parts[1], float(parts[2])
        if typ == "EVT":
            obs_mets.append(met)
        elif typ == "FILL_GAP":
            fill_mets.append(met)
    return np.array(obs_mets), np.array(fill_mets)


def load_resets(epoch, box_name):
    cmd = ["./target/release/blink_cli", "sat", epoch, "--box", box_name.lower(), "detect"]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    resets = []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 5 or p[0] == "box" or p[1] != "FifoReset":
            continue
        resets.append((float(p[2]), float(p[3])))
    return resets


def load_1k(fits_path, box_name, t_ref, before, after):
    from astropy.io import fits
    with fits.open(fits_path, memmap=True) as hdul:
        d = hdul[1].data
        t, det = d["Time"], d["Det_ID"]
    d_lo, d_hi = BOX_DET_RANGES[box_name]
    mask = (det >= d_lo) & (det <= d_hi) & (t >= t_ref - before) & (t <= t_ref + after)
    return t[mask].copy()


def main():
    parser = argparse.ArgumentParser(description="Plot reconstruct: observed + filled vs 1K")
    parser.add_argument("epoch", help="Epoch (e.g. 2026-02-26T10)")
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--before", type=float, default=10.0)
    parser.add_argument("--after", type=float, default=100.0)
    parser.add_argument("--bin", type=float, default=0.2, help="Bin width (seconds)")
    parser.add_argument("--box", type=str, default=None, dest="box_filter",
                        help="Single box (A/B/C), default all")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    t_ref = parse_met(args.trigger)
    boxes = [args.box_filter.upper()] if args.box_filter else ["A", "B", "C"]
    grb_info = GRBS.get(args.epoch, {})
    grb_name = grb_info.get("name", args.epoch)
    fits_path = grb_info.get("fits")

    for box in boxes:
        print(f"Processing {grb_name} Box {box}...", file=sys.stderr)

        obs, fill = run_reconstruct(args.epoch, args.trigger,
                                    args.before, args.after, box)
        resets = load_resets(args.epoch, box)

        m1k = np.array([])
        if fits_path and os.path.exists(fits_path):
            print(f"  Loading 1K...", file=sys.stderr)
            m1k = load_1k(fits_path, box, t_ref, args.before, args.after)

        print(f"  obs={len(obs):,}  fill={len(fill):,}  1K={len(m1k):,}  resets={len(resets)}",
              file=sys.stderr)

        if len(obs) == 0 and len(fill) == 0:
            print(f"  No events, skipping.", file=sys.stderr)
            continue

        # Time axis
        bin_w = args.bin
        t_min = t_ref - args.before
        t_max = t_ref + args.after
        edges = np.arange(t_min, t_max + bin_w, bin_w)
        x = edges[:-1] - t_ref

        # Rates
        r_obs = np.histogram(obs, bins=edges)[0] / bin_w if len(obs) > 0 else np.zeros(len(x))
        r_fill = np.histogram(fill, bins=edges)[0] / bin_w if len(fill) > 0 else np.zeros(len(x))
        r_total = r_obs + r_fill
        r_1k = np.histogram(m1k, bins=edges)[0] / bin_w if len(m1k) > 0 else np.zeros(len(x))

        # Plot: 2 panels (light curve + residual)
        fig, (ax_lc, ax_res) = plt.subplots(
            2, 1, figsize=(24, 12), sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})

        # Shade FIFO resets
        for r_start, r_stop in resets:
            for ax in (ax_lc, ax_res):
                ax.axvspan(r_start - t_ref, r_stop - t_ref,
                           color="#F4A460", alpha=0.25, zorder=0, linewidth=0)

        # Light curve panel
        if len(m1k) > 0:
            ax_lc.fill_between(x, r_1k, step="post", color="#DDDDDD", alpha=0.9, zorder=1)
            ax_lc.step(x, r_1k, where="post", color="#AAAAAA", lw=0.8,
                       label=f"1K ({len(m1k):,})", zorder=2)

        # Stacked: observed + filled
        ax_lc.fill_between(x, r_obs, step="post", color="#92C5DE", alpha=0.6, zorder=3)
        ax_lc.step(x, r_obs, where="post", color="#2166AC", lw=0.8,
                   label=f"1B observed ({len(obs):,})", zorder=4)

        if len(fill) > 0:
            ax_lc.fill_between(x, r_obs, r_total, step="post",
                               color="#F4A582", alpha=0.7, edgecolor="#B2182B", linewidth=0.8,
                               zorder=5, label=f"1B + filled ({len(fill):,} filled)")

        if resets:
            ax_lc.axvspan(0, 0, color="#F4A460", alpha=0.25,
                          label=f"FIFO reset ({len(resets)})")

        ax_lc.set_ylabel("Count rate (evt/s)", fontsize=14)
        ax_lc.legend(loc="upper right", fontsize=12)
        ax_lc.set_ylim(bottom=0)
        ax_lc.grid(alpha=0.15)
        ax_lc.tick_params(labelsize=12)

        # Residual panel: (obs + fill) - 1K
        if len(m1k) > 0:
            residual = r_total - r_1k
            res_pos = np.maximum(residual, 0)
            res_neg = np.minimum(residual, 0)
            ax_res.fill_between(x, res_pos, step="post", color="#2166AC", alpha=0.4, zorder=2)
            ax_res.fill_between(x, res_neg, step="post", color="#D6604D", alpha=0.4, zorder=2)
            ax_res.step(x, residual, where="post", color="#333333", lw=0.6, zorder=3)
            ax_res.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
            ax_res.set_ylabel("(1B+fill) − 1K (evt/s)", fontsize=14)
        else:
            ax_res.set_ylabel("(no 1K data)", fontsize=14)
        ax_res.grid(alpha=0.15)
        ax_res.tick_params(labelsize=12)

        utc_str = met_to_utc(t_ref)
        ax_res.set_xlabel(f"Time − T₀ (s)    [T₀ = {utc_str} UTC]", fontsize=13)
        ax_lc.set_xlim(t_min - t_ref, t_max - t_ref)

        fig.suptitle(f"{grb_name}  Box {box}  Reconstruct vs 1K  ({bin_w}s bins)",
                     fontsize=16, fontweight="bold")
        plt.tight_layout()

        if args.output and len(boxes) == 1:
            out = args.output
        else:
            safe = grb_name.replace(" ", "_").lower()
            out = f"reconstruct_{safe}_box_{box.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
