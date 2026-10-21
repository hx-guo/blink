#!/usr/bin/env python3
"""Plot 1B vs 1K comparison: light curve + time-channel scatter.

Usage:
    python3 scripts/saturation/plot_compare.py 2020-04-15T08 --trigger 2020-04-15T08:48:08 --before 5 --after 10
    python3 scripts/saturation/plot_compare.py 2022-10-09T13 --trigger 2022-10-09T13:17:02 --before 50 --after 600 --bin 1
"""

import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import os
import sys

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)


def parse_met_or_utc(s):
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
    dt = MET_EPOCH + timedelta(seconds=met)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def run_cli(epoch, subcmd, trigger=None, before=10, after=100, box_filter=None):
    """Call blink_cli sat <subcmd> and parse CSV output."""
    cmd = ["./target/release/blink_cli", "sat", epoch]
    if box_filter:
        cmd.extend(["--box", box_filter])
    cmd.append(subcmd)
    if trigger:
        cmd.append(trigger)
        cmd.extend(["--before", str(before), "--after", str(after)])

    env = os.environ.copy()
    if "HXMT_1B_DIR" not in env:
        env["HXMT_1B_DIR"] = "data/1B"
    if "HXMT_1K_DIR" not in env:
        env["HXMT_1K_DIR"] = "data/1K"

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)

    # Parse: box,type,met,channel,pkt_idx,evt_idx
    mets = []
    channels = []
    box_labels = []
    sec_mets = []  # SEC event times for confidence marking
    for line in proc.stdout:
        parts = line.strip().split(",")
        if len(parts) < 4 or parts[0] == "box":
            continue
        box_name, typ, met_str, ch_str = parts[0], parts[1], parts[2], parts[3]
        if typ == "SEC":
            sec_mets.append(float(met_str))
            continue
        mets.append(float(met_str))
        channels.append(int(ch_str))
        box_labels.append(box_name)

    stderr = proc.stderr.read()
    proc.wait()
    if stderr:
        for line in stderr.strip().split("\n"):
            print(f"  {line}", file=sys.stderr)

    return np.array(mets), np.array(channels), box_labels, np.array(sec_mets)


def main():
    parser = argparse.ArgumentParser(description="Plot 1B vs 1K: light curve + time-channel scatter")
    parser.add_argument("epoch", help="Epoch (e.g. 2020-04-15T08)")
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--before", type=float, default=10.0)
    parser.add_argument("--after", type=float, default=100.0)
    parser.add_argument("--bin", type=float, default=1.0, help="Light curve bin width (seconds)")
    parser.add_argument("--box", type=str, default=None, dest="box_filter")
    parser.add_argument("--no-1k", action="store_true", help="Skip 1K reference")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    # Load 1B data
    print("Loading 1B (new two-pass algorithm)...", file=sys.stderr)
    met_1b, ch_1b, box_1b, sec_mets_1b = run_cli(args.epoch, "solve", trigger=args.trigger,
                                                    before=args.before, after=args.after,
                                                    box_filter=args.box_filter)

    # Load 1K data
    met_1k, ch_1k, box_1k = np.array([]), np.array([]), []
    if not args.no_1k:
        print("Loading 1K reference...", file=sys.stderr)
        try:
            met_1k, ch_1k, box_1k, _ = run_cli(args.epoch, "solve1k", trigger=args.trigger,
                                                  before=args.before, after=args.after,
                                                  box_filter=args.box_filter)
        except Exception as e:
            print(f"  1K not available: {e}", file=sys.stderr)

    if len(met_1b) == 0:
        print("No 1B events found!", file=sys.stderr)
        sys.exit(1)

    print(f"  1B: {len(met_1b):,} events  |  1K: {len(met_1k):,} events", file=sys.stderr)

    # Time reference
    all_mets = np.concatenate([met_1b, met_1k]) if len(met_1k) > 0 else met_1b
    t_ref = parse_met_or_utc(args.trigger) if args.trigger else np.median(all_mets)
    t_min, t_max = all_mets.min(), all_mets.max()

    # ── Compute uncertain regions (SEC gaps > WRAP_PERIOD) ──
    WRAP_PERIOD = 1.048576
    uncertain_intervals = []  # list of (start, stop) in MET
    if len(sec_mets_1b) >= 2:
        sec_sorted = np.sort(sec_mets_1b)
        for i in range(len(sec_sorted) - 1):
            gap = sec_sorted[i + 1] - sec_sorted[i]
            if gap > WRAP_PERIOD:
                uncertain_intervals.append((sec_sorted[i], sec_sorted[i + 1]))

    # ── Figure: 3 rows (light curve, residual, scatter) ──
    fig, (ax_lc, ax_res, ax_ch) = plt.subplots(
        3, 1, figsize=(24, 18), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 2], "hspace": 0.06})

    # ── Shade uncertain regions on all panels ──
    for start, stop in uncertain_intervals:
        for ax in (ax_lc, ax_res, ax_ch):
            ax.axvspan(start - t_ref, stop - t_ref,
                       color="#FFF3E0", alpha=0.5, zorder=0)

    # ── Panel 1: Light curve ──
    bin_width = args.bin
    edges = np.arange(t_min, t_max + bin_width, bin_width)
    x = edges[:-1] - t_ref

    rates_1k = np.zeros(len(x))
    if len(met_1k) > 0:
        rates_1k = np.histogram(met_1k, bins=edges)[0] / bin_width
        ax_lc.fill_between(x, rates_1k, step="post", color="#DDDDDD", alpha=0.9,
                           edgecolor="none", zorder=1)
        ax_lc.step(x, rates_1k, where="post", color="#AAAAAA", lw=0.8,
                   label=f"1K ({len(met_1k):,})", zorder=2)

    rates_1b = np.histogram(met_1b, bins=edges)[0] / bin_width
    ax_lc.fill_between(x, rates_1b, step="post", color="#92C5DE", alpha=0.6,
                       edgecolor="none", zorder=3)
    ax_lc.step(x, rates_1b, where="post", color="#2166AC", lw=0.8,
               label=f"1B ({len(met_1b):,})", zorder=4)

    ax_lc.set_ylabel("Count rate (evt/s)", fontsize=14)
    ax_lc.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax_lc.set_ylim(bottom=0)
    ax_lc.grid(alpha=0.15)
    ax_lc.tick_params(labelsize=12)

    # ── Panel 2: Residual (1B - 1K) ──
    if len(met_1k) > 0:
        residual = rates_1b - rates_1k
        ax_res.fill_between(x, residual, step="post",
                            where=residual >= 0, color="#2166AC", alpha=0.4,
                            edgecolor="none", zorder=2)
        ax_res.fill_between(x, residual, step="post",
                            where=residual < 0, color="#D6604D", alpha=0.4,
                            edgecolor="none", zorder=2)
        ax_res.step(x, residual, where="post", color="#333333", lw=0.6, zorder=3)
        ax_res.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax_res.set_ylabel("1B - 1K (evt/s)", fontsize=14)
    else:
        ax_res.set_ylabel("(no 1K data)", fontsize=14)
    ax_res.grid(alpha=0.15)
    ax_res.tick_params(labelsize=12)

    # ── Panel 3: Time vs Channel scatter ──
    dot_size = max(0.1, min(3.0, 80000 / max(len(met_1b), 1)))

    if len(met_1k) > 0:
        ax_ch.scatter(met_1k - t_ref, ch_1k, s=dot_size, c="#CCCCCC", alpha=0.3,
                      edgecolors="none", rasterized=True, zorder=1, label=f"1K")

    ax_ch.scatter(met_1b - t_ref, ch_1b, s=dot_size, c="#2166AC", alpha=0.4,
                  edgecolors="none", rasterized=True, zorder=2, label=f"1B")

    ax_ch.set_ylabel("Channel", fontsize=14)
    ax_ch.legend(loc="upper right", fontsize=12, framealpha=0.9, markerscale=5)
    ax_ch.grid(alpha=0.15)
    ax_ch.tick_params(labelsize=12)

    # X-axis label
    if args.trigger:
        utc_str = met_to_utc(t_ref)
        ax_ch.set_xlabel(
            f"Time - T₀ (s)    [T₀ = {utc_str} UTC = MET {t_ref:.3f}]", fontsize=13)
    else:
        ax_ch.set_xlabel(f"Time - {t_ref:.3f} (s)", fontsize=13)

    ax_lc.set_xlim(t_min - t_ref, t_max - t_ref)

    box_label = f" Box {args.box_filter.upper()}" if args.box_filter else ""
    fig.suptitle(f"{args.epoch}{box_label}  1B vs 1K  ({bin_width}s bins)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()

    out = args.output or "compare.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
