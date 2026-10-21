#!/usr/bin/env python3
"""Plot 1B light curve with 1K reference and saturation intervals.

Calls Rust CLI via subprocess (no intermediate files).

Usage:
    python3 scripts/saturation/plot_solve.py 2020-04-15T08 --trigger 2020-04-15T08:48:08 --before 5 --after 10
    python3 scripts/saturation/plot_solve.py 2022-10-09T13 --trigger 2022-10-09T13:17:02 --before 50 --after 600 --bin 1
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
    """Call blink_cli sat <subcmd> and parse event output via pipe."""
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

    # boxes[box][type] = [met, ...]
    boxes = {}
    for line in proc.stdout:
        parts = line.strip().split(",")
        if len(parts) < 3 or parts[0] == "box":
            continue
        box, typ, met = parts[0], parts[1], float(parts[2])
        if typ == "SEC":
            continue
        boxes.setdefault(box, {}).setdefault(typ, []).append(met)

    stderr = proc.stderr.read()
    proc.wait()
    if stderr:
        for line in stderr.strip().split("\n"):
            print(f"  {line}", file=sys.stderr)

    for box in boxes:
        for typ in boxes[box]:
            boxes[box][typ] = np.array(boxes[box][typ])
    return boxes


def run_detect(epoch, trigger=None, before=10, after=100):
    """Call blink_cli sat detect and parse saturation intervals."""
    cmd = ["./target/release/blink_cli", "sat", epoch, "detect"]
    if trigger:
        cmd.append(trigger)
        cmd.extend(["--before", str(before), "--after", str(after)])

    env = os.environ.copy()
    if "HXMT_1B_DIR" not in env:
        env["HXMT_1B_DIR"] = "data/1B"

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)

    fifo_resets = {}   # box -> list of (start, stop)
    silent_drops = {}  # box -> list of (start, stop)

    for line in proc.stdout:
        parts = line.strip().split(",")
        if len(parts) < 5 or parts[0] == "box":
            continue
        box, typ = parts[0], parts[1]
        start, stop = float(parts[2]), float(parts[3])
        if typ == "FifoReset":
            fifo_resets.setdefault(box, []).append((start, stop))
        elif typ == "SilentDrop":
            silent_drops.setdefault(box, []).append((start, stop))

    stderr = proc.stderr.read()
    proc.wait()
    if stderr:
        for line in stderr.strip().split("\n"):
            print(f"  {line}", file=sys.stderr)

    return fifo_resets, silent_drops


def plot_lightcurve(b1, k1, fifo_resets, silent_drops,
                    trigger_met=None, bin_width=1.0, output=None, epoch=""):
    all_times_list = []
    for box_data in [b1, k1]:
        for type_dict in box_data.values():
            for arr in type_dict.values():
                if len(arr) > 0:
                    all_times_list.append(arr)
    all_times = np.concatenate(all_times_list) if all_times_list else np.array([0])
    t_min = all_times.min()
    t_max = all_times.max()
    t_ref = trigger_met if trigger_met else (t_min + t_max) / 2

    edges = np.arange(t_min, t_max + bin_width, bin_width)
    plot_edges = edges - t_ref

    colors_1b = {"A": "#2166AC", "B": "#D6604D", "C": "#1B7837"}
    fill_1b = {"A": "#92C5DE", "B": "#F4A582", "C": "#A6D96A"}

    box_names = sorted(set(list(b1.keys()) + list(k1.keys())))
    # Each box: light curve + saturation strip below it, then merged + strip
    # height ratios: [lc, sat, lc, sat, lc, sat, lc_merged, sat_merged]
    n_groups = len(box_names) + 1  # 3 boxes + merged
    height_ratios = []
    for _ in range(n_groups):
        height_ratios.extend([1, 0.12])  # lc tall, sat strip thin

    fig, all_axes = plt.subplots(n_groups * 2, 1,
                                  figsize=(16, 3.5 * n_groups),
                                  sharex=True,
                                  gridspec_kw={"height_ratios": height_ratios,
                                               "hspace": 0.05})

    def draw_lc(ax, box_1k, box_1b, fill_color, line_color):
        """Draw: 1K background + 1B observed + FILL_GAP bars + FILL_SD bars."""
        n_bins = len(plot_edges) - 1
        x = plot_edges[:-1]

        # 1K background
        t_1k = box_1k.get("EVT", np.array([])) if box_1k else np.array([])
        if len(t_1k) > 0:
            rates_k = np.histogram(t_1k, bins=edges)[0] / bin_width
            ax.fill_between(x, rates_k, step="post", color="#DDDDDD",
                      alpha=0.8, edgecolor="none", linewidth=0, zorder=1)
            ax.step(x, rates_k, where="post", color="#AAAAAA",
                    lw=0.5, label=f"1K ({len(t_1k):,})", zorder=2)

        if not box_1b:
            return

        # 1B observed
        t_evt = box_1b.get("EVT", np.array([]))
        rates_evt = np.histogram(t_evt, bins=edges)[0] / bin_width if len(t_evt) else np.zeros(n_bins)
        ax.fill_between(x, rates_evt, step="post", color=fill_color, alpha=0.5,
                  edgecolor="none", linewidth=0, zorder=3)
        ax.step(x, rates_evt, where="post", color=line_color, lw=0.5,
                label=f"1B ({len(t_evt):,})", zorder=4)

        # FILL_GAP: red bars only where fill > 0
        t_gap = box_1b.get("FILL_GAP", np.array([]))
        if len(t_gap) > 0:
            rates_gap = np.histogram(t_gap, bins=edges)[0] / bin_width
            mask = rates_gap > 0
            if mask.any():
                ax.bar(x[mask], rates_gap[mask], width=bin_width,
                       bottom=rates_evt[mask], color="#E74C3C", alpha=0.5,
                       align="edge", edgecolor="none", zorder=5,
                       label=f"Gap fill ({len(t_gap):,})")
        else:
            rates_gap = np.zeros(n_bins)

        # FILL_SD: purple bars only where fill > 0
        t_sd = box_1b.get("FILL_SD", np.array([]))
        if len(t_sd) > 0:
            rates_sd = np.histogram(t_sd, bins=edges)[0] / bin_width
            mask = rates_sd > 0
            if mask.any():
                base = rates_evt + rates_gap
                ax.bar(x[mask], rates_sd[mask], width=bin_width,
                       bottom=base[mask], color="#9B59B6", alpha=0.5,
                       align="edge", edgecolor="none", zorder=6,
                       label=f"SD fill ({len(t_sd):,})")

    def draw_sat_strip(ax, box, color):
        """Draw saturation indicators as a thin strip."""
        for start, stop in fifo_resets.get(box, []):
            ax.axvspan(start - t_ref, stop - t_ref, color=color, alpha=0.7)
        for start, stop in silent_drops.get(box, []):
            mid = (start + stop) / 2 - t_ref
            ax.axvline(mid, color="purple", alpha=0.8, lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.patch.set_facecolor("#F8F8F8")
        n_fr = len(fifo_resets.get(box, []))
        n_sd = len(silent_drops.get(box, []))
        if n_fr or n_sd:
            label = f"FR:{n_fr}"
            if n_sd:
                label += f" SD:{n_sd}"
            ax.text(0.99, 0.5, label, transform=ax.transAxes,
                    ha="right", va="center", fontsize=7, color="#666666")

    # Per-box panels
    for i, box in enumerate(box_names):
        ax_lc = all_axes[i * 2]
        ax_sat = all_axes[i * 2 + 1]

        draw_lc(ax_lc, k1.get(box, {}), b1.get(box, {}),
                fill_1b.get(box, "#88BBDD"),
                colors_1b.get(box, "#336699"))
        ax_lc.set_ylabel(f"Box {box}\n(evt/s)", fontsize=12)
        ax_lc.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_lc.grid(alpha=0.12)
        ax_lc.set_ylim(bottom=0)

        draw_sat_strip(ax_sat, box, colors_1b.get(box, "#888888"))

    # Merged panel
    idx_merged = len(box_names) * 2
    ax_lc = all_axes[idx_merged]
    ax_sat = all_axes[idx_merged + 1]

    def merge_boxes(data):
        merged = {}
        for type_dict in data.values():
            for typ, arr in type_dict.items():
                merged.setdefault(typ, []).append(arr)
        return {t: np.concatenate(v) for t, v in merged.items()}

    merged_k = merge_boxes(k1) if k1 else {}
    merged_b = merge_boxes(b1) if b1 else {}
    draw_lc(ax_lc, merged_k, merged_b, "#B0C4DE", "#4A6A8A")
    ax_lc.set_ylabel("A+B+C\n(evt/s)", fontsize=12)
    ax_lc.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_lc.grid(alpha=0.12)
    ax_lc.set_ylim(bottom=0)

    # Merged saturation: show all boxes
    for box in box_names:
        for start, stop in fifo_resets.get(box, []):
            ax_sat.axvspan(start - t_ref, stop - t_ref,
                          color=colors_1b.get(box, "#888888"), alpha=0.5)
        for start, stop in silent_drops.get(box, []):
            mid = (start + stop) / 2 - t_ref
            ax_sat.axvline(mid, color="purple", alpha=0.6, lw=0.6)
    ax_sat.set_ylim(0, 1)
    ax_sat.set_yticks([])
    ax_sat.patch.set_facecolor("#F8F8F8")
    n_fr = sum(len(v) for v in fifo_resets.values())
    n_sd = sum(len(v) for v in silent_drops.values())
    ax_sat.text(0.99, 0.5, f"FR:{n_fr} SD:{n_sd}", transform=ax_sat.transAxes,
                ha="right", va="center", fontsize=7, color="#666666")

    if trigger_met:
        utc_str = met_to_utc(t_ref)
        ax_sat.set_xlabel(
            f"Time - T₀ (s)    [T₀ = {utc_str} UTC = MET {t_ref:.3f}]", fontsize=11)
    else:
        ax_sat.set_xlabel(f"Time - {t_ref:.3f} (s)", fontsize=11)

    all_axes[0].set_xlim(plot_edges[0], plot_edges[-1])
    fig.suptitle(f"{epoch}  Light Curve ({bin_width}s bins)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = output or "lightcurve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot 1B light curve with 1K ref + saturation")
    parser.add_argument("epoch", help="Epoch (e.g. 2020-04-15T08)")
    parser.add_argument("--trigger", type=str, default=None,
                        help="Trigger time (MET or UTC)")
    parser.add_argument("--before", type=float, default=10.0)
    parser.add_argument("--after", type=float, default=100.0)
    parser.add_argument("--bin", type=float, default=1.0, help="Bin width (seconds)")
    parser.add_argument("--box", type=str, default=None, dest="box_filter")
    parser.add_argument("--no-1k", action="store_true", help="Skip 1K reference")
    parser.add_argument("--no-sat", action="store_true", help="Skip saturation detection")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    if not args.no_sat:
        print("Loading 1B + reconstruction...")
        b1 = run_cli(args.epoch, "reconstruct", trigger=args.trigger,
                     before=args.before, after=args.after, box_filter=args.box_filter)
    else:
        print("Loading 1B data...")
        b1 = run_cli(args.epoch, "solve", trigger=args.trigger,
                     before=args.before, after=args.after, box_filter=args.box_filter)

    k1 = {}
    if not args.no_1k:
        print("Loading 1K data...")
        try:
            k1 = run_cli(args.epoch, "solve1k", trigger=args.trigger,
                         before=args.before, after=args.after, box_filter=args.box_filter)
        except Exception as e:
            print(f"  1K not available: {e}", file=sys.stderr)

    fifo_resets, silent_drops = {}, {}
    if not args.no_sat:
        print("Detecting saturation...")
        try:
            fifo_resets, silent_drops = run_detect(
                args.epoch, trigger=args.trigger,
                before=args.before, after=args.after)
        except Exception as e:
            print(f"  Detection failed: {e}", file=sys.stderr)

    if not b1:
        print("No 1B events found!", file=sys.stderr)
        sys.exit(1)

    n_evt = sum(len(td.get("EVT", [])) for td in b1.values())
    n_gap = sum(len(td.get("FILL_GAP", [])) for td in b1.values())
    n_fsd = sum(len(td.get("FILL_SD", [])) for td in b1.values())
    n_1k = sum(len(td.get("EVT", [])) for td in k1.values())
    n_fr = sum(len(v) for v in fifo_resets.values())
    n_sd = sum(len(v) for v in silent_drops.values())
    print(f"EVT: {n_evt:,}  Gap: {n_gap:,}  SD: {n_fsd:,}  |  1K: {n_1k:,}  |  FR: {n_fr}  SD: {n_sd}")

    trigger_met = parse_met_or_utc(args.trigger) if args.trigger else None
    plot_lightcurve(b1, k1, fifo_resets, silent_drops,
                    trigger_met=trigger_met, bin_width=args.bin,
                    output=args.output, epoch=args.epoch)


if __name__ == "__main__":
    main()
