#!/usr/bin/env python3
"""Plot FIFO reset with all 3 boxes: event strips + info rows + light curve.

Usage (quick debug, single reset):
    python3 scripts/saturation/plot_fifo_resets.py --grb 2 --idx 1 40
"""

import subprocess, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

GRBS = [
    {"name": "grb200415a", "epoch": "2020-04-15T08", "trigger": "2020-04-15T08:48:08"},
    {"name": "grb221009a", "epoch": "2022-10-09T13", "trigger": "2022-10-09T13:17:02"},
    {"name": "grb260226a", "epoch": "2026-02-26T10", "trigger": "2026-02-26T10:37:53"},
]
BOXES = ["a", "b", "c"]
BOX_LABELS = ["A", "B", "C"]
CONTEXT_PKTS = 3
LC_BIN = 0.001

from datetime import datetime, timezone, timedelta
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

C_EVT = [plt.cm.tab10(0), plt.cm.tab10(2)]
C_FILL = "#D62728"
C_SEC = "#E377C2"
C_GAP = plt.cm.tab10(1)
C_LC = ["#2166AC", "#2CA02C", "#9467BD"]


def parse_met(s):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (dt - MET_EPOCH).total_seconds()


def run_cli(epoch, box, *args):
    cmd = ["./target/release/blink_cli", "sat", epoch, "--box", box] + list(args)
    env = os.environ.copy()
    env["HXMT_1B_DIR"] = "data/1B"
    env["HXMT_1K_DIR"] = "data/1K"
    env["MAX_SEC_GAP"] = "999"
    env["PASS1_ONLY"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def load_solve(epoch, box):
    proc = run_cli(epoch, box, "solve")
    events = []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 6 or p[0] == "box":
            continue
        events.append((float(p[2]), p[1], int(p[4]), int(p[5])))
    return events


def load_resets(epoch, box):
    proc = run_cli(epoch, box, "detect")
    resets = []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if p[0] == "box" or p[1] != "FifoReset":
            continue
        resets.append({
            "start": float(p[2]), "stop": float(p[3]),
            "gap_s": float(p[4]), "pkt": int(p[5]),
            "n_lost": int(p[7]), "box": box,
        })
    return resets


def load_reconstruct(epoch, box, trigger):
    proc = run_cli(epoch, box, "reconstruct", trigger, "--before", "3600", "--after", "3600")
    fill_mets = []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 3 or p[0] == "box":
            continue
        if p[1] == "FILL_GAP":
            fill_mets.append(float(p[2]))
    return np.array(fill_mets)


def build_pkt_events(events, t_ref):
    pkts = {}
    for met, typ, pkt, evt in events:
        t = met - t_ref
        if pkt not in pkts:
            pkts[pkt] = []
        pkts[pkt].append((t, evt, typ))
    return pkts


def load_all_data(grb):
    epoch = grb["epoch"]
    t_ref = parse_met(grb["trigger"])
    all_data = []
    for box in BOXES:
        print(f"  Loading {box.upper()}...", file=sys.stderr)
        events = load_solve(epoch, box)
        resets = load_resets(epoch, box)
        fill_abs = load_reconstruct(epoch, box, grb["trigger"])
        fill_rel = fill_abs - t_ref if len(fill_abs) > 0 else np.array([])
        pkt_events = build_pkt_events(events, t_ref)
        all_pkt_ids = sorted(pkt_events.keys())
        obs_mets = np.array([met - t_ref for met, typ, _, _ in events if typ != "SEC"])
        print(f"    {len(events)} events, {len(resets)} resets, {len(fill_rel)} filled",
              file=sys.stderr)
        all_data.append({
            "pkt_events": pkt_events, "all_pkt_ids": all_pkt_ids,
            "resets": resets, "fill_mets": fill_rel, "obs_mets": obs_mets,
        })
    return all_data, t_ref


def get_pkts_by_index(pkt_events, all_pkt_ids, reset_pkt):
    """Get nearby packets around a reset packet by index."""
    if reset_pkt not in all_pkt_ids:
        return []
    pos = all_pkt_ids.index(reset_pkt)
    lo = max(0, pos - CONTEXT_PKTS)
    hi = min(len(all_pkt_ids), pos + CONTEXT_PKTS + 2)
    return all_pkt_ids[lo:hi]


def get_pkts_by_time(pkt_events, all_pkt_ids, t_lo, t_hi):
    """Get packets overlapping a time window."""
    result = []
    for pkt in all_pkt_ids:
        if pkt not in pkt_events:
            continue
        ts = [e[0] for e in pkt_events[pkt]]
        if max(ts) >= t_lo and min(ts) <= t_hi:
            result.append(pkt)
    return result


def collect_events(pkt_events, pkt_list):
    """Collect events from a list of packets, return (evt_times, sec_times, pkt_spans)."""
    evt_times = []
    sec_times = []
    pkt_spans = []
    for i, pkt in enumerate(pkt_list):
        if pkt not in pkt_events:
            continue
        evts = sorted(pkt_events[pkt])
        color = C_EVT[i % 2]
        n_evt = n_sec = 0
        for t, ev, typ in evts:
            if typ == "SEC":
                sec_times.append(t)
                n_sec += 1
            else:
                evt_times.append((t, color))
                n_evt += 1
        if evts:
            ts = [e[0] for e in evts]
            pkt_spans.append((min(ts), max(ts), pkt, n_evt, n_sec, color))
    return evt_times, sec_times, pkt_spans


def draw_event_strip(ax, evt_times, sec_times, fill_visible, box_label):
    """Draw one event strip panel."""
    for t, color in evt_times:
        ax.vlines(t, 0.2, 0.8, colors=color, alpha=0.6, lw=0.4, zorder=2)
    if sec_times:
        ax.vlines(sec_times, 0.1, 0.9, colors=C_SEC, alpha=0.9, lw=1.2, zorder=3)
    if len(fill_visible) > 0:
        ax.vlines(fill_visible, 0.2, 0.8, colors=C_FILL, alpha=0.7, lw=0.4, zorder=4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(f"Box {box_label}", fontsize=9, fontweight="bold", rotation=0,
                  labelpad=25, va="center")


def draw_info_row(ax, pkt_spans, resets_visible, box_color):
    """Draw packet info + gap labels in an info row."""
    # Packet brackets and labels
    for t_min, t_max, pkt, n_evt, n_sec, color in pkt_spans:
        t_mid = (t_min + t_max) / 2
        label = f"{pkt}"
        if n_sec > 0:
            label += f"\n{n_evt}E+{n_sec}S"
        else:
            label += f"\n{n_evt}E"
        ax.text(t_mid, 0.65, label, fontsize=5, ha="center", va="center",
                color=color, fontweight="bold")
        ax.plot([t_min, t_min, t_max, t_max], [0.92, 0.98, 0.98, 0.92],
                color=color, lw=0.6, alpha=0.5)

    # Gap labels
    for r in resets_visible:
        t_mid = (r[0] + r[1]) / 2
        label = f"{r[2]*1000:.1f}ms +{r[3]}"
        ax.text(t_mid, 0.25, label, fontsize=6, ha="center", va="center",
                color=box_color, fontweight="bold", alpha=0.9)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("", fontsize=1)


def plot_reset(idx, reset, target_box_idx, all_data, t_ref, outdir, grb_name):
    td = all_data[target_box_idx]

    # Get time window from target box's nearby packets
    nearby = get_pkts_by_index(td["pkt_events"], td["all_pkt_ids"], reset["pkt"])
    if not nearby:
        return
    init_evt, init_sec, _ = collect_events(td["pkt_events"], nearby)
    if not init_evt and not init_sec:
        return

    all_t = [t for t, _ in init_evt] + init_sec
    t_lo = min(all_t) - 0.002
    t_hi = max(all_t) + 0.002

    # Figure: 3×(event strip + info row) + light curve = 7 rows
    fig, axes = plt.subplots(7, 1, figsize=(20, 9), sharex=True,
        gridspec_kw={
            "height_ratios": [2, 0.8, 2, 0.8, 2, 0.8, 2],
            "hspace": 0.02,
        })
    ax_evts = [axes[0], axes[2], axes[4]]
    ax_infos = [axes[1], axes[3], axes[5]]
    ax_lc = axes[6]

    # Shade all FIFO resets on all panels
    for bi in range(3):
        for r in all_data[bi]["resets"]:
            r_lo = r["start"] - t_ref
            r_hi = r["stop"] - t_ref
            if r_hi >= t_lo and r_lo <= t_hi:
                for ax in list(axes):
                    ax.axvspan(r_lo, r_hi, alpha=0.18, color=C_GAP, zorder=0, linewidth=0)

    # Draw each box
    legend_counts = {}
    for bi in range(3):
        bd = all_data[bi]
        ax_e = ax_evts[bi]
        ax_i = ax_infos[bi]

        pkts = get_pkts_by_time(bd["pkt_events"], bd["all_pkt_ids"], t_lo, t_hi)
        evt_times, sec_times, pkt_spans = collect_events(bd["pkt_events"], pkts)

        # Packet background bands on event strip
        for t_min, t_max, pkt, n_evt, n_sec, color in pkt_spans:
            ax_e.axvspan(t_min, t_max, alpha=0.05, color=color, zorder=0)

        # Fill events
        fill = bd["fill_mets"]
        fill_visible = np.array([])
        if len(fill) > 0:
            fill_visible = fill[(fill >= t_lo) & (fill <= t_hi)]

        draw_event_strip(ax_e, evt_times, sec_times, fill_visible, BOX_LABELS[bi])

        # Info row: packets + gap labels for this box
        fill = bd["fill_mets"]
        resets_visible = []
        for r in bd["resets"]:
            r_lo = r["start"] - t_ref
            r_hi = r["stop"] - t_ref
            if r_hi >= t_lo and r_lo <= t_hi:
                # Count actual filled events in this gap
                n_filled = int(np.sum((fill >= r_lo) & (fill <= r_hi))) if len(fill) > 0 else 0
                resets_visible.append((r_lo, r_hi, r["gap_s"], n_filled))
        draw_info_row(ax_i, pkt_spans, resets_visible, C_LC[bi])

        legend_counts[bi] = (len(evt_times), len(sec_times), len(fill_visible))

    # Title
    t_center = (reset["start"] + reset["stop"]) / 2.0 - t_ref
    n_target = len(all_data[target_box_idx]["resets"])
    fig.suptitle(
        f"{grb_name} Box {BOX_LABELS[target_box_idx]}  "
        f"FIFO Reset #{idx+1}/{n_target}  T+{t_center:.4f}s",
        fontsize=13, fontweight="bold", y=0.99)

    # Light curve
    edges = np.arange(t_lo, t_hi + LC_BIN, LC_BIN)
    if len(edges) < 2:
        plt.close()
        return
    x = edges[:-1]
    for bi in range(3):
        bd = all_data[bi]
        obs = bd["obs_mets"]
        obs_in = obs[(obs >= t_lo) & (obs <= t_hi)]
        fill = bd["fill_mets"]
        fill_in = fill[(fill >= t_lo) & (fill <= t_hi)] if len(fill) > 0 else np.array([])
        combined = np.concatenate([obs_in, fill_in]) if len(fill_in) > 0 else obs_in
        rate = np.histogram(combined, bins=edges)[0] / LC_BIN
        ax_lc.step(x, rate, where="post", color=C_LC[bi], lw=0.8, alpha=0.8, zorder=2 + bi)

    ax_lc.set_ylabel("Rate\n(evt/s)", fontsize=9, rotation=0, labelpad=25, va="center")
    ax_lc.set_ylim(bottom=0)
    ax_lc.grid(alpha=0.15)
    ax_lc.set_xlim(t_lo, t_hi)
    ax_lc.set_xlabel("Time − T₀ (s)", fontsize=11)

    # Legend
    handles = [
        Patch(facecolor=C_GAP, alpha=0.2, label="FIFO reset"),
        Line2D([], [], color=C_FILL, lw=1.5, label="Filled"),
        Line2D([], [], color=C_SEC, lw=1.5, label="SEC"),
    ]
    for bi in range(3):
        n_e, n_s, n_f = legend_counts[bi]
        lbl = f"Box {BOX_LABELS[bi]}: {n_e}E"
        if n_s: lbl += f" {n_s}S"
        if n_f: lbl += f" {n_f}F"
        handles.append(Line2D([], [], color=C_LC[bi], lw=1.5, label=lbl))
    ax_lc.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

    out = os.path.join(outdir, f"fifo_reset_{idx+1:04d}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grb", type=int, default=2, help="GRB index (0/1/2)")
    parser.add_argument("--box", type=int, default=0, help="Target box (0=A,1=B,2=C)")
    parser.add_argument("--idx", type=int, nargs="+", default=None,
                        help="Reset indices to plot (1-based), default=all")
    args = parser.parse_args()

    grb = GRBS[args.grb]
    print(f"Loading {grb['name']}...", file=sys.stderr)
    all_data, t_ref = load_all_data(grb)

    target_bi = args.box
    resets = all_data[target_bi]["resets"]
    outdir = os.path.join("plots/fifo_resets", f"{grb['name']}_{BOXES[target_bi]}")
    os.makedirs(outdir, exist_ok=True)

    indices = [i - 1 for i in args.idx] if args.idx else range(len(resets))
    for i in indices:
        if 0 <= i < len(resets):
            out = plot_reset(i, resets[i], target_bi, all_data, t_ref, outdir, grb["name"])
            if out:
                print(f"  Saved: {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
