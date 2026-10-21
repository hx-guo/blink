#!/usr/bin/env python3
"""Plot each silent drop with surrounding events in the same packet."""

import subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRBS = [
    {"name": "grb200415a", "epoch": "2020-04-15T08", "trigger": "2020-04-15T08:48:08"},
    {"name": "grb221009a", "epoch": "2022-10-09T13", "trigger": "2022-10-09T13:17:02"},
    {"name": "grb260226a", "epoch": "2026-02-26T10", "trigger": "2026-02-26T10:37:53"},
]
BOXES = ["a", "b", "c"]

from datetime import datetime, timezone, timedelta
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)


def parse_met(s):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (dt - MET_EPOCH).total_seconds()


def run_cli(epoch, box, *args):
    cmd = ["./target/release/blink", "sat", epoch, "--box", box] + list(args)
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


def load_detect(epoch, box):
    """Load both FIFO resets and silent drops from detect output."""
    proc = run_cli(epoch, box, "detect")
    drops = []
    resets = []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if p[0] == "box":
            continue
        if p[1] == "FifoReset":
            resets.append({"start": float(p[2]), "stop": float(p[3]), "gap_s": float(p[4])})
        elif p[1] == "SilentDrop":
            drops.append({
            "start": float(p[2]),
            "stop": float(p[3]),
            "dt": float(p[4]),
            "pkt": int(p[5]),
            "evt": int(p[6]),
            "n_lost": int(p[7]),
            "log10p": float(p[8]),
        })
    return drops, resets


def build_pkt_events(events, t_ref):
    pkts = {}
    for met, typ, pkt, evt in events:
        t = met - t_ref
        if pkt not in pkts:
            pkts[pkt] = []
        pkts[pkt].append((t, evt, typ))
    return pkts


def plot_drop(idx, drop, all_drops, all_resets, pkt_events, t_ref, outdir, grb_name, box):
    C0, C1, C2 = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)
    C3 = plt.cm.tab10(3)  # red for FIFO resets

    pkt = drop["pkt"]
    if pkt not in pkt_events:
        return

    # Zoom: time window centered on the drop, ±20× drop duration (min ±5ms)
    drop_t_lo = drop["start"] - t_ref
    drop_t_hi = drop["stop"] - t_ref
    drop_dt = drop["dt"]
    margin = max(0.005, drop_dt * 20)
    view_lo = drop_t_lo - margin
    view_hi = drop_t_hi + margin

    # Collect events in the view window from this packet and neighbors
    nearby_pkts = [p for p in sorted(pkt_events.keys()) if abs(p - pkt) <= 1]
    evt_times = []
    sec_times = []
    pkt_colors = [C0, C2]

    for i, pk in enumerate(nearby_pkts):
        color = pkt_colors[i % 2]
        for t, ev, typ in pkt_events[pk]:
            if t < view_lo or t > view_hi:
                continue
            if typ == "SEC":
                sec_times.append(t)
            else:
                evt_times.append((t, color))

    if not evt_times and not sec_times:
        return

    all_t = [t for t, _ in evt_times] + sec_times
    t_lo = min(all_t) - drop_dt
    t_hi = max(all_t) + drop_dt

    pkt_spans = [(t_lo, t_hi, pkt, len(evt_times), len(sec_times), C0)]

    fig, (ax, ax_info) = plt.subplots(2, 1, figsize=(18, 3), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})

    # FIFO resets visible in this time range
    n_fifo = 0
    for r in all_resets:
        r_lo = r["start"] - t_ref
        r_hi = r["stop"] - t_ref
        if r_hi >= t_lo and r_lo <= t_hi:
            for a in (ax, ax_info):
                a.axvspan(r_lo, r_hi, alpha=0.2, color=C3, zorder=0)
            n_fifo += 1
            t_mid = (r_lo + r_hi) / 2
            ax.text(t_mid, 0.5, f"FIFO\n{r['gap_s']*1000:.0f}ms",
                    fontsize=5, ha="center", va="center", color=C3, alpha=0.7, zorder=5)

    # All silent drops visible in this time range
    n_visible = 0
    for d in all_drops:
        d_lo = d["start"] - t_ref
        d_hi = d["stop"] - t_ref
        if d_hi >= t_lo and d_lo <= t_hi:
            is_current = (d is drop)
            for a in (ax, ax_info):
                a.axvspan(d_lo, d_hi, alpha=0.3 if is_current else 0.12, color=C1, zorder=0)
            n_visible += 1
            t_mid = (d_lo + d_hi) / 2
            ax.text(t_mid, 0.5,
                    f"{d['dt']*1000:.1f}ms\n~{d['n_lost']}",
                    fontsize=6 if not is_current else 7, ha="center", va="center",
                    color=C1, fontweight="bold" if is_current else "normal",
                    alpha=1.0 if is_current else 0.6, zorder=5)

    # Packet background bands
    for t_min, t_max, pk, n_evt, n_sec, color in pkt_spans:
        ax.axvspan(t_min, t_max, alpha=0.06, color=color, zorder=0)

    # EVT events
    for t, color in evt_times:
        ax.vlines(t, 0.05, 0.95, colors=color, alpha=0.6, lw=0.5, zorder=2)

    # SEC events
    if sec_times:
        ax.vlines(sec_times, 0.0, 1.0, colors="red", alpha=0.9, lw=1.5, zorder=3)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=C1, alpha=0.25, label=f"Silent drop ({n_visible})"),
        Patch(facecolor=C3, alpha=0.2, label=f"FIFO reset ({n_fifo})"),
        Line2D([], [], color=C0, lw=1.5, label=f"EVT ({len(evt_times)})"),
    ]
    if sec_times:
        handles.append(Line2D([], [], color="red", lw=2, label=f"SEC ({len(sec_times)})"))
    ax.legend(handles=handles, loc="lower right", fontsize=7)

    drop_t = (drop["start"] + drop["stop"]) / 2 - t_ref
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(
        f"{grb_name} Box {box.upper()}  Silent Drop #{idx+1}/{len(all_drops)}  "
        f"T+{drop_t:.4f}s  pkt {pkt}  log10p={drop['log10p']:.1f}",
        fontsize=12, fontweight="bold",
    )

    # Lower panel: packet info
    for t_min, t_max, pk, n_evt, n_sec, color in pkt_spans:
        t_mid = (t_min + t_max) / 2
        label = f"pkt {pk}"
        if n_sec > 0:
            label += f"\n{n_evt}E+{n_sec}S"
        else:
            label += f"\n{n_evt}E"
        ax_info.text(t_mid, 0.5, label, fontsize=6, ha="center", va="center",
                     color=color, fontweight="bold")
        ax_info.plot([t_min, t_min, t_max, t_max], [0.9, 0.95, 0.95, 0.9],
                     color=color, lw=0.8, alpha=0.5)

    ax_info.set_ylim(0, 1)
    ax_info.set_yticks([])
    ax_info.set_xlim(t_lo, t_hi)
    ax_info.set_xlabel("Time − T₀ (s)", fontsize=11)

    plt.tight_layout()
    out = os.path.join(outdir, f"silent_drop_{idx+1:04d}.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    return out


def process_grb_box(grb, box):
    grb_name = grb["name"]
    epoch = grb["epoch"]
    t_ref = parse_met(grb["trigger"])

    outdir = os.path.join("plots/silent_drops", f"{grb_name}_{box}")
    os.makedirs(outdir, exist_ok=True)

    print(f"  Loading solve...", file=sys.stderr)
    events = load_solve(epoch, box)
    print(f"    {len(events)} events", file=sys.stderr)

    print(f"  Loading detect...", file=sys.stderr)
    drops, resets = load_detect(epoch, box)
    print(f"    {len(drops)} silent drops, {len(resets)} FIFO resets", file=sys.stderr)

    if not drops:
        print(f"    No drops, skipping", file=sys.stderr)
        return

    pkt_events = build_pkt_events(events, t_ref)

    print(f"  Plotting {len(drops)} drops to {outdir}/...", file=sys.stderr)
    for i, d in enumerate(drops):
        plot_drop(i, d, drops, resets, pkt_events, t_ref, outdir, grb_name, box)

    print(f"    Done: {len(drops)} plots", file=sys.stderr)


def main():
    for grb in GRBS:
        for box in BOXES:
            print(f"{grb['name']} Box {box.upper()}:", file=sys.stderr)
            process_grb_box(grb, box)


if __name__ == "__main__":
    main()
