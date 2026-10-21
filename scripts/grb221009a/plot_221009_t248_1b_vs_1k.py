#!/usr/bin/env python3
"""221009A T+248 to T+268 (20s window) — 1B LIS vs 1K, 3-box panel.

Aim: check whether 1K also fails in the wrap-uniqueness validation window
(Fig 3 in paper), beyond the well-known Box C T+82-T+90 failure (Fig 4).
"""
import os, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

EPOCH    = "2022-10-09T13"
TRIGGER  = "2022-10-09T13:17:02"
BEFORE   = 10.0
AFTER    = 280.0    # covers T+248 to T+268
BIN      = 0.1      # 100 ms
ZOOM_LO  = 248.0
ZOOM_HI  = 268.0
MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)


def parse_met(s):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (dt - MET_EPOCH).total_seconds()


def run_cli(subcmd):
    cmd = ["./target/release/blink", "sat", EPOCH, subcmd,
           TRIGGER, "--before", str(BEFORE), "--after", str(AFTER)]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    boxes = {}
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 3 or parts[0] == "box":
            continue
        box, typ, met = parts[0], parts[1], float(parts[2])
        if typ == "SEC":
            continue
        boxes.setdefault(box, []).append(met)
    return {b: np.array(v) for b, v in boxes.items()}


def main():
    t_ref = parse_met(TRIGGER)
    print("Loading 1B (LIS-reconstructed)...", file=sys.stderr)
    b1 = run_cli("solve")
    print("Loading 1K...", file=sys.stderr)
    k1 = run_cli("solve1k")

    edges = np.arange(t_ref + ZOOM_LO, t_ref + ZOOM_HI + BIN, BIN)
    x = edges[:-1] - t_ref

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    colors_1b = {"A": "#2166AC", "B": "#D6604D", "C": "#1B7837"}
    fill_1b   = {"A": "#92C5DE", "B": "#F4A582", "C": "#A6D96A"}

    for ax, box in zip(axes, ["A", "B", "C"]):
        evt_1b = b1.get(box, np.array([]))
        evt_1k = k1.get(box, np.array([]))
        win_1b = evt_1b[(evt_1b >= edges[0]) & (evt_1b < edges[-1])]
        win_1k = evt_1k[(evt_1k >= edges[0]) & (evt_1k < edges[-1])]
        r_1k = np.histogram(win_1k, bins=edges)[0] / BIN
        r_1b = np.histogram(win_1b, bins=edges)[0] / BIN

        ax.fill_between(x, r_1k, step="post", color="#DDDDDD",
                        alpha=0.9, edgecolor="none", zorder=1)
        ax.step(x, r_1k, where="post", color="#888888", lw=0.6,
                label=f"1K ({len(win_1k):,})", zorder=2)
        ax.fill_between(x, r_1b, step="post", color=fill_1b[box],
                        alpha=0.5, edgecolor="none", zorder=3)
        ax.step(x, r_1b, where="post", color=colors_1b[box], lw=0.7,
                label=f"1B LIS ({len(win_1b):,})", zorder=4)

        ax.set_ylabel(f"Box {box}\nevt/s", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.15)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel(
        f"Time − T₀ (s)    [T₀ = {TRIGGER} UTC; bin = {BIN*1000:.0f} ms]",
        fontsize=11)
    axes[0].set_xlim(ZOOM_LO, ZOOM_HI)
    fig.suptitle(f"GRB 221009A  T₀+{ZOOM_LO:.0f} to T₀+{ZOOM_HI:.0f} s   "
                 f"1B LIS vs 1K  (wrap-uniqueness window)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "plots/221009a_t248_268_1b_vs_1k.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
