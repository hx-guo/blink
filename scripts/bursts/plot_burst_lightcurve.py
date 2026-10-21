#!/usr/bin/env python3
"""Quick observed-vs-reconstructed HXMT/HE light curve for an arbitrary burst.

Runs `blink sat reconstruct <UTC> --before B --after A`, parses the per-event
output (box,type,met,channel,pulse_width,...), and plots the summed-over-boxes
light curve: observed (EVT) and observed+reconstructed (EVT+FILL_GAP), with the
FIFO-reset gaps shaded. A NaI-selected panel (pulse_width in [54,70]) is added
when --nai is passed.

Run from blink/ with HXMT_1B_DIR pointing at the pulled data:
    HXMT_1B_DIR=data/1B .venv/bin/python scripts/bursts/plot_burst_lightcurve.py \
        --t0 2025-09-19T00:29:15 --before 20 --after 140 --bin 0.5 \
        --title "GRB 250919A" -o GRB250919A_lc.png
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_reconstruct(t0_utc, before, after):
    cmd = ["./target/release/blink", "sat", "reconstruct", t0_utc,
           "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stderr.strip().split("\n")[-12:]:
        print(f"    {line}", file=sys.stderr)
    rows = {"obs": [], "fill": []}
    obs_pw, fill_pw = [], []
    ncols = None
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 5 or p[0] == "box":
            continue
        typ, met, pw = p[1], float(p[2]), int(p[4])
        if typ == "EVT":
            rows["obs"].append(met); obs_pw.append(pw)
        elif typ == "FILL_GAP":
            rows["fill"].append(met); fill_pw.append(pw)
    return (np.asarray(rows["obs"]), np.asarray(obs_pw, dtype=int),
            np.asarray(rows["fill"]), np.asarray(fill_pw, dtype=int))


def met_of_utc(t0_utc):
    """HXMT MET of a UTC ISO string. MET = continuous TAI seconds since the
    2012-01-01 mission epoch (leap-second-aware). Anchored by
    MET(2017-07-01T00:00:00 UTC) = 173491203 (blink util.rs test)."""
    from astropy.time import Time
    return (Time(t0_utc, scale="utc").unix_tai
            - Time("2012-01-01T00:00:00", scale="utc").unix_tai)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True, help="trigger UTC, e.g. 2025-09-19T00:29:15")
    ap.add_argument("--csv", default=None,
                    help="read a pre-dumped reconstruct CSV instead of re-running blink")
    ap.add_argument("--before", type=float, default=20.0)
    ap.add_argument("--after", type=float, default=120.0)
    ap.add_argument("--bin", type=float, default=0.5)
    ap.add_argument("--nai", action="store_true", help="add a NaI-selected panel")
    ap.add_argument("--title", default="")
    ap.add_argument("-o", "--output", default="burst_lc.png")
    args = ap.parse_args()

    if args.csv:
        o, opw, f, fpw = [], [], [], []
        with open(args.csv) as fh:
            for line in fh:
                p = line.split(",")
                if len(p) < 5 or p[0] == "box":
                    continue
                typ, met, pw = p[1], float(p[2]), int(p[4])
                if typ == "EVT":
                    o.append(met); opw.append(pw)
                elif typ == "FILL_GAP":
                    f.append(met); fpw.append(pw)
        obs, obs_pw = np.asarray(o), np.asarray(opw, dtype=int)
        fill, fill_pw = np.asarray(f), np.asarray(fpw, dtype=int)
        print(f"  from CSV: {len(obs):,} obs + {len(fill):,} fill", file=sys.stderr)
    else:
        obs, obs_pw, fill, fill_pw = run_reconstruct(args.t0, args.before, args.after)
    if len(obs) == 0:
        print("  NO observed events in window — burst likely Earth-occulted or "
              "not in HE FoV for this hour.", file=sys.stderr)
        sys.exit(2)

    trig_met = met_of_utc(args.t0)
    t_obs = obs - trig_met
    t_fill = fill - trig_met

    edges = np.arange(-args.before, args.after + args.bin, args.bin)
    x = edges[:-1] + args.bin / 2
    all_t = np.concatenate([obs, fill])

    def lc(mets):
        return np.histogram(mets - trig_met, bins=edges)[0] / args.bin

    npan = 2 if args.nai else 1
    fig, axes = plt.subplots(npan, 1, figsize=(11, 4.2 * npan), sharex=True, squeeze=False)
    axes = axes[:, 0]

    def draw(ax, o_t, a_t, label_suffix, fill_frac):
        r_obs = np.histogram(o_t, bins=edges)[0] / args.bin
        r_all = np.histogram(a_t, bins=edges)[0] / args.bin
        ax.fill_between(x, r_obs, r_all, step="mid", color="C1", alpha=0.30, zorder=1)
        ax.step(x, r_obs, where="mid", color="navy", lw=0.9,
                label="observed (EVT)", zorder=3)
        ax.step(x, r_all, where="mid", color="C1", lw=1.0,
                label=f"observed + reconstructed{label_suffix}", zorder=4)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("count rate (evt/s, 3 boxes)")
        ax.legend(loc="upper right", fontsize=8)
        ax.margins(x=0)

    draw(axes[0], t_obs, all_t - trig_met,
         f" (+{len(fill):,} filled)", len(fill) / max(1, len(all_t)))
    axes[0].text(0.01, 0.92, "all NaI+CsI events", transform=axes[0].transAxes,
                 fontweight="bold")

    if args.nai:
        nai_o = obs[(obs_pw >= 54) & (obs_pw <= 70)]
        nai_f = fill[(fill_pw >= 54) & (fill_pw <= 70)]
        draw(axes[1], nai_o - trig_met, np.concatenate([nai_o, nai_f]) - trig_met,
             f" (+{len(nai_f):,} filled)", 0)
        axes[1].text(0.01, 0.92, "NaI-selected (pw 54-70)",
                     transform=axes[1].transAxes, fontweight="bold")

    axes[-1].set_xlabel(f"time since T0 (s)   [T0 = {args.t0} UTC]")
    title = args.title or args.t0
    axes[0].set_title(
        f"{title}  —  HXMT/HE 1B reconstruction  "
        f"[{args.bin*1e3:.0f} ms bins;  {len(obs):,} obs + {len(fill):,} filled events]")
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"  obs={len(obs):,}  filled={len(fill):,}  "
          f"fill_frac={len(fill)/max(1,len(all_t))*100:.2f}%", file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
