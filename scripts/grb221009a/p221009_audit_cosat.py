#!/usr/bin/env python3
"""Three-box co-saturation statistics for GRB 221009A FIFO-reset gaps.

Input: `blink sat detect` outputs for boxes a/b/c written to
data/p221009/audit/detect_{a,b,c}.txt (run with --before 50 --after 700).
For each box's gaps, measures the fraction that overlap gaps in both other
boxes (any overlap) and the fraction of gap time with no unsaturated
reference box, plus where (in T0-relative time) the co-saturated time sits.

Run from the blink repo root:
    ./.venv/bin/python scripts/grb221009a/p221009_audit_cosat.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

T0 = 339945423.0
D = Path("data/p221009/audit")
gaps = {b: pd.read_csv(D / f"detect_{b}.txt") for b in "abc"}
lines = []
def say(s=""):
    print(s); lines.append(s)

def overlaps(s, e, other):
    os_, oe = other.start_met.values, other.stop_met.values
    i = np.searchsorted(oe, s)
    return i < len(os_) and os_[i] < e

def covered_time(s, e, other):
    os_, oe = other.start_met.values, other.stop_met.values
    i = np.searchsorted(oe, s)
    tot = 0.0
    segs = []
    while i < len(os_) and os_[i] < e:
        a, b = max(s, os_[i]), min(e, oe[i])
        if b > a:
            segs.append((a, b))
        i += 1
    return segs

def inter(segs1, segs2):
    out = []
    for a1, b1 in segs1:
        for a2, b2 in segs2:
            a, b = max(a1, a2), min(b1, b2)
            if b > a:
                out.append((a, b))
    return out

for b in "abc":
    o1, o2 = [x for x in "abc" if x != b]
    g = gaps[b].sort_values("start_met")
    any_both, t_gap, t_cosat, full, half = 0, 0.0, 0.0, 0, 0
    cos_t = []
    for s, e in zip(g.start_met.values, g.stop_met.values):
        ov1 = overlaps(s, e, gaps[o1].sort_values("start_met"))
        ov2 = overlaps(s, e, gaps[o2].sort_values("start_met"))
        any_both += ov1 and ov2
        t_gap += e - s
        c = inter(covered_time(s, e, gaps[o1]), covered_time(s, e, gaps[o2]))
        ct = sum(bb - aa for aa, bb in c)
        t_cosat += ct
        if ct > 0:
            cos_t.append(((s + e) / 2 - T0, ct))
        full += ct >= 0.999 * (e - s)
        half += ct >= 0.5 * (e - s)
    n = len(g)
    say(f"Box {b.upper()}: {n} gaps; overlapping gaps in BOTH other boxes: {any_both} "
        f"({100*any_both/n:.1f}%); gap time {t_gap:.2f} s, of which no reference box "
        f"{t_cosat:.3f} s ({100*t_cosat/t_gap:.1f}%)")
    say(f"   gaps with NO reference over the whole gap (linear-ramp fallback): {full} "
        f"({100*full/n:.1f}%); over >=50% of the gap: {half} ({100*half/n:.1f}%)")
    if cos_t:
        tt = np.array(cos_t)
        h, ed = np.histogram(tt[:, 0], bins=np.arange(170, 320, 10), weights=tt[:, 1])
        say("   co-saturated time per 10-s bin (s): " +
            ", ".join(f"{int(ed[i])}:{h[i]:.2f}" for i in range(len(h)) if h[i] > 0))
    gt = (g.start_met.values - T0)
    say(f"   gap time range T0+{gt.min():.1f}..{gt.max():.1f}")

(D / "audit_cosat.txt").write_text("\n".join(lines) + "\n")
