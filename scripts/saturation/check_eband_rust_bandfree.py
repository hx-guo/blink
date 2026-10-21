#!/usr/bin/env python3
"""Cross-check the Rust band-free gap-fill channels against an independent
reimplementation on the 260226A pack: for every real reset gap, the recovered
filler channels (events_rec.csv) should reproduce the pooled in-gap channel
distribution of the other two boxes (events_obs.csv), since the Rust code draws
deterministic even-quantiles from exactly that pool. PASS = small KS distance.

Usage: python3 scripts/saturation/check_eband_rust_bandfree.py [pack_dir]
"""
import sys
import csv
from pathlib import Path
import numpy as np

PACK = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pack_260226a_v2")
BOXES = "abc"


def load_obs(b):
    met, ch = [], []
    with open(PACK / f"box_{b}/events_obs.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[5] == "0":  # is_second == 0
                met.append(float(row[0]))
                ch.append(int(row[1]))
    met = np.asarray(met)
    ch = np.asarray(ch)
    o = np.argsort(met)
    return met[o], ch[o]


def load_resets(b):
    out = []
    with open(PACK / f"box_{b}/resets.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            out.append((float(row[0]), float(row[1])))
    return out


def load_rec(b):
    met, ch = [], []
    with open(PACK / f"box_{b}/events_rec.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            met.append(float(row[0]))
            ch.append(int(row[1]))
    return np.asarray(met), np.asarray(ch)


def ks(a, b):
    a = np.sort(a)
    b = np.sort(b)
    xs = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, xs, side="right") / len(a)
    cb = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def in_any(t, ivs):
    return any(s <= t <= e for s, e in ivs)


obs = {b: load_obs(b) for b in BOXES}
res = {b: load_resets(b) for b in BOXES}

ks_list = []
for tgt in BOXES:
    rmet, rch = load_rec(tgt)
    refs = [b for b in BOXES if b != tgt]
    for g_lo, g_hi in res[tgt]:
        rec_sel = rch[(rmet > g_lo) & (rmet < g_hi)]
        if len(rec_sel) == 0:
            continue
        pool = []
        for rb in refs:
            m, c = obs[rb]
            i, j = np.searchsorted(m, (g_lo, g_hi))
            for k in range(i, j):
                if not in_any(m[k], res[rb]):
                    pool.append(c[k])
        pool = np.asarray(pool)
        if len(pool) < 30:
            continue
        ks_list.append(ks(rec_sel, pool))

ks_arr = np.asarray(ks_list)
print(f"pack: {PACK}")
print(f"checked {len(ks_arr)} gaps")
print(
    f"KS(recovered vs pooled-ref in-gap): "
    f"median={np.median(ks_arr):.4f}  p90={np.percentile(ks_arr, 90):.4f}  "
    f"max={ks_arr.max():.4f}"
)
print("PASS" if np.median(ks_arr) < 0.08 else "FAIL")
