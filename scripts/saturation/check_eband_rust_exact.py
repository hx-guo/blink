#!/usr/bin/env python3
"""Strongest verification of the Rust band-free port: per-filler exact match.

Takes the Rust filler timestamps (events_rec.csv met column) as input and, with
an independent from-scratch Python reimplementation of assign_gap_fill_channels
(sub-window even-quantile of the pooled reference in-gap channels, scattered by
the bit-reversal permutation, with the whole-gap / target-calib fallbacks),
recomputes each filler's channel and diffs it against the Rust channel.

If the Rust energy assignment were wrong in ANY step (quantile, windowing,
bit-reversal pairing, reference sourcing), the per-filler values would diverge.
Ties in the integer channel pool can shift a quantile index by 1, so a handful
of +-1 deviations at window/tie boundaries are expected; large or frequent
deviations are a real port bug.

Usage: python3 scripts/saturation/check_eband_rust_exact.py [pack_dir]
"""
import sys
import csv
from pathlib import Path
import numpy as np

PACK = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pack_260226a_v2")
BOXES = "abc"
WIN_TARGET = 0.05
CALIB_MARGIN = 0.5


def wrap(c):
    return np.where(c < 20, c + 256, c)


def unwrap(c):
    return c - 256 if c >= 256 else c


def lowdisc_ranks(n):
    phi = []
    for i in range(n):
        f, b, x = 0.0, 0.5, i
        while x > 0:
            f += (x & 1) * b
            x >>= 1
            b *= 0.5
        phi.append(f)
    order = sorted(range(n), key=lambda k: phi[k])
    ranks = [0] * n
    for k, idx in enumerate(order):
        ranks[idx] = k
    return ranks


def quantile_value(sorted_arr, ell, n):
    if len(sorted_arr) == 0:
        return 0
    q = (ell + 0.5) / max(n, 1)
    idx = min(int(q * len(sorted_arr)), len(sorted_arr) - 1)
    return int(sorted_arr[idx])


def load_obs_clean(b):
    """Wrapped-channel obs events, sorted by met, with a mask of events NOT in
    this box's own resets (mirrors sorted_pairs' unreliable exclusion)."""
    met, ch = [], []
    with open(PACK / f"box_{b}/events_obs.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[5] == "0":  # is_second == 0
                met.append(float(row[0]))
                ch.append(int(row[1]))
    met = np.asarray(met)
    chw = wrap(np.asarray(ch))
    o = np.argsort(met)
    met, chw = met[o], chw[o]
    resets = load_resets(b)
    clean = np.ones(len(met), dtype=bool)
    for s, e in resets:
        clean &= ~((met >= s) & (met <= e))
    return met, chw, clean


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
    met = np.asarray(met)
    ch = np.asarray(ch)
    o = np.argsort(met)
    return met[o], ch[o]


def chan_in(met, chw, clean, lo, hi):
    a, b = np.searchsorted(met, (lo, hi))
    sel = clean[a:b]
    v = chw[a:b][sel]
    v = np.sort(v)
    return v


obs = {b: load_obs_clean(b) for b in BOXES}

total = 0
exact = 0
devs = []
for tgt in BOXES:
    rmet, rch = load_rec(tgt)
    tmet, tchw, tclean = obs[tgt]
    refs = [b for b in BOXES if b != tgt]
    for g_lo, g_hi in load_resets(tgt):
        m = (rmet > g_lo) & (rmet < g_hi)
        filler_met = rmet[m]
        rust_ch = rch[m]
        n = len(filler_met)
        if n == 0:
            continue

        whole_gap = np.sort(
            np.concatenate([chan_in(*obs[rb], g_lo, g_hi) for rb in refs])
        )
        calib = np.sort(
            np.concatenate(
                [
                    chan_in(tmet, tchw, tclean, g_lo - CALIB_MARGIN, g_lo),
                    chan_in(tmet, tchw, tclean, g_hi, g_hi + CALIB_MARGIN),
                ]
            )
        )

        d = g_hi - g_lo
        n_win = max(1, round(d / WIN_TARGET))
        py_ch = np.zeros(n, dtype=int)
        for wi in range(n_win):
            w_lo = g_lo + d * wi / n_win
            w_hi = g_hi if wi + 1 == n_win else g_lo + d * (wi + 1) / n_win
            s = np.searchsorted(filler_met, w_lo, side="left")
            e = n if wi + 1 == n_win else np.searchsorted(filler_met, w_hi, side="left")
            n_w = e - s
            if n_w == 0:
                continue
            src = np.sort(
                np.concatenate([chan_in(*obs[rb], w_lo, w_hi) for rb in refs])
            )
            spectrum = src if len(src) else (whole_gap if len(whole_gap) else calib)
            ranks = lowdisc_ranks(n_w)
            for k in range(n_w):
                py_ch[s + k] = unwrap(quantile_value(spectrum, ranks[k], n_w))

        total += n
        diff = np.abs(py_ch - rust_ch)
        exact += int((diff == 0).sum())
        devs.extend(diff[diff != 0].tolist())

devs = np.asarray(devs)
print(f"pack: {PACK}")
print(f"fillers compared : {total}")
print(f"exact match      : {exact}  ({100 * exact / total:.3f}%)")
if len(devs):
    print(
        f"deviations       : n={len(devs)}  "
        f"median={np.median(devs):.0f}  p99={np.percentile(devs, 99):.0f}  "
        f"max={devs.max()}"
    )
else:
    print("deviations       : none — byte-for-byte identical to Python reimpl")
# PASS: >=99% exact and no large systematic offset (ties give <= a few channels)
ok = (exact / total >= 0.99) and (len(devs) == 0 or np.percentile(devs, 99) <= 3)
print("PASS" if ok else "FAIL")
