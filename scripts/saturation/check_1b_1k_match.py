#!/usr/bin/env python3
"""Check 1B vs 1K event-level match in confirmed (non-uncertain) regions.

Compares time, channel, and det_id for each event.
"""

import subprocess
import sys
import os
import numpy as np
from astropy.io import fits

WRAP_PERIOD = 1.048576

EPOCH = "2022-10-09T13"
TRIGGER = "2022-10-09T13:17:02"
BEFORE = 50.0
AFTER = 900.0
FITS_PATH = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"

BOX_DET_RANGES = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}


def parse_met_or_utc(s):
    from datetime import datetime, timezone
    MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"]:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return (dt - MET_EPOCH).total_seconds()
        except ValueError:
            continue
    return float(s)


def load_1b(box_name):
    cmd = [
        "./target/release/blink_cli", "sat", EPOCH, "--box", box_name.lower(),
        "solve", TRIGGER, "--before", str(BEFORE), "--after", str(AFTER),
    ]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")

    print(f"  Running 1B solve for Box {box_name}...", file=sys.stdout)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stderr.strip().split("\n"):
        print(f"    {line}", file=sys.stdout)

    mets, channels, sec_mets = [], [], []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 4 or parts[0] == "box":
            continue
        typ, met, ch = parts[1], float(parts[2]), int(parts[3])
        if typ == "SEC":
            sec_mets.append(met)
        else:
            mets.append(met)
            channels.append(ch)

    return np.array(mets), np.array(channels), np.sort(sec_mets)


def load_1k(box_name):
    print(f"  Loading 1K FITS for Box {box_name}...", file=sys.stdout)
    with fits.open(FITS_PATH, memmap=True) as hdul:
        data = hdul[1].data
        times = data["Time"]
        det_ids = data["Det_ID"]
        channels = data["Channel"]

    d_lo, d_hi = BOX_DET_RANGES[box_name]
    t_ref = parse_met_or_utc(TRIGGER)
    met_lo = t_ref - BEFORE
    met_hi = t_ref + AFTER

    mask = (det_ids >= d_lo) & (det_ids <= d_hi) & (times >= met_lo) & (times <= met_hi)
    return times[mask].copy(), channels[mask].copy(), det_ids[mask].copy()


def compute_confirmed_mask(mets, sec_mets):
    mask = np.ones(len(mets), dtype=bool)
    if len(sec_mets) < 2:
        return np.zeros(len(mets), dtype=bool)
    for i in range(len(sec_mets) - 1):
        if sec_mets[i + 1] - sec_mets[i] > WRAP_PERIOD:
            mask &= ~((mets >= sec_mets[i]) & (mets <= sec_mets[i + 1]))
    mask &= (mets >= sec_mets[0]) & (mets <= sec_mets[-1])
    return mask


def match_events(met_1b, ch_1b, met_1k, ch_1k, det_1k, tol=6e-6):
    """Match events using set-based approach: build a dict of 1K events
    keyed by quantized MET, then look up each 1B event."""
    # Sort both
    order_1b = np.argsort(met_1b)
    order_1k = np.argsort(met_1k)
    m1b = met_1b[order_1b]
    c1b = ch_1b[order_1b]
    m1k = met_1k[order_1k]
    c1k = ch_1k[order_1k]
    d1k = det_1k[order_1k]

    n1b, n1k = len(m1b), len(m1k)

    # Two-pointer matching with strict tolerance
    i, j = 0, 0
    matched_dt = []
    matched_ch_ok = 0
    matched_ch_bad = 0
    unmatched_1b_mets = []
    unmatched_1k_mets = []
    ch_mismatch_examples = []

    # Use a stricter two-pointer: for each 1B event, find the closest 1K event
    j_start = 0
    matched_1k = set()

    for ii in range(n1b):
        best_j = -1
        best_dt = tol + 1
        # Search forward from j_start
        for jj in range(j_start, n1k):
            dt = m1b[ii] - m1k[jj]
            if dt > tol:
                j_start = jj + 1
                continue
            if dt < -tol:
                break
            if abs(dt) < best_dt and jj not in matched_1k:
                # Also require channel match for the best match
                best_dt = abs(dt)
                best_j = jj

        if best_j >= 0:
            matched_1k.add(best_j)
            matched_dt.append(m1b[ii] - m1k[best_j])
            if c1b[ii] == c1k[best_j]:
                matched_ch_ok += 1
            else:
                matched_ch_bad += 1
                if len(ch_mismatch_examples) < 20:
                    ch_mismatch_examples.append(
                        (m1b[ii], c1b[ii], c1k[best_j], d1k[best_j]))
        else:
            unmatched_1b_mets.append(m1b[ii])

    for jj in range(n1k):
        if jj not in matched_1k:
            unmatched_1k_mets.append(m1k[jj])

    return {
        "matched_ok": matched_ch_ok,
        "matched_ch_bad": matched_ch_bad,
        "matched_dt": np.array(matched_dt),
        "unmatched_1b": np.array(unmatched_1b_mets),
        "unmatched_1k": np.array(unmatched_1k_mets),
        "ch_mismatch_examples": ch_mismatch_examples,
    }


def compare_box(box_name):
    print(f"\n{'='*60}")
    print(f"Box {box_name}")
    print(f"{'='*60}")

    met_1b, ch_1b, sec_mets = load_1b(box_name)
    met_1k, ch_1k, det_1k = load_1k(box_name)

    t_ref = parse_met_or_utc(TRIGGER)

    # Print uncertain intervals so user can verify against plot
    uncertain_intervals = []
    if len(sec_mets) >= 2:
        for i in range(len(sec_mets) - 1):
            gap = sec_mets[i + 1] - sec_mets[i]
            if gap > WRAP_PERIOD:
                uncertain_intervals.append((sec_mets[i], sec_mets[i + 1]))
    print(f"  SEC anchors: {len(sec_mets)}")
    print(f"  Uncertain (yellow) intervals: {len(uncertain_intervals)}")
    for s, e in uncertain_intervals:
        print(f"    T+{s-t_ref:.1f} ~ T+{e-t_ref:.1f}  ({e-s:.1f}s)")

    mask_1b = compute_confirmed_mask(met_1b, sec_mets)
    mask_1k = compute_confirmed_mask(met_1k, sec_mets)

    met_1b_c = met_1b[mask_1b]
    ch_1b_c = ch_1b[mask_1b]
    met_1k_c = met_1k[mask_1k]
    ch_1k_c = ch_1k[mask_1k]
    det_1k_c = det_1k[mask_1k]

    n_1b = len(met_1b_c)
    n_1k = len(met_1k_c)
    print(f"  Confirmed (white) events: 1B={n_1b:,}, 1K={n_1k:,} (diff={n_1b-n_1k:+,})")

    r = match_events(met_1b_c, ch_1b_c, met_1k_c, ch_1k_c, det_1k_c, tol=6e-6)

    print(f"\n  Matched (time+channel): {r['matched_ok']:>10,}")
    print(f"  Matched (time only):    {r['matched_ch_bad']:>10,}  <- same MET, diff channel")
    print(f"  Only in 1B:             {len(r['unmatched_1b']):>10,}")
    print(f"  Only in 1K:             {len(r['unmatched_1k']):>10,}")

    if len(r["matched_dt"]) > 0:
        dt = r["matched_dt"]
        print(f"  MET diff: mean={dt.mean():.3e}s, std={dt.std():.3e}s, max={np.max(np.abs(dt)):.3e}s")

    # Bin unmatched events into 10s windows to show where they cluster
    all_unmatched = []
    if len(r["unmatched_1b"]) > 0:
        all_unmatched.extend(r["unmatched_1b"])
    if len(r["unmatched_1k"]) > 0:
        all_unmatched.extend(r["unmatched_1k"])

    if len(all_unmatched) > 0:
        all_unmatched = np.array(all_unmatched) - t_ref
        bins = np.arange(int(all_unmatched.min()) - 1, int(all_unmatched.max()) + 11, 10)
        hist, edges = np.histogram(all_unmatched, bins=bins)
        print(f"\n  Mismatch time distribution (1B-only + 1K-only, 10s bins):")
        for k in range(len(hist)):
            if hist[k] > 0:
                print(f"    T+{edges[k]:>6.0f}~{edges[k+1]:>6.0f}: {hist[k]:>6,}")

    if r["ch_mismatch_examples"]:
        print(f"\n  Channel mismatch examples:")
        print(f"    {'T-T0':>10s}  1B_ch  1K_ch  1K_det")
        for met, ch1b, ch1k, det in r["ch_mismatch_examples"][:10]:
            print(f"    {met-t_ref:10.3f}  {ch1b:5d}  {ch1k:5d}  {det:5d}")

    status = "PERFECT" if (len(r["unmatched_1b"]) == 0 and len(r["unmatched_1k"]) == 0
                           and r["matched_ch_bad"] == 0) else "MISMATCH"
    print(f"\n  Box {box_name}: {status}")
    return {
        "box": box_name,
        "matched_ok": r["matched_ok"],
        "matched_ch_bad": r["matched_ch_bad"],
        "unmatched_1b": len(r["unmatched_1b"]),
        "unmatched_1k": len(r["unmatched_1k"]),
    }


def main():
    results = []
    for box_name in ["A", "B", "C"]:
        r = compare_box(box_name)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print(f"Summary (confirmed regions, TOL=6us)")
    print(f"{'='*60}")
    print(f"{'Box':>4s} {'Time+Ch OK':>12s} {'Time only':>10s} {'1B only':>10s} {'1K only':>10s}")
    for r in results:
        print(f"  {r['box']:>2s} {r['matched_ok']:>12,} {r['matched_ch_bad']:>10,} "
              f"{r['unmatched_1b']:>10,} {r['unmatched_1k']:>10,}")


if __name__ == "__main__":
    main()
