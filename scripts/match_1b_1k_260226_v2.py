#!/usr/bin/env python3
"""1B vs 1K direction verification, GRB 260226A Box A.

Triple-check the symmetric difference direction:
  - Forward match: for each 1B event, find nearest 1K
  - Reverse match: for each 1K event, find nearest 1B
  - Also confirm by simple set-difference on rounded MET (catches any
    ambiguity from the nearest-neighbour tie-breaking)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

BLINK = Path("/Users/skyair/Developer/ihep/blink")
RECON = BLINK / "data/cache_260226a_reconstruct.csv"
K1 = BLINK / "data/1K/Y202602/20260226-3179/HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS"

T0 = 446726273.0  # leap-second-aware MET of 2026-02-26T10:37:50 UTC (1B/1K MET includes 3 leap s since 2012)
T_LO = T0 - 50.0
T_HI = T0 + 100.0
TOL_S = 5e-6

print(f"Window: T0-50 to T0+100, MET [{T_LO:.0f}, {T_HI:.0f}]")
print(f"Tolerance: {TOL_S*1e6} us\n")

# ----- Load 1B Box A events -----
df = pd.read_csv(RECON, dtype={"box": "string", "type": "string", "met": "float64"})
b1 = df[(df["type"] == "EVT") & (df["box"] == "A") &
        (df["met"] >= T_LO) & (df["met"] < T_HI)]["met"].values
b1 = np.sort(b1)
print(f"1B Box A events: {len(b1):,}")
print(f"  MET range: {b1.min():.6f} -- {b1.max():.6f}")

# ----- Load 1K Box A events -----
fe = fits.open(K1, memmap=True)
d = fe["Events"].data
t = d["Time"].astype(float)
det = d["Det_ID"].astype(int)
mask = (t >= T_LO) & (t < T_HI) & (det >= 0) & (det <= 5)
k1 = np.sort(t[mask])
fe.close()
print(f"1K Box A events: {len(k1):,}")
print(f"  MET range: {k1.min():.6f} -- {k1.max():.6f}")

print(f"\nRaw count delta: 1B - 1K = {len(b1) - len(k1):+d}")
print(f"  (positive = 1B has more events)\n")


def nearest_neighbour_match(src, tgt):
    """For each event in src, return the time-distance to its nearest neighbour in tgt."""
    if len(tgt) == 0:
        return np.full(len(src), np.inf)
    idx = np.searchsorted(tgt, src)
    left = np.where(idx > 0, tgt[np.clip(idx - 1, 0, len(tgt) - 1)], np.inf)
    right = np.where(idx < len(tgt), tgt[np.clip(idx, 0, len(tgt) - 1)], np.inf)
    return np.minimum(np.abs(src - left), np.abs(src - right))


# Forward: each 1B looks for 1K neighbour
dist_b1_to_k1 = nearest_neighbour_match(b1, k1)
b1_only = int((dist_b1_to_k1 > TOL_S).sum())
print(f"FORWARD: for each 1B event, find nearest 1K within {TOL_S*1e6}us:")
print(f"  Matched (1B has 1K neighbour): {len(b1) - b1_only:,} / {len(b1):,}")
print(f"  Unmatched (1B-only, no 1K neighbour): {b1_only:,}")

# Reverse: each 1K looks for 1B neighbour
dist_k1_to_b1 = nearest_neighbour_match(k1, b1)
k1_only = int((dist_k1_to_b1 > TOL_S).sum())
print(f"\nREVERSE: for each 1K event, find nearest 1B within {TOL_S*1e6}us:")
print(f"  Matched (1K has 1B neighbour): {len(k1) - k1_only:,} / {len(k1):,}")
print(f"  Unmatched (1K-only, no 1B neighbour): {k1_only:,}")

print(f"\n==> Symmetric difference:")
print(f"    1B-only (present in 1B, absent in 1K): {b1_only}")
print(f"    1K-only (present in 1K, absent in 1B): {k1_only}")
print(f"    Total: {b1_only + k1_only}")

# Sanity check: dump a few 1B-only events to see what they look like
print(f"\n--- 1B-only events (first 10) ---")
b1_only_mask = dist_b1_to_k1 > TOL_S
b1_only_mets = b1[b1_only_mask][:10]
print(f"  MET values: {[f'{m:.6f}' for m in b1_only_mets]}")
# How close to integer-second boundary are they?
print(f"  Fractional second part: {[f'{m - int(m):.6f}' for m in b1_only_mets]}")
print(f"  Distance to nearest integer second: {[f'{min(m - int(m), int(m) + 1 - m):.6f}' for m in b1_only_mets]}")

# Time-residual for matched events (FORWARD direction, signed)
matched_mask = dist_b1_to_k1 <= TOL_S
# Re-do the match to extract signed residual
idx = np.searchsorted(k1, b1[matched_mask])
left = np.where(idx > 0, k1[np.clip(idx - 1, 0, len(k1) - 1)], np.inf)
right = np.where(idx < len(k1), k1[np.clip(idx, 0, len(k1) - 1)], np.inf)
matched_src = b1[matched_mask]
nearest_neighbor = np.where(np.abs(matched_src - left) <= np.abs(matched_src - right), left, right)
resid = matched_src - nearest_neighbor  # 1B - 1K
print(f"\nTime residual (1B_matched - 1K_matched) for {len(resid):,} matched events:")
print(f"  median: {np.median(resid)*1e6:+.4f} us")
print(f"  mean:   {np.mean(resid)*1e6:+.4f} us")
print(f"  std:    {np.std(resid)*1e6:.4f} us")
print(f"  range:  [{resid.min()*1e6:+.3f}, {resid.max()*1e6:+.3f}] us")
