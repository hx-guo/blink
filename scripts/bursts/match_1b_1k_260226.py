#!/usr/bin/env python3
"""Event-by-event 1B vs 1K match for GRB 260226A Box A.

For each 1B reconstructed event, find nearest 1K event within tolerance.
Report symmetric difference, time residual stats.
"""
from __future__ import annotations
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
TOL_S = 5e-6  # 5 μs match tolerance

print(f"Window: T0-50s to T0+100s, MET [{T_LO:.0f}, {T_HI:.0f}]")

# 1B Box A events
df = pd.read_csv(RECON, dtype={"box": "string", "type": "string", "met": "float64"})
b1 = df[(df["type"] == "EVT") & (df["box"] == "A") &
        (df["met"] >= T_LO) & (df["met"] < T_HI)]["met"].values
b1 = np.sort(b1)
print(f"1B Box A events: {len(b1):,}")

# 1K Box A events (Det_ID 0-5)
fe = fits.open(K1, memmap=True)
d = fe["Events"].data
t = d["Time"].astype(float)
det = d["Det_ID"].astype(int)
mask = (t >= T_LO) & (t < T_HI) & (det >= 0) & (det <= 5)
k1 = np.sort(t[mask])
fe.close()
print(f"1K Box A events: {len(k1):,}")

# Match: for each 1B event, find nearest 1K event
i_match = np.searchsorted(k1, b1)
# Check both sides
left = np.where(i_match > 0, k1[np.maximum(i_match - 1, 0)], np.inf)
right = np.where(i_match < len(k1), k1[np.minimum(i_match, len(k1) - 1)], np.inf)
dist_left = np.abs(b1 - left)
dist_right = np.abs(b1 - right)
nearest = np.where(dist_left <= dist_right, left, right)
dist = np.minimum(dist_left, dist_right)

matched_b1 = dist < TOL_S
n_matched = int(matched_b1.sum())
print(f"\n1B with 1K match within {TOL_S*1e6}us: {n_matched:,} / {len(b1):,}")
print(f"1B-only events (no 1K match): {len(b1) - n_matched}")

# Now look at 1K-only: for each 1K event, find nearest 1B
i_match_rev = np.searchsorted(b1, k1)
left2 = np.where(i_match_rev > 0, b1[np.maximum(i_match_rev - 1, 0)], np.inf)
right2 = np.where(i_match_rev < len(b1), b1[np.minimum(i_match_rev, len(b1) - 1)], np.inf)
dist2 = np.minimum(np.abs(k1 - left2), np.abs(k1 - right2))
matched_k1 = dist2 < TOL_S
n_k1_matched = int(matched_k1.sum())
print(f"1K with 1B match within {TOL_S*1e6}us: {n_k1_matched:,} / {len(k1):,}")
print(f"1K-only events: {len(k1) - n_k1_matched}")

# Residual stats for matched events
resid = b1[matched_b1] - nearest[matched_b1]  # 1B - 1K time
print(f"\nTime residual (1B - 1K) for matched events:")
print(f"  median: {np.median(resid)*1e6:+.3f} us")
print(f"  std:    {np.std(resid)*1e6:.3f} us")
print(f"  range:  [{resid.min()*1e6:.3f}, {resid.max()*1e6:.3f}] us")

print(f"\nSymmetric difference:")
print(f"  1B-only: {len(b1) - n_matched}")
print(f"  1K-only: {len(k1) - n_k1_matched}")
print(f"  Total:   {len(b1) - n_matched + len(k1) - n_k1_matched}")
