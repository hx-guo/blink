#!/usr/bin/env python3
"""1B vs 1K match for GRB 260226A Box A — CORRECTED to exclude SEC anchors
from the 1B count.

blink_cli reconstruct outputs SEC anchors (1Hz GPS-disciplined integer-second
events) as type=EVT with MET at exact integer second. These are NOT physical
events; they're time anchors. The 1K event file does not include them in the
science event HDU. Filter them out before comparison.
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

# ----- Load 1B Box A events, EXCLUDE SEC anchors (MET at exact integer sec) -----
df = pd.read_csv(RECON, dtype={"box": "string", "type": "string", "met": "float64"}, low_memory=False)
mask_b = ((df["type"] == "EVT") & (df["box"] == "A") &
          (df["met"] >= T_LO) & (df["met"] < T_HI))
b1_all = df[mask_b]["met"].values
frac = b1_all - b1_all.astype(np.int64)
is_sec = frac < 1e-6
b1 = np.sort(b1_all[~is_sec])
n_sec = int(is_sec.sum())
print(f"1B Box A EVT rows (raw, incl. SEC anchors): {len(b1_all):,}")
print(f"  SEC-like (integer-second MET, channel=0): {n_sec}")
print(f"  Physical events (1B Sci): {len(b1):,}")

# ----- 1K Box A physical events -----
fe = fits.open(K1, memmap=True)
d = fe["Events"].data
t = d["Time"].astype(float)
det = d["Det_ID"].astype(int)
mask_k = (t >= T_LO) & (t < T_HI) & (det >= 0) & (det <= 5)
k1 = np.sort(t[mask_k])
fe.close()
print(f"\n1K Box A events: {len(k1):,}")

print(f"\nPhysical count delta: 1B - 1K = {len(b1) - len(k1):+d}\n")


def nn_distance(src, tgt):
    if len(tgt) == 0:
        return np.full(len(src), np.inf)
    idx = np.searchsorted(tgt, src)
    left = np.where(idx > 0, tgt[np.clip(idx - 1, 0, len(tgt) - 1)], np.inf)
    right = np.where(idx < len(tgt), tgt[np.clip(idx, 0, len(tgt) - 1)], np.inf)
    return np.minimum(np.abs(src - left), np.abs(src - right))


# Forward
dist_b1 = nn_distance(b1, k1)
b1_only = int((dist_b1 > TOL_S).sum())
print(f"For each 1B event, nearest 1K within {TOL_S*1e6}us:")
print(f"  Matched: {len(b1) - b1_only:,} / {len(b1):,}")
print(f"  1B-only: {b1_only}")

# Reverse
dist_k1 = nn_distance(k1, b1)
k1_only = int((dist_k1 > TOL_S).sum())
print(f"\nFor each 1K event, nearest 1B within {TOL_S*1e6}us:")
print(f"  Matched: {len(k1) - k1_only:,} / {len(k1):,}")
print(f"  1K-only: {k1_only}")

print(f"\n==> Symmetric difference:")
print(f"    1B-only: {b1_only}")
print(f"    1K-only: {k1_only}")
print(f"    Total:   {b1_only + k1_only}")

# Dump 1K-only event positions if any
if k1_only > 0:
    only_mask = dist_k1 > TOL_S
    k1_only_mets = k1[only_mask]
    print(f"\n--- 1K-only events ({len(k1_only_mets)} total, first 10) ---")
    for m in k1_only_mets[:10]:
        i = int(m)
        f = m - i
        # distance to nearest integer-second boundary
        d_int = min(f, 1.0 - f)
        print(f"  MET {m:.6f}  (Δ to nearest integer-s boundary: {d_int*1e6:+.3f} us)")
    # ptime tick from boundary
    frac_residuals = [min(m - int(m), int(m) + 1 - m) for m in k1_only_mets]
    print(f"\n  Stats on |MET fractional - 0 or 1| (in us):")
    print(f"    min:    {min(frac_residuals)*1e6:.3f}")
    print(f"    median: {np.median(frac_residuals)*1e6:.3f}")
    print(f"    max:    {max(frac_residuals)*1e6:.3f}")
    print(f"    (compare to ptime tick 2.0 us)")

# Signed residual
matched_mask = dist_b1 <= TOL_S
idx = np.searchsorted(k1, b1[matched_mask])
left = np.where(idx > 0, k1[np.clip(idx - 1, 0, len(k1) - 1)], np.inf)
right = np.where(idx < len(k1), k1[np.clip(idx, 0, len(k1) - 1)], np.inf)
matched_src = b1[matched_mask]
nb = np.where(np.abs(matched_src - left) <= np.abs(matched_src - right), left, right)
resid = matched_src - nb
print(f"\nTime residual (1B - 1K) on {len(resid):,} matched events:")
print(f"  median: {np.median(resid)*1e6:+.4f} us")
print(f"  mean:   {np.mean(resid)*1e6:+.4f} us")
print(f"  std:    {np.std(resid)*1e6:.4f} us")
print(f"  range:  [{resid.min()*1e6:+.3f}, {resid.max()*1e6:+.3f}] us")
