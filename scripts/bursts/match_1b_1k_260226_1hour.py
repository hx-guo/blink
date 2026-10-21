#!/usr/bin/env python3
"""1B vs 1K event-by-event match for the full 2026-02-26T10 hour (Box A).

This is the "scale reference" used in the paper - the longer baseline
against which the burst-window match (965k events) is compared.

Inputs:
  /tmp/box_a_1b_1h.csv  -- blink_cli extract output (1B raw)
  data/1K/.../HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS

Compares:
  1B physical events (EVT rows, excluding SEC anchors)
  1K Event_Type==0 & Flag==0 events for Box A (det 0..5)

Reports residual statistics at full precision (microsecond and below).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path("/Users/skyair/Developer/ihep/blink")
EXTRACT_CSV = Path("/tmp/box_a_1b_1h.csv")
K1 = ROOT / "data/1K/Y202602/20260226-3179/HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS"

T_LO = 446724003.0
T_HI = 446727600.0
TOL_S = 5e-6

print(f"Window: MET [{T_LO:.0f}, {T_HI:.0f})  span {(T_HI-T_LO)/60:.1f} min")
print(f"Match tolerance: {TOL_S*1e6} us\n")

# ---------- 1B Box A from blink_cli extract ----------
print("Loading 1B extract...")
df = pd.read_csv(EXTRACT_CSV, dtype={"box": "string", "type": "string", "met": "float64"})
mask_b = ((df["type"] == "EVT") & (df["box"] == "A") &
          (df["met"] >= T_LO) & (df["met"] < T_HI))
b1 = np.sort(df.loc[mask_b, "met"].values)
print(f"  1B Box A EVT in window: {len(b1):,}")

# ---------- 1K Box A physical events ----------
print("Loading 1K FITS...")
fe = fits.open(K1, memmap=True)
d = fe["Events"].data
t  = d["Time"].astype(np.float64)
det = d["Det_ID"].astype(int)
etype = d["Event_Type"].astype(int)
flag  = d["Flag"].astype(int)
mask_k = ((t >= T_LO) & (t < T_HI) &
          (det >= 0) & (det <= 5) &
          (etype == 0) & (flag == 0))
k1 = np.sort(t[mask_k])
fe.close()
print(f"  1K Box A (Event_Type==0 & Flag==0) in window: {len(k1):,}\n")

print(f"Physical count delta: 1B - 1K = {len(b1) - len(k1):+d}\n")


def nn_distance(src, tgt):
    if len(tgt) == 0:
        return np.full(len(src), np.inf)
    idx = np.searchsorted(tgt, src)
    left  = np.where(idx > 0,         tgt[np.clip(idx-1, 0, len(tgt)-1)], np.inf)
    right = np.where(idx < len(tgt),  tgt[np.clip(idx,   0, len(tgt)-1)], np.inf)
    return np.minimum(np.abs(src - left), np.abs(src - right))


# ---- forward (1B -> 1K) ----
dist_b1 = nn_distance(b1, k1)
b1_only = int((dist_b1 > TOL_S).sum())
print(f"For each 1B event, nearest 1K within {TOL_S*1e6}us:")
print(f"  Matched : {len(b1)-b1_only:,} / {len(b1):,}")
print(f"  1B-only : {b1_only}")

# ---- reverse (1K -> 1B) ----
dist_k1 = nn_distance(k1, b1)
k1_only = int((dist_k1 > TOL_S).sum())
print(f"\nFor each 1K event, nearest 1B within {TOL_S*1e6}us:")
print(f"  Matched : {len(k1)-k1_only:,} / {len(k1):,}")
print(f"  1K-only : {k1_only}")

# ---- signed residual on matched events ----
matched_mask = dist_b1 <= TOL_S
src = b1[matched_mask]
idx = np.searchsorted(k1, src)
left  = np.where(idx > 0,        k1[np.clip(idx-1, 0, len(k1)-1)], np.inf)
right = np.where(idx < len(k1),  k1[np.clip(idx,   0, len(k1)-1)], np.inf)
nb = np.where(np.abs(src - left) <= np.abs(src - right), left, right)
resid = src - nb  # 1B - 1K, in seconds

n = len(resid)
abs_resid = np.abs(resid)
print(f"\n----- Time residual (1B - 1K) on {n:,} matched events -----")
print(f"  median       : {np.median(resid)*1e6:+.6f} us  ({np.median(resid)*1e9:+.3f} ns)")
print(f"  mean         : {np.mean(resid)*1e6:+.6f} us  ({np.mean(resid)*1e9:+.3f} ns)")
print(f"  std (sigma)  : {np.std(resid)*1e6:.6f} us   ({np.std(resid)*1e9:.3f} ns)")
print(f"  MAD          : {np.median(abs_resid - np.median(abs_resid))*1e6:.6f} us")
print(f"  min          : {resid.min()*1e6:+.6f} us")
print(f"  max          : {resid.max()*1e6:+.6f} us")
print(f"  |resid| max  : {abs_resid.max()*1e6:.6f} us")

# Percentile and bin distribution
print(f"\n  Percentiles of |resid|, us:")
for pct in (50, 90, 99, 99.9, 99.99, 100):
    print(f"    {pct:>6.2f}%: {np.percentile(abs_resid, pct)*1e6:.6f}")

# How many EXACTLY zero?
exact_zero = int((resid == 0.0).sum())
print(f"\n  Exactly 0.0 residual : {exact_zero:,} / {n:,}  ({100*exact_zero/n:.4f}%)")

# Bins
edges = [0, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6]
labels = ["0", "<1ns", "1ns-10ns", "10-100ns", "100ns-0.5us", "0.5-1us", "1-2us", "2-5us"]
hist, _ = np.histogram(abs_resid, bins=edges)
print(f"\n  |resid| distribution:")
for lab, h in zip(labels[1:], hist):
    print(f"    {lab:>14s}: {h:>10,}  ({100*h/n:6.3f}%)")
