#!/usr/bin/env python3
"""1B vs 1K event-count residual on GRB 260226A.

Replaces the placeholder "3 / 549,661" in Table 5 with measured values.

For each box (A/B/C), count events in window [T0-50, T0+100]s:
  1B: from cache_260226a_reconstruct.csv (EVT rows only; FILL_GAP excluded)
  1K: from HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS
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
# Match paper §5.1 window: [T0-50, T0+100]s
T_LO = T0 - 50.0
T_HI = T0 + 100.0

BOX_DET = {"A": (0, 5), "B": (6, 11), "C": (12, 17)}


def load_1b():
    df = pd.read_csv(RECON, dtype={"box": "string", "type": "string", "met": "float64"})
    df = df[df["type"] == "EVT"].copy()
    df = df[(df["met"] >= T_LO) & (df["met"] < T_HI)]
    return df.groupby("box").size().rename("n_1b").reset_index()


def load_1k():
    fe = fits.open(K1, memmap=True)
    d = fe["Events"].data
    t = d["Time"].astype(float)
    det = d["Det_ID"].astype(int)
    etype = d["Event_Type"].astype(int)
    flag = d["Flag"].astype(int)
    # Science events only (Event_Type==0), good flag (Flag==0)
    science_mask = (etype == 0) & (flag == 0)
    mask = science_mask & (t >= T_LO) & (t < T_HI)
    t = t[mask]; det = det[mask]
    print(f"  1K Event_Type==0 events in window: {len(t):,}")
    rows = []
    for box, (lo, hi) in BOX_DET.items():
        n = int(((det >= lo) & (det <= hi)).sum())
        rows.append({"box": box, "n_1k": n})
    fe.close()
    return pd.DataFrame(rows)


def main():
    print(f"Window: T0-50s to T0+200s, MET [{T_LO:.0f}, {T_HI:.0f}]")
    print(f"Trigger: T0 = MET {T0}")
    print()

    b1 = load_1b()
    print(f"  1B EVT rows: {b1['n_1b'].sum():,}")

    k1 = load_1k()
    print()

    df = b1.merge(k1, on="box", how="outer").fillna(0)
    df["delta"] = (df["n_1b"] - df["n_1k"]).astype(int)
    df["delta_pct"] = 100.0 * df["delta"] / df["n_1k"].clip(lower=1)
    print(f"{'box':>4s}  {'1B':>10s}  {'1K':>10s}  {'1B-1K':>8s}  {'%':>8s}")
    for _, r in df.iterrows():
        print(f"  {r['box']:>2s}  {r['n_1b']:>10,.0f}  {r['n_1k']:>10,.0f}  "
              f"{r['delta']:>+8d}  {r['delta_pct']:>+7.4f}%")
    print()
    tot_1b = int(df["n_1b"].sum())
    tot_1k = int(df["n_1k"].sum())
    tot_delta = tot_1b - tot_1k
    print(f"Total: 1B={tot_1b:,}, 1K={tot_1k:,}, 1B-1K={tot_delta:+,} "
          f"({100*tot_delta/tot_1k:.4f}%)")


if __name__ == "__main__":
    main()
