#!/usr/bin/env python3
"""Inspect confidence labels assigned by unwrap_large_v2.

Reports:
- Confidence distribution
- How residual structure differs by confidence
- Whether dropping LOW-confidence rows cleans the remaining anomalies
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_HIGH, CONF_MEDIUM, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"
USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    large_corr, conf = unwrap_large_v2(
        df["PHO"].values, df["Large"].values, df["Wide"].values,
        df["Sci_1s"].values, df["L_cycles"].values, df["Dt"].values,
        C=150.0, return_confidence=True,
    )

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    base = ((df["PHO"].astype("float64") - large_corr) * lf - df["Wide"].astype("float64")) / L
    sci = df["Sci_1s"].astype("float64")
    residual = (base - sci).values

    print(f"\n=== Confidence distribution ===")
    n_high = (conf == CONF_HIGH).sum()
    n_med = (conf == CONF_MEDIUM).sum()
    n_low = (conf == CONF_LOW).sum()
    print(f"  HIGH:   {n_high:>10,}  ({n_high/len(conf)*100:.4f}%)")
    print(f"  MEDIUM: {n_med:>10,}  ({n_med/len(conf)*100:.4f}%)")
    print(f"  LOW:    {n_low:>10,}  ({n_low/len(conf)*100:.4f}%)")

    print(f"\n=== Stubborn upper-cloud (base > 1000 AND Sci < 300) ===")
    upper = (base.values > 1000) & (sci.values < 300) & (sci.values > 0)
    print(f"  Total stubborn after v2:                  {upper.sum()}")
    print(f"    of which HIGH confidence:    {((conf == CONF_HIGH) & upper).sum()}")
    print(f"    of which MEDIUM confidence:  {((conf == CONF_MEDIUM) & upper).sum()}")
    print(f"    of which LOW confidence:     {((conf == CONF_LOW) & upper).sum()}")

    print(f"\n=== Residual structure by confidence (main range Sci 300-2000) ===")
    main = (base.values > 0) & (sci.values > 300) & (sci.values < 2000) & (base.values < 3000)
    for c, label in [(CONF_HIGH, "HIGH"), (CONF_MEDIUM, "MEDIUM"), (CONF_LOW, "LOW")]:
        m = main & (conf == c)
        if m.sum() == 0:
            continue
        r = residual[m]
        print(f"  {label:<8} (n={m.sum():>10,}): median={np.median(r):+.0f}, "
              f"Q25={np.quantile(r, 0.25):+.0f}, Q75={np.quantile(r, 0.75):+.0f}")

    print(f"\n=== If we DROP all LOW-confidence rows ===")
    kept = conf != CONF_LOW
    print(f"  Kept: {kept.sum():,} ({kept.sum()/len(conf)*100:.3f}%)")
    main_kept = main & kept
    r_kept = residual[main_kept]
    print(f"  Residual in main range after drop: median={np.median(r_kept):+.0f}, "
          f"Q25={np.quantile(r_kept, 0.25):+.0f}, Q75={np.quantile(r_kept, 0.75):+.0f}")
    upper_kept = upper & kept
    print(f"  Stubborn upper-cloud remaining:    {upper_kept.sum()}")


if __name__ == "__main__":
    main()
