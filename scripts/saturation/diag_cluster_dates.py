#!/usr/bin/env python3
"""Date distribution of the three Sci_obs clusters (<150 / 150-350 / >350).

For each cluster, count which year-months contribute, to see whether the first
two clusters concentrate on specific magnetar-observation campaigns.
"""
import glob, os
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import pandas as pd

files = [f for f in sorted(glob.glob("/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_20*.parquet"))
         if "sample" not in f]
SEGS = [("1: <150", lambda s: s < 150),
        ("2: 150-350", lambda s: (s >= 150) & (s < 350)),
        ("3: >350", lambda s: s >= 350)]
month_ct = {k: Counter() for k, _ in SEGS}
year_ct = {k: Counter() for k, _ in SEGS}
ntot = {k: 0 for k, _ in SEGS}

for f in files:
    pf = pq.ParquetFile(f)
    for rg in np.unique(np.linspace(0, pf.num_row_groups - 1, 8).astype(int)):
        df = pf.read_row_group(int(rg), columns=["Sci_1s", "date"]).to_pandas()
        sci = df["Sci_1s"].values.astype(float)
        ym = df["date"].str[:7].values   # 'YYYY-MM'
        yr = df["date"].str[:4].values
        for key, fn in SEGS:
            m = fn(sci)
            month_ct[key].update(ym[m])
            year_ct[key].update(yr[m])
            ntot[key] += int(m.sum())
    print(f"  scanned {os.path.basename(f)}", flush=True)

for key, _ in SEGS:
    tot = ntot[key]
    print(f"\n=== cluster {key}   N={tot:,} ===")
    yd = dict(sorted(year_ct[key].items()))
    print("  by year:", {y: f"{n/tot*100:.0f}%" for y, n in yd.items()})
    print("  top 12 months (share of this cluster):")
    for ym, n in month_ct[key].most_common(12):
        print(f"    {ym}: {n:,}  ({n/tot*100:.1f}%)")
