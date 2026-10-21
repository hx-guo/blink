#!/usr/bin/env python3
"""During magnetar observation anomaly, WHY is Sci so low?

Test by looking at Sci sub-categories:
- Sci_pure_1s:  events with no ACD coincidence (clean NaI)
- Sci_ACD1_1s:  events with 1 ACD coincidence (single CsI hit)
- Sci_ACDN_1s:  events with multiple ACD coincidences

Three hypotheses:
A) PSD classifies most events as Wide → ALL three Sci subcategories should drop equally
B) ACD veto more aggressive → Sci_pure drops less, Sci_ACD1/N drops more
C) Eventizer drops events differently → asymmetric

The ratios of sub-categories during anom vs normal tell the story.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")

def stats(name, x):
    return f"  {name:<20}  n={len(x):>10,}  median={np.median(x):>9.1f}  mean={np.mean(x):>9.1f}"


def main():
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df):,} rows")

    pho = df["PHO"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    wide_pho = np.where(pho > 0, wide / pho, np.nan)

    # Categories
    in_2020_05 = df["date"].str.startswith("2020-05").values
    in_2020_10 = ((df["date"] >= "2020-10-10") & (df["date"] <= "2020-10-16")).values
    normal = ~in_2020_05 & ~in_2020_10

    anom_in_05 = in_2020_05 & (wide_pho > 0.3) & np.isfinite(wide_pho)
    anom_in_10 = in_2020_10 & (wide_pho > 0.3) & np.isfinite(wide_pho)
    nonanom_10 = in_2020_10 & ~(wide_pho > 0.3) & np.isfinite(wide_pho)

    # Sub-counts (1s window)
    sci_total = df["Sci_1s"].astype("float64").values
    sci_pure = df["Sci_pure_1s"].astype("float64").values
    sci_ACD1 = df["Sci_ACD1_1s"].astype("float64").values
    sci_ACDN = df["Sci_ACDN_1s"].astype("float64").values

    print("\n=== Sci sub-category counts (per-second) ===\n")
    for label, mask in [("Normal days", normal),
                          ("2020-05 anom", anom_in_05),
                          ("2020-10 anom", anom_in_10),
                          ("2020-10 NON-anom (same days)", nonanom_10)]:
        if mask.sum() == 0:
            continue
        print(f"--- {label} ({mask.sum():,} rows) ---")
        print(stats("Sci_pure", sci_pure[mask]))
        print(stats("Sci_ACD1", sci_ACD1[mask]))
        print(stats("Sci_ACDN", sci_ACDN[mask]))
        print(stats("Sci_total", sci_total[mask]))
        # Ratios
        med_pure = np.median(sci_pure[mask])
        med_acd1 = np.median(sci_ACD1[mask])
        med_acdn = np.median(sci_ACDN[mask])
        med_tot = np.median(sci_total[mask])
        print(f"  Ratio   pure/total = {med_pure/max(med_tot,1):.3f},  "
              f"ACD1/total = {med_acd1/max(med_tot,1):.3f},  "
              f"ACDN/total = {med_acdn/max(med_tot,1):.3f}")
        print()

    # Verify subtotal invariant: pure + ACD1 + ACDN = total
    print("=== Sanity: pure + ACD1 + ACDN should = Sci_1s ===")
    diff = (sci_pure + sci_ACD1 + sci_ACDN) - sci_total
    print(f"  Max |diff| over all rows: {np.max(np.abs(diff)):.6f}")
    print(f"  Should be exactly 0 due to Stage 2 invariant")

    # Now compute conservation residual per category to see if conservation holds in sub-categories
    print("\n=== Detailed comparison: 2020-05 ANOM vs Normal days ===")
    print("(showing rate ratios anom / normal for each sub-category)")
    for cat_label, cat in [("Sci_pure", sci_pure), ("Sci_ACD1", sci_ACD1), ("Sci_ACDN", sci_ACDN), ("Sci_total", sci_total)]:
        n = np.median(cat[normal])
        a05 = np.median(cat[anom_in_05])
        a10 = np.median(cat[anom_in_10])
        print(f"  {cat_label:<10}  Normal={n:>9.1f}, 2020-05 anom={a05:>9.1f} (ratio {a05/max(n,1):.3f}), "
              f"2020-10 anom={a10:>9.1f} (ratio {a10/max(n,1):.3f})")


if __name__ == "__main__":
    main()
