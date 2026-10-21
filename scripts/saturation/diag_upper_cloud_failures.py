#!/usr/bin/env python3
"""Characterize the upper-cloud rows that REMAIN after unwrap_large is applied.

The unwrap_large fixes 97.8% of the 4502-row upper cloud. The remaining 2.2%
(~97 rows) are failure modes we want to understand.

Hypotheses to test:
- A) Multi-wrap missed: real Large is 2048+ but algorithm assigned only 1 wrap
- B) Particle event: Wide is anomalously high, formula breakdown not wrap
- C) r mis-calibration for high-rate / specific detector
- D) Physical impossibility (Large_corr > PHO - Wide post-unwrap)

Reports per-failure-class counts and per-(box, det) distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

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

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")

    # Per-det unwrap with per-det r tracking
    large_corr = np.zeros(len(df), dtype=np.float64)
    r_cals = {}
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            pho_d = df.loc[m, "PHO"].values
            large_d = df.loc[m, "Large"].values
            low = (pho_d > 200) & (pho_d < 2500) & (large_d < 900)
            r_cals[(box, det)] = float(np.median(large_d[low] / pho_d[low])) if low.sum() >= 20 else 0.3
            large_corr[m] = unwrap_large(pho_d, large_d)

    df["Large_raw"] = df["Large"].astype("float64")
    df["Large_corr"] = large_corr
    df["n_wraps"] = ((df["Large_corr"] - df["Large_raw"]) / 1024).round().astype(int)
    base_raw = ((df["PHO"] - df["Large_raw"]) * lf - df["Wide"]) / L
    base_corr = ((df["PHO"] - df["Large_corr"]) * lf - df["Wide"]) / L
    df["base_raw"] = base_raw.values
    df["base_corr"] = base_corr.values
    df["sci_obs"] = df["Sci_1s"].astype("float64")

    # The original upper cloud
    upper_raw = (df["base_raw"] > 1000) & (df["sci_obs"] < 300) & (df["sci_obs"] > 0)
    # Post-unwrap survivors
    stubborn = upper_raw & (df["base_corr"] > 1000)
    sd = df.loc[stubborn].copy()
    print(f"\nOriginal upper cloud:   {upper_raw.sum():,} rows")
    print(f"Stubborn after unwrap:  {stubborn.sum():,} rows ({stubborn.sum() / max(upper_raw.sum(), 1) * 100:.1f}%)")

    if stubborn.sum() == 0:
        print("No stubborn rows — nothing to characterize.")
        return

    # Predicted Large under per-det r (what algorithm thinks Large should be)
    sd["r_cal"] = sd.apply(lambda r: r_cals[(r["box"], r["det"])], axis=1)
    sd["predicted_Large"] = sd["r_cal"] * sd["PHO"]
    sd["pho_minus_wide"] = sd["PHO"] - sd["Wide"]

    # Failure classification
    sd["fail_phys_violation"] = sd["Large_corr"] > sd["pho_minus_wide"]
    sd["fail_undercorrected"] = (~sd["fail_phys_violation"]) & ((sd["predicted_Large"] - sd["Large_corr"]) > 512)
    # particle event: Wide rate (per-second) high
    sd["wide_rate"] = sd["Wide"] / (sd["L_cycles"] * L_CYCLES_TO_SEC)
    sd["fail_particle_event"] = sd["wide_rate"] > 500  # arbitrary high-Wide threshold
    # residual class
    sd["residual_after"] = sd["base_corr"] - sd["sci_obs"]

    print(f"\n=== Per-row characterization of {stubborn.sum()} stubborn rows ===")
    print(f"  has physical violation (Large_corr > PHO - Wide):  {sd['fail_phys_violation'].sum():>4}  ({sd['fail_phys_violation'].mean()*100:.1f}%)")
    print(f"  undercorrected (predicted Large > corrected + 512): {sd['fail_undercorrected'].sum():>4}  ({sd['fail_undercorrected'].mean()*100:.1f}%)")
    print(f"  particle event (Wide rate > 500 cnt/s):             {sd['fail_particle_event'].sum():>4}  ({sd['fail_particle_event'].mean()*100:.1f}%)")
    print(f"  overlap classes:")
    only_phys = sd['fail_phys_violation'] & ~sd['fail_undercorrected'] & ~sd['fail_particle_event']
    only_under = ~sd['fail_phys_violation'] & sd['fail_undercorrected'] & ~sd['fail_particle_event']
    only_part = ~sd['fail_phys_violation'] & ~sd['fail_undercorrected'] & sd['fail_particle_event']
    none = ~sd['fail_phys_violation'] & ~sd['fail_undercorrected'] & ~sd['fail_particle_event']
    print(f"    only physical violation:        {only_phys.sum():>4}")
    print(f"    only undercorrected:            {only_under.sum():>4}")
    print(f"    only particle event:            {only_part.sum():>4}")
    print(f"    no flagged failure mode:        {none.sum():>4}")

    print(f"\n=== Per-(box, det) distribution of stubborn rows ===")
    print(sd.groupby(["box", "det"]).size().unstack(fill_value=0).to_string())

    print(f"\n=== Per-(box, det) r_cal vs # stubborn ===")
    for box in "ABC":
        for det in range(6):
            n = ((sd["box"] == box) & (sd["det"] == det)).sum()
            r = r_cals[(box, det)]
            if n > 0:
                print(f"  {box}-{det}: r_cal={r:.3f}, stubborn={n:>3}")

    print(f"\n=== Stubborn row stats ===")
    for c in ["PHO", "Large_raw", "Large_corr", "n_wraps", "Wide", "Sci_1s", "base_raw", "base_corr", "residual_after", "wide_rate", "predicted_Large"]:
        v = sd[c]
        print(f"  {c:<18}  median={v.median():>10.1f}  Q05={v.quantile(0.05):>10.1f}  Q95={v.quantile(0.95):>10.1f}  max={v.max():>10.1f}")

    print(f"\n=== Date distribution (stubborn) ===")
    print("  top 5 dates:")
    for d, n in sd["date"].value_counts().head(5).items():
        print(f"    {d}: {n} rows")

    print(f"\n=== Example rows (highest base_corr) ===")
    cols_show = ["date", "box", "det", "PHO", "Large_raw", "Large_corr", "n_wraps", "Wide", "Sci_1s", "base_raw", "base_corr", "predicted_Large"]
    print(sd.sort_values("base_corr", ascending=False).head(10)[cols_show].to_string())


if __name__ == "__main__":
    main()
