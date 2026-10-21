#!/usr/bin/env python3
"""Merge per-year v5 aggregator npz files into one combined npz.

Sums histograms and concatenates per-day s_det/dates. Output schema matches
the single-year npz so plot_v5_final_full.py can read it unchanged.

Usage:
    python3 v5_merge_npz.py --input-glob 'v5_agg_*.npz' --output v5_agg_full.npz
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"no files match {args.input_glob}")
    print(f"Merging {len(files)} files...")

    H_loglog = H_before = H_after = None
    XE = YE_LIN = YE_LOG = None
    k_global = None
    dates = []
    s_det_daily = []
    n_eq_daily = []
    stats = {"n_total": 0, "n_clean": 0, "n_valid": 0, "n_capped": 0,
             "n_blob": 0, "n_main": 0, "n_below_yx": 0}

    for f in files:
        z = np.load(f)
        if H_loglog is None:
            H_loglog = z["H_loglog"].copy()
            H_before = z["H_before"].copy()
            H_after = z["H_after"].copy()
            XE = z["XE"].copy()
            YE_LIN = z["YE_LIN"].copy()
            YE_LOG = z["YE_LOG"].copy()
            k_global = float(z["k_global"])
        else:
            H_loglog += z["H_loglog"]
            H_before += z["H_before"]
            H_after += z["H_after"]
        dates.extend(z["dates"].tolist())
        s_det_daily.extend(z["s_det_daily"].tolist())
        n_eq_daily.extend(z["n_eq_daily"].tolist())
        for k in stats:
            stats[k] += int(z[k])
        print(f"  + {Path(f).name}: {len(z['dates'])} days, {int(z['n_total']):,} rows")

    dates_arr = np.array(dates)
    idx = np.argsort(dates_arr)
    dates_arr = dates_arr[idx]
    s_det_arr = np.array(s_det_daily)[idx]
    n_eq_arr = np.array(n_eq_daily)[idx]

    np.savez_compressed(
        args.output,
        H_loglog=H_loglog, H_before=H_before, H_after=H_after,
        XE=XE, YE_LIN=YE_LIN, YE_LOG=YE_LOG,
        dates=dates_arr,
        s_det_daily=s_det_arr,
        n_eq_daily=n_eq_arr,
        k_global=k_global,
        n_total=stats["n_total"], n_clean=stats["n_clean"],
        n_valid=stats["n_valid"], n_capped=stats["n_capped"],
        n_blob=stats["n_blob"], n_main=stats["n_main"],
        n_below_yx=stats["n_below_yx"],
    )
    print(f"\nMerged → {args.output}")
    print(f"  {len(dates_arr)} days, {stats['n_total']:,} rows")
    print(f"  valid={stats['n_valid']:,}, below_yx={stats['n_below_yx']:,}")
    print(f"  blob/main={stats['n_blob']}/{stats['n_main']}")


if __name__ == "__main__":
    main()
