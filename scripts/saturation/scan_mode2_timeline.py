#!/usr/bin/env python3
"""Scan every daily per_sec_parquet for Mode 2 (high-Wide / low-Sci) fraction.

Mode 2 definition: rows where log10((pho - large - wide) / sci) ∈ (0.30, 0.60),
i.e. sci_pred ≈ 2-4× sci_obs. This is the anomalous cloud surfaced during PHO
hypothesis verification.

Per day we report: total clean rows, Mode 2 count, Mode 2 percentage, plus
median Sci/Wide rates for context.

Usage:
    python3 scripts/saturation/scan_mode2_timeline.py [--input-dir PATH] [--out PATH] [--workers N]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


L_CYCLES_MIN = 50_000


def process_one_day(parquet_path: Path) -> dict | None:
    try:
        df = pd.read_parquet(parquet_path,
                              columns=["L_cycles", "PHO", "Sci_1s", "Wide", "Large"])
    except Exception as exc:
        print(f"[skip] {parquet_path.name}: {type(exc).__name__}: {exc}")
        return None
    df = df[df["L_cycles"] > L_CYCLES_MIN]
    if len(df) == 0:
        return None
    length = df["L_cycles"].astype("float64") * 16e-6
    pho_rate = df["PHO"] / length
    sci_rate = df["Sci_1s"] / length
    wide_rate = df["Wide"] / length
    large_rate = df["Large"] / length
    sci_pred = pho_rate - large_rate - wide_rate

    pos = (sci_pred > 0) & (sci_rate > 0)
    if pos.sum() == 0:
        return None
    log_ratio = np.log10(sci_pred[pos] / sci_rate[pos])

    n_total = int(pos.sum())
    n_mode2 = int(((log_ratio > 0.30) & (log_ratio < 0.60)).sum())
    n_high_wide = int((wide_rate[pos] > 200).sum())

    return {
        "date": parquet_path.stem,
        "n_total": n_total,
        "n_mode2": n_mode2,
        "mode2_pct": n_mode2 / n_total * 100,
        "n_high_wide": n_high_wide,
        "high_wide_pct": n_high_wide / n_total * 100,
        "median_wide_rate": float(wide_rate[pos].median()),
        "median_sci_rate": float(sci_rate[pos].median()),
        "median_pho_rate": float(pho_rate[pos].median()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="per_sec_parquet")
    p.add_argument("--out", default="n_below_study/mode2_timeline.csv")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.glob("*.parquet")
                    if f.stem and f.stem[0].isdigit())
    print(f"Scanning {len(files)} daily parquets with {args.workers} workers...")

    if args.workers == 1:
        results = [process_one_day(f) for f in files]
    else:
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_one_day, files)

    results = [r for r in results if r is not None]
    print(f"  collected {len(results)} day records")

    df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nTop 10 Mode 2 days:")
    print(df.nlargest(10, "mode2_pct")[["date", "n_total", "mode2_pct", "median_wide_rate", "median_sci_rate"]].to_string(index=False))
    print(f"\nTop 10 quietest Mode 2 days:")
    print(df.nsmallest(10, "mode2_pct")[["date", "n_total", "mode2_pct", "median_wide_rate", "median_sci_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
