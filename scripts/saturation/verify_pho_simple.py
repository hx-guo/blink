#!/usr/bin/env python3
"""Verify the simplest PHO hypothesis: PHO_rate == sci_rate_094 + large_rate + wide_rate.

Zero free parameters. For each row compute residual_rate = pho_rate - (sci + large + wide),
aggregate per (box, det), print a summary table, and save a 2-panel diagnostic plot.

Usage:
    python3 scripts/saturation/verify_pho_simple.py [--cache PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CACHE = Path("n_below_study/clean_2020H1.parquet")
DEFAULT_PLOT = Path("plots/pho_simple_verification.png")

# 2020-04-30 → 2020-05-31: on-board PSD threshold was changed, classifying many
# NaI events as wide-pulse (Wide). PHO unchanged but Sci↔Wide flipped. Drop this
# 32-day window to keep the analysis on the normal configuration.
PSD_ANOMALY_START = "2020-04-30"
PSD_ANOMALY_END = "2020-05-31"


def exclude_psd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 2020-05 PSD-threshold-anomaly period (inclusive)."""
    mask = ~((df["date"] >= PSD_ANOMALY_START) & (df["date"] <= PSD_ANOMALY_END))
    return df.loc[mask].copy()


L_CYCLES_TO_SEC = 16e-6  # 16 µs per L_cycles tick (per A1005 / PDAUA.c)


def derive_inline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute downstream conveniences from raw cache columns (cache stores raw only).

    `length` is the per-row engineering cycle wallclock duration (≈0.94s nominally
    but varies ±0.7%), so we use it instead of a hardcoded 0.94 for all
    engineering-counter rate scalings.
    """
    df = df.copy()
    df["length"] = df["L_cycles"].astype("float32") * L_CYCLES_TO_SEC
    df["dt_frac"] = df["Dt"].astype("float32") / df["L_cycles"].astype("float32")
    return df


def compute_residual(df: pd.DataFrame) -> pd.Series:
    """Hypothesis residual in events per 1.0s wallclock, with dead-time correction.

    PHO and Large are front-end trigger counts that catch every event regardless
    of eventizer state (no dead-time loss). Sci and Wide are eventizer outputs
    that miss events during dead-time intervals. Scale PHO and Large by
    (1 - dt_frac) to represent the count the eventizer would have processed.

    All engineering rates use per-row `length` (= L_cycles × 16μs ≈ 0.94s)
    so the cycle-duration variability is properly accounted for.
    """
    live_frac = 1.0 - df["dt_frac"]
    pho_per_1s_live = df["PHO"] / df["length"] * live_frac
    large_per_1s_live = df["Large"] / df["length"] * live_frac
    wide_per_1s = df["Wide"] / df["length"]
    return pho_per_1s_live - df["Sci_1s"] - large_per_1s_live - wide_per_1s


def summarize_per_group(df: pd.DataFrame) -> pd.DataFrame:
    """Per (box, det) stats on residual_rate. Expects df to have 'residual_rate'."""
    grp = df.groupby(["box", "det"], observed=True)["residual_rate"]
    return grp.agg(
        N="size",
        mean="mean",
        std="std",
        median="median",
        q01=lambda s: s.quantile(0.01),
        q99=lambda s: s.quantile(0.99),
    ).reset_index()


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: boxplot of residuals across 18 (box, det) groups
    df = df.copy()
    df["group"] = df["box"].astype(str) + df["det"].astype(str)
    groups = sorted(df["group"].unique())
    data = [df.loc[df["group"] == g, "residual_rate"].values for g in groups]
    ax1.boxplot(data, tick_labels=groups, showfliers=False)
    ax1.axhline(0, color="red", lw=1, ls="--", label="residual=0")
    ax1.set_xlabel("(box, det)")
    ax1.set_ylabel("residual rate (cnt/s)")
    ax1.set_title("(PHO − Large)·(1−dt) − (Sci_1s + Wide) per (box, det), 2020-H1 (1s, dt-corrected, PSD anomaly excluded)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="best")

    # Right panel: hexbin of residual vs Sci_1s, all groups combined
    hb = ax2.hexbin(df["Sci_1s"], df["residual_rate"],
                     gridsize=80, cmap="viridis", bins="log", mincnt=1)
    ax2.axhline(0, color="red", lw=1, ls="--")
    ax2.set_xlabel("Sci_1s (cnt/s)")
    ax2.set_ylabel("residual rate (cnt/s)")
    ax2.set_title("Residual structure vs Sci rate (all rows)")
    plt.colorbar(hb, ax=ax2, label="log N")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--out", default=str(DEFAULT_PLOT))
    args = p.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)

    print(f"Loading {cache}...")
    df = pd.read_parquet(cache)
    print(f"  rows: {len(df):,}")

    n_before = len(df)
    df = exclude_psd_anomaly(df)
    print(f"  after PSD anomaly exclusion ({PSD_ANOMALY_START} to {PSD_ANOMALY_END}): "
          f"{len(df):,} rows ({n_before - len(df):,} dropped)")

    df = derive_inline(df)
    df["residual_rate"] = compute_residual(df)

    df["pho_per_1s"] = df["PHO"] / df["length"]
    print("\nGlobal stats (per 1.0s wallclock):")
    print(f"  mean residual:   {df['residual_rate'].mean():+.2f} cnt/s")
    print(f"  std residual:    {df['residual_rate'].std():.2f} cnt/s")
    print(f"  median residual: {df['residual_rate'].median():+.2f} cnt/s")
    print(f"  PHO median:      {df['pho_per_1s'].median():.2f} cnt/s (for context)")
    print(f"  mean/PHO median: {df['residual_rate'].mean() / df['pho_per_1s'].median() * 100:+.2f}%")

    summary = summarize_per_group(df)
    print("\nPer (box, det) (residual cnt/s):")
    print(summary.to_string(index=False))

    print(f"\nGenerating plot at {out}...")
    make_plot(df, out)
    print("Done.")


if __name__ == "__main__":
    main()
