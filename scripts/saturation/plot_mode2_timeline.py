#!/usr/bin/env python3
"""Plot Mode 2 (high-Wide / low-Sci anomaly) prevalence over the full HXMT-HE mission."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = Path("n_below_study/mode2_timeline.csv")
DEFAULT_OUT = Path("plots/mode2_timeline.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.plot(df["date"], df["mode2_pct"], lw=0.5, color="tab:red")
    ax.fill_between(df["date"], 0, df["mode2_pct"], color="tab:red", alpha=0.3)
    ax.set_ylabel("Mode 2 %\n(sci_pred > 2×sci_obs)")
    ax.set_title("Mode 2 (high-Wide / low-Sci anomaly) prevalence — full HXMT-HE mission",
                  fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(df["mode2_pct"].max() * 1.1, 5))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["date"], df["median_wide_rate"], lw=0.6, color="tab:blue", label="Wide")
    ax.plot(df["date"], df["median_sci_rate"], lw=0.6, color="tab:green", label="Sci")
    ax.plot(df["date"], df["median_pho_rate"], lw=0.6, color="tab:orange", label="PHO")
    ax.set_ylabel("Median rate (cnt/s)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Zoom on 2020 only
    ax.plot(df["date"], df["mode2_pct"], lw=0.8, color="tab:red")
    ax.fill_between(df["date"], 0, df["mode2_pct"], color="tab:red", alpha=0.3)
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-09-01"))
    ax.set_ylabel("Mode 2 % (zoom 2020)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print key transitions
    print("\n=== Transitions ===")
    # When did Mode 2 first exceed 10%?
    over10 = df[df["mode2_pct"] > 10]
    if len(over10) > 0:
        print(f"First day with Mode 2 > 10%: {over10.iloc[0]['date'].date()} ({over10.iloc[0]['mode2_pct']:.1f}%)")
    over30 = df[df["mode2_pct"] > 30]
    if len(over30) > 0:
        print(f"First day with Mode 2 > 30%: {over30.iloc[0]['date'].date()} ({over30.iloc[0]['mode2_pct']:.1f}%)")
        print(f"Last day with Mode 2 > 30%:  {over30.iloc[-1]['date'].date()} ({over30.iloc[-1]['mode2_pct']:.1f}%)")
    over50 = df[df["mode2_pct"] > 50]
    if len(over50) > 0:
        print(f"Range of Mode 2 > 50%: {over50.iloc[0]['date'].date()} → {over50.iloc[-1]['date'].date()} ({len(over50)} days)")


if __name__ == "__main__":
    main()
