#!/usr/bin/env python3
"""Standalone, large date-timeline of LOW-mode transitions.
Wider figure, clearer bars, year-band background, readable top-N annotations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_LO, SCI_HI = 400.0, 1000.0
HEPOCH = datetime(2012, 1, 1, 0, 0, 0)


def scan():
    """Return dict: date_str → (n_transitions, n_clean_rows)."""
    files = sorted(CSV_DIR.glob("*.csv"))
    files = [f for f in files if f.stat().st_size > 1000]
    print(f"Scanning {len(files):,} files...")
    date_stats = {}
    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, usecols=["box","det","met_sec","L_cycles","Sci","Large"])
        except Exception:
            continue
        if len(d) == 0: continue
        d = d[d["L_cycles"] > L_THRESH]
        d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
        if len(d) < 100: continue
        d["r"] = d["Large"] / d["Sci"].clip(lower=1)
        d = d.sort_values(["box","det","met_sec"])
        # Total transitions in this file (sum over 6 dets)
        total_trans = 0
        for (_, _), g in d.groupby(["box","det"]):
            r = g["r"].values
            cls = np.where(r > 0.5, 1, np.where(r < 0.4, -1, 0))
            prev = None
            for k in range(len(cls)):
                if cls[k] == 0: continue
                if prev is not None and cls[k] != prev:
                    total_trans += 1
                prev = cls[k]
        # File date
        met0 = int(d["met_sec"].min())
        date = (HEPOCH + timedelta(seconds=met0)).strftime("%Y-%m-%d")
        if date not in date_stats:
            date_stats[date] = [0, 0]
        date_stats[date][0] += total_trans
        date_stats[date][1] += len(d)
        if (i+1) % 1000 == 0:
            print(f"  ...{i+1}/{len(files)}")
    return date_stats


def main():
    date_stats = scan()
    dates = sorted(date_stats.keys())
    n_trans = np.array([date_stats[d][0] for d in dates])
    n_clean = np.array([date_stats[d][1] for d in dates])
    date_objs = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates])

    print(f"\nDates with data: {len(dates):,}, Years: "
          f"{date_objs[0].year}-{date_objs[-1].year}")
    print(f"Total transitions: {n_trans.sum():,}")

    # ----- Figure -----
    fig, axes = plt.subplots(2, 1, figsize=(22, 9), sharex=True,
                              gridspec_kw={"hspace": 0.10})

    # Top: absolute transition count per date (log y)
    ax = axes[0]

    # Year background bands
    year_min, year_max = date_objs[0].year, date_objs[-1].year
    for yr in range(year_min, year_max + 1):
        if yr % 2 == 0:
            ax.axvspan(datetime(yr, 1, 1), datetime(yr+1, 1, 1),
                        color="#eef4ff", alpha=0.6, zorder=0)

    # Bars per date (small width)
    ax.bar(date_objs, n_trans + 1, width=1.5, color="#ff7f0e",
            edgecolor="none", alpha=0.85, log=True, zorder=2)

    # Top-15 dates with annotations
    top_idx = np.argsort(n_trans)[::-1][:15]
    for i, idx in enumerate(top_idx):
        d_obj = date_objs[idx]
        c = n_trans[idx]
        ax.annotate(f"{dates[idx]} ({c:,})",
                     (d_obj, c),
                     xytext=(8, 6 + (i % 3) * 14), textcoords='offset points',
                     fontsize=8, color="darkred",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="#cc4444", alpha=0.9),
                     arrowprops=dict(arrowstyle="-", color="#cc4444", lw=0.7))

    ax.set_ylabel("# transitions per date (log)")
    ax.set_title(f"HXMT HE LOW-mode transitions per date  "
                  f"({n_trans.sum():,} total across {len(dates):,} dates)",
                  fontsize=12)
    ax.grid(axis='y', alpha=0.4, which='both')
    ax.set_ylim(0.7, n_trans.max() * 3)

    # Bottom: relative — transitions per 10000 CLEAN rows (normalize for sampling density)
    ax = axes[1]
    for yr in range(year_min, year_max + 1):
        if yr % 2 == 0:
            ax.axvspan(datetime(yr, 1, 1), datetime(yr+1, 1, 1),
                        color="#eef4ff", alpha=0.6, zorder=0)

    rate = 10000 * n_trans / np.maximum(n_clean, 1)
    ax.bar(date_objs, rate, width=1.5, color="#2ca02c",
            edgecolor="none", alpha=0.85, zorder=2)

    # Top-10 by RATE (different ranking from absolute count)
    top_rate_idx = np.argsort(rate)[::-1][:10]
    for i, idx in enumerate(top_rate_idx):
        d_obj = date_objs[idx]
        r = rate[idx]
        ax.annotate(f"{dates[idx]} ({r:.0f}/10k)",
                     (d_obj, r),
                     xytext=(8, 8 + (i % 3) * 14), textcoords='offset points',
                     fontsize=8, color="darkgreen",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="#226622", alpha=0.9),
                     arrowprops=dict(arrowstyle="-", color="#226622", lw=0.7))

    ax.set_ylabel("transition rate (per 10k CLEAN rows)")
    ax.set_xlabel("date")
    ax.set_title("Transition rate per 10k rows (normalized — removes sampling-density bias)",
                  fontsize=11)
    ax.grid(axis='y', alpha=0.4)

    # Year labels at top
    for yr in range(year_min, year_max + 1):
        axes[0].text(datetime(yr, 7, 1), n_trans.max() * 1.5, str(yr),
                      ha='center', fontsize=10, color="#555555", weight='bold')

    fig.suptitle("LOW-mode transitions: date timeline (top = absolute count, bottom = rate)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "low_date_timeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}")
    desktop = Path.home() / "Desktop" / "low_date_timeline.png"
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    print(f"Saved: {desktop}")

    # Also print top-10 lists for reference
    print("\nTop 10 dates by absolute transition count:")
    for idx in np.argsort(n_trans)[::-1][:10]:
        print(f"  {dates[idx]}: {n_trans[idx]:,} transitions  "
              f"({n_clean[idx]:,} CLEAN rows, rate={rate[idx]:.0f}/10k)")

    print("\nTop 10 dates by NORMALIZED rate:")
    for idx in np.argsort(rate)[::-1][:10]:
        print(f"  {dates[idx]}: rate={rate[idx]:.0f}/10k  "
              f"({n_trans[idx]:,} trans / {n_clean[idx]:,} rows)")


if __name__ == "__main__":
    main()
