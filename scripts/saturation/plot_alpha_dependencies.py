#!/usr/bin/env python3
"""Diagnostic: does Large coefficient α depend on Sci_rate / HV / time / detector?

Per-row implied α from the hypothesis  PHO = Sci + α·Large + Wide  (with dt-correction):
    α = [PHO·(1-dt) - Wide - Sci_1s·length] / [Large·(1-dt)]

If α is a true constant systematic (e.g. PMT-pulse-ringing multi-count), it should
be flat vs all other variables. If α scales with rate → pileup mechanism. If α
varies with HV → gain-dependent. If α drifts in time → aging.

Usage:
    python3 scripts/saturation/plot_alpha_dependencies.py [--cache PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_CACHE = Path("n_below_study/clean_2020H1.parquet")
DEFAULT_OUT = Path("plots/alpha_dependencies.png")

PSD_ANOMALY_START = "2020-04-30"
PSD_ANOMALY_END = "2020-05-31"
LARGE_MIN = 100  # require enough Large counts so per-row α isn't noise-dominated


def exclude_psd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~((df["date"] >= PSD_ANOMALY_START) & (df["date"] <= PSD_ANOMALY_END))
    return df.loc[mask].copy()


def compute_alpha(df: pd.DataFrame) -> pd.Series:
    length = df["L_cycles"].astype("float64") * 16e-6
    dt_frac = df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    live_frac = 1.0 - dt_frac
    num = df["PHO"] * live_frac - df["Wide"] - df["Sci_1s"] * length
    den = df["Large"] * live_frac
    return num / den


def binned_median(x, y, bins):
    """Median y in each x-bin. Returns (bin_centers, median_y, q25, q75)."""
    idx = np.digitize(x, bins)
    centers, meds, q25, q75 = [], [], [], []
    for i in range(1, len(bins)):
        mask = idx == i
        if mask.sum() < 100:
            continue
        centers.append(0.5 * (bins[i - 1] + bins[i]))
        meds.append(np.median(y[mask]))
        q25.append(np.quantile(y[mask], 0.25))
        q75.append(np.quantile(y[mask], 0.75))
    return np.array(centers), np.array(meds), np.array(q25), np.array(q75)


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: α vs Sci_1s
    ax = axes[0, 0]
    bins = np.linspace(50, 1200, 30)
    centers, meds, q25, q75 = binned_median(df["Sci_1s"].values, df["alpha"].values, bins)
    ax.plot(centers, meds, "o-", color="tab:blue", label="median α")
    ax.fill_between(centers, q25, q75, color="tab:blue", alpha=0.25, label="Q25-Q75")
    ax.axhline(meds.mean(), color="red", ls="--", lw=0.8, label=f"global mean α ≈ {meds.mean():.3f}")
    ax.set_xlabel("Sci_1s (cnt/s)")
    ax.set_ylabel("implied α")
    ax.set_title("α vs Sci_1s rate")
    ax.set_ylim(1.0, 1.6)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: α vs HV
    ax = axes[0, 1]
    bins = np.linspace(-1100, -900, 30)
    centers, meds, q25, q75 = binned_median(df["HV"].values, df["alpha"].values, bins)
    ax.plot(centers, meds, "o-", color="tab:green", label="median α")
    ax.fill_between(centers, q25, q75, color="tab:green", alpha=0.25, label="Q25-Q75")
    ax.axhline(meds.mean(), color="red", ls="--", lw=0.8, label=f"global mean α ≈ {meds.mean():.3f}")
    ax.set_xlabel("HV (V)")
    ax.set_ylabel("implied α")
    ax.set_title("α vs HV")
    ax.set_ylim(1.0, 1.6)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: α vs time (date, monthly bin)
    ax = axes[1, 0]
    df_t = df.copy()
    df_t["date_dt"] = pd.to_datetime(df_t["date"], format="%Y-%m-%d")
    daily = df_t.groupby("date_dt")["alpha"].agg(["median",
                                                     lambda s: s.quantile(0.25),
                                                     lambda s: s.quantile(0.75)]).reset_index()
    daily.columns = ["date_dt", "median", "q25", "q75"]
    ax.plot(daily["date_dt"], daily["median"], "-", color="tab:purple", label="daily median α")
    ax.fill_between(daily["date_dt"], daily["q25"], daily["q75"], color="tab:purple", alpha=0.25, label="Q25-Q75")
    ax.axhline(daily["median"].mean(), color="red", ls="--", lw=0.8,
                label=f"mean ≈ {daily['median'].mean():.3f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("implied α")
    ax.set_title("α vs date (daily median)")
    ax.set_ylim(1.0, 1.6)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 4: α per (box, det) boxplot
    ax = axes[1, 1]
    groups, labels = [], []
    for box in "ABC":
        for det in range(6):
            sub = df[(df["box"] == box) & (df["det"] == det)]
            groups.append(sub["alpha"].values)
            labels.append(f"{box}-{det}")
    bp = ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True)
    for box_idx, patch in enumerate(bp["boxes"]):
        c = ["#ffcccc", "#ccffcc", "#ccccff"][box_idx // 6]
        patch.set_facecolor(c)
    ax.axhline(np.median([np.median(g) for g in groups]), color="red", ls="--", lw=0.8,
                label=f"global median ≈ {np.median([np.median(g) for g in groups]):.3f}")
    ax.set_xlabel("(box, det)")
    ax.set_ylabel("implied α")
    ax.set_title("α per (box, det) — boxplot (whiskers Q25-Q75, fliers hidden)")
    ax.set_ylim(1.0, 1.6)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        r"Implied $\alpha$ from   $\mathrm{PHO} \cdot (1-\mathrm{dt}) = (\mathrm{Sci}_{1\mathrm{s}} \cdot \mathrm{length} + \mathrm{Wide}) + \alpha \cdot \mathrm{Large} \cdot (1-\mathrm{dt})$" + "\n"
        "Check: is α a true constant or does it depend on rate / HV / time / detector?",
        fontsize=13, fontweight="bold", y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)

    print(f"Loading {cache}...")
    df = pd.read_parquet(cache)
    print(f"  rows: {len(df):,}")

    n_before = len(df)
    df = exclude_psd_anomaly(df)
    print(f"  after PSD anomaly exclusion: {len(df):,} ({n_before - len(df):,} dropped)")

    df["alpha"] = compute_alpha(df)
    print(f"  per-row α: median {df['alpha'].median():.3f}, IQR [{df['alpha'].quantile(0.25):.3f}, {df['alpha'].quantile(0.75):.3f}]")

    # Filter rows with low Large (noisy α) and extreme outliers
    n_before = len(df)
    df = df[(df["Large"] >= LARGE_MIN) & (df["alpha"] > 0.5) & (df["alpha"] < 3.0)].copy()
    print(f"  after Large >= {LARGE_MIN} and 0.5 < α < 3.0 filter: {len(df):,} ({n_before - len(df):,} dropped)")

    print(f"Generating plot at {out}...")
    make_plot(df, out)
    print("Done.")


if __name__ == "__main__":
    main()
