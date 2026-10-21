#!/usr/bin/env python3
"""Diagnostic: does the additive constant C depend on Sci_rate / HV / time / detector?

Per-row implied C from the hypothesis  PHO·(1−dt) = (Sci+C)·L + Wide + Large·(1−dt):
    C = (PHO−Large)·(1−dt)/L − Wide/L − Sci_1s

If C is a true electronics-fixed term (per-det threshold/baseline/internal trigger),
it should be flat vs Sci_1s, HV, and time — only varying across (box, det).
If C drifts with HV → gain/threshold-dependent (environmental).
If C drifts in time → aging / temperature / orbit-related (environmental).
If C scales with Sci_1s → not really constant, additive model is wrong.

Usage:
    python3 scripts/saturation/plot_const_dependencies.py [--cache PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_CACHE = Path("n_below_study/clean_2020H1.parquet")
DEFAULT_OUT = Path("plots/const_dependencies.png")

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "HV"]

PSD_ANOMALY_START = "2020-04-30"
PSD_ANOMALY_END = "2020-05-31"
L_CYCLES_TO_SEC = 16e-6


def exclude_psd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~((df["date"] >= PSD_ANOMALY_START) & (df["date"] <= PSD_ANOMALY_END))
    return df.loc[mask].copy()


def compute_const(df: pd.DataFrame) -> pd.Series:
    length = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    dt_frac = df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    live_frac = 1.0 - dt_frac
    return ((df["PHO"] - df["Large"]) * live_frac - df["Wide"]) / length - df["Sci_1s"]


def binned_median(x, y, bins):
    """Median y in each x-bin. Returns (centers, median, q25, q75)."""
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

    Y_LO, Y_HI = 0, 400

    # Panel 1: C vs Sci_1s (sanity check — should be flat)
    ax = axes[0, 0]
    bins = np.linspace(50, 1200, 30)
    centers, meds, q25, q75 = binned_median(df["Sci_1s"].values, df["const"].values, bins)
    ax.plot(centers, meds, "o-", color="tab:blue", label="median C")
    ax.fill_between(centers, q25, q75, color="tab:blue", alpha=0.25, label="Q25-Q75")
    ax.axhline(meds.mean(), color="red", ls="--", lw=0.8, label=f"global mean C ≈ {meds.mean():.1f}")
    ax.set_xlabel("Sci_1s (cnt/s)")
    ax.set_ylabel("implied C (cnt/s)")
    ax.set_title("C vs Sci_1s rate  (sanity: should be flat if additive model is right)")
    ax.set_ylim(Y_LO, Y_HI)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: C vs HV
    ax = axes[0, 1]
    bins = np.linspace(-1100, -900, 30)
    centers, meds, q25, q75 = binned_median(df["HV"].values, df["const"].values, bins)
    ax.plot(centers, meds, "o-", color="tab:green", label="median C")
    ax.fill_between(centers, q25, q75, color="tab:green", alpha=0.25, label="Q25-Q75")
    ax.axhline(meds.mean(), color="red", ls="--", lw=0.8, label=f"global mean C ≈ {meds.mean():.1f}")
    ax.set_xlabel("HV (V)")
    ax.set_ylabel("implied C (cnt/s)")
    ax.set_title("C vs HV  (gain/threshold dependence?)")
    ax.set_ylim(Y_LO, Y_HI)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: C vs date (daily median)
    ax = axes[1, 0]
    df_t = df.copy()
    df_t["date_dt"] = pd.to_datetime(df_t["date"], format="%Y-%m-%d")
    daily = df_t.groupby("date_dt")["const"].agg(["median",
                                                     lambda s: s.quantile(0.25),
                                                     lambda s: s.quantile(0.75)]).reset_index()
    daily.columns = ["date_dt", "median", "q25", "q75"]
    ax.plot(daily["date_dt"], daily["median"], "-", color="tab:purple", label="daily median C")
    ax.fill_between(daily["date_dt"], daily["q25"], daily["q75"], color="tab:purple", alpha=0.25, label="Q25-Q75")
    ax.axhline(daily["median"].mean(), color="red", ls="--", lw=0.8,
                label=f"mean ≈ {daily['median'].mean():.1f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("implied C (cnt/s)")
    ax.set_title("C vs date (daily median)  (aging / environmental drift?)")
    ax.set_ylim(Y_LO, Y_HI)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 4: C per (box, det) boxplot
    ax = axes[1, 1]
    groups, labels = [], []
    for box in "ABC":
        for det in range(6):
            sub = df[(df["box"] == box) & (df["det"] == det)]
            groups.append(sub["const"].values)
            labels.append(f"{box}-{det}")
    bp = ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True)
    for box_idx, patch in enumerate(bp["boxes"]):
        c = ["#ffcccc", "#ccffcc", "#ccccff"][box_idx // 6]
        patch.set_facecolor(c)
    global_med = np.median([np.median(g) for g in groups])
    ax.axhline(global_med, color="red", ls="--", lw=0.8,
                label=f"global median ≈ {global_med:.1f}")
    ax.set_xlabel("(box, det)")
    ax.set_ylabel("implied C (cnt/s)")
    ax.set_title("C per (box, det) — boxplot (whiskers Q25-Q75, fliers hidden)")
    ax.set_ylim(Y_LO, Y_HI)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        r"Implied additive constant $C$ from   $\mathrm{PHO}\cdot(1-\mathrm{dt}) - \mathrm{Wide} - \mathrm{Large}\cdot(1-\mathrm{dt}) = (\mathrm{Sci}_{1\mathrm{s}} + C) \cdot L$" + "\n"
        "Check: is C a true electronics-fixed constant or does it depend on rate / HV / time?",
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
    df = pd.read_parquet(cache, columns=USE_COLS, dtype_backend="pyarrow")
    print(f"  rows: {len(df):,}")

    n_before = len(df)
    df = exclude_psd_anomaly(df)
    print(f"  after PSD anomaly exclusion: {len(df):,} ({n_before - len(df):,} dropped)")

    df["const"] = compute_const(df)
    print(f"  per-row C: median {df['const'].median():.1f}, IQR [{df['const'].quantile(0.25):.1f}, {df['const'].quantile(0.75):.1f}]")

    # Filter extreme outliers (FIFO resets etc make wild C values)
    n_before = len(df)
    df = df[(df["const"] > -200) & (df["const"] < 800)].copy()
    print(f"  after -200 < C < 800 filter: {len(df):,} ({n_before - len(df):,} dropped)")

    print(f"Generating plot at {out}...")
    make_plot(df, out)
    print("Done.")


if __name__ == "__main__":
    main()
