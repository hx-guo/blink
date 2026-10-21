#!/usr/bin/env python3
"""Per-(box, det) verification plot with α=1.32 correction on Large.

Modified hypothesis (1 free param):
    Sci_1s = [ (PHO − α·Large)·(1−dt) − Wide ] / length

α=1.32 comes from least-squares fit assuming Large multi-counting (mean across
18 detectors). If the multi-counting hypothesis is correct, residual cloud should
collapse onto y=x line.

Output: plots/pho_simple_perdet_alpha.png  (does NOT overwrite the α=1 version).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DEFAULT_CACHE = Path("n_below_study/clean_2020H1.parquet")
DEFAULT_PLOT = Path("plots/pho_simple_perdet_alpha.png")

PSD_ANOMALY_START = "2020-04-30"
PSD_ANOMALY_END = "2020-05-31"
L_CYCLES_TO_SEC = 16e-6

# Global α (mean across 18 dets from prior fit); per-det α range 1.23-1.44
ALPHA = 1.32

N_SCATTER_PER_DET = 40_000
N_BINS = 120


def exclude_psd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~((df["date"] >= PSD_ANOMALY_START) & (df["date"] <= PSD_ANOMALY_END))
    return df.loc[mask].copy()


def _density_color_array(x, y, xb, yb):
    H, xedges, yedges = np.histogram2d(x, y, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xedges, x) - 1, 0, len(xedges) - 2)
    iy = np.clip(np.searchsorted(yedges, y) - 1, 0, len(yedges) - 2)
    density = H[ix, iy].astype(float)
    density[density < 1] = 1
    return density


def make_perdet_plot(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    length = df["L_cycles"].astype("float32") * L_CYCLES_TO_SEC
    dt_frac = df["Dt"].astype("float32") / df["L_cycles"].astype("float32")
    live_frac = 1.0 - dt_frac
    # α-corrected hypothesis: subtract α·Large instead of Large
    df["sci_pred"] = ((df["PHO"] - ALPHA * df["Large"]) * live_frac - df["Wide"]) / length
    df["sci_obs"] = df["Sci_1s"]

    pos = df[(df["sci_obs"] > 0) & (df["sci_pred"] > 0)]
    lo = max(20.0, min(float(pos["sci_obs"].quantile(0.01)),
                         float(pos["sci_pred"].quantile(0.01))) * 0.5)
    hi = max(float(pos["sci_obs"].quantile(0.9999)),
              float(pos["sci_pred"].quantile(0.9999))) * 1.5

    xb = np.logspace(np.log10(lo), np.log10(hi), N_BINS)
    yb = np.logspace(np.log10(lo), np.log10(hi), N_BINS)

    fig, axes = plt.subplots(3, 6, figsize=(24, 13), sharex=True, sharey=True)
    rng = np.random.RandomState(0)
    last_sc = None

    for row, box in enumerate("ABC"):
        for col in range(6):
            ax = axes[row, col]
            sub = df[(df["box"] == box) & (df["det"] == col) &
                      (df["sci_obs"] > 0) & (df["sci_pred"] > 0)]
            x = sub["sci_obs"].values
            y = sub["sci_pred"].values

            if len(sub) > 0:
                density = _density_color_array(x, y, xb, yb)
                if len(sub) > N_SCATTER_PER_DET:
                    idx = rng.choice(len(sub), N_SCATTER_PER_DET, replace=False)
                else:
                    idx = np.arange(len(sub))
                order = np.argsort(density[idx])
                sc = ax.scatter(x[idx][order], y[idx][order],
                                 c=density[idx][order],
                                 cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(density.max(), 2)),
                                 s=1.5, alpha=0.6, rasterized=True, edgecolor="none")
                last_sc = sc

            ax.plot([lo, hi], [lo, hi], "r--", lw=1,
                     label="y = x" if (row == 0 and col == 0) else None)

            full_sub = df[(df["box"] == box) & (df["det"] == col)]
            rms = float(np.sqrt(np.mean((full_sub["sci_pred"] - full_sub["sci_obs"]) ** 2)))
            resid_med = float((full_sub["sci_pred"] - full_sub["sci_obs"]).median())
            ax.text(0.96, 0.05, f"RMS={rms:.0f}\nmed={resid_med:+.0f}\nN={len(full_sub):,}",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor="white", alpha=0.75, edgecolor="none"))

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{box}-{col}", fontsize=11)

            if col == 0:
                ax.set_ylabel(f"Box {box}\nSci predicted [cnt/s/det]", fontsize=10)
            if row == 2:
                ax.set_xlabel("Sci observed [cnt/s/det]", fontsize=10)

            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.93, top=0.86, bottom=0.07,
                          hspace=0.22, wspace=0.10)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.07, 0.012, 0.79])
        fig.colorbar(last_sc, cax=cbar_ax, label="2020-H1 local density (log)")

    fig.suptitle(
        r"Hypothesis with α-correction   $\mathrm{Sci}_{1\mathrm{s}} = "
        r"\dfrac{(\mathrm{PHO}-\alpha\cdot\mathrm{Large}) \cdot (1 - \mathrm{DeadTime}/\mathrm{L_{cycles}}) - \mathrm{Wide}}"
        r"{\mathrm{L_{cycles}} \cdot 16\,\mu\mathrm{s}}$" + "\n"
        f"α = {ALPHA} (global; per-det 1.23-1.44).   2020-H1 clean, PSD anomaly excluded.",
        fontsize=13, fontweight="bold", y=0.97)

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
    print(f"  after PSD anomaly exclusion: {len(df):,} ({n_before - len(df):,} dropped)")

    print(f"Applying α={ALPHA} correction on Large...")
    print(f"Generating plot at {out}...")
    make_perdet_plot(df, out)
    print("Done.")


if __name__ == "__main__":
    main()
