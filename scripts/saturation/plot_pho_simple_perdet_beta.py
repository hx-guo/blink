#!/usr/bin/env python3
"""Per-(box, det) verification plot with PER-DET multiplicative β on full Sci_pred.

Sci_pred = β(box, det) · [(PHO − Large)·(1−dt)/L − Wide/L]

β is the multiplicative shift in log-log space (just translates the cloud
along the diagonal). Fitted per-det by least squares:
    β = Σ(base · Sci_obs) / Σ(base²)

This is the user's preferred form: simple multiplicative correction to the
whole prediction (NOT α-on-Large). Physical interpretation deferred.

Output: plots/pho_simple_perdet_beta.png  (new file, doesn't overwrite).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DEFAULT_CACHE = Path("n_below_study/clean_2020H1.parquet")
DEFAULT_PLOT = Path("plots/pho_simple_perdet_beta.png")

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]

PSD_ANOMALY_START = "2020-04-30"
PSD_ANOMALY_END = "2020-05-31"
L_CYCLES_TO_SEC = 16e-6

N_SCATTER_PER_DET = 40_000
N_BINS = 120


def exclude_psd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~((df["date"] >= PSD_ANOMALY_START) & (df["date"] <= PSD_ANOMALY_END))
    return df.loc[mask].copy()


def compute_base(df: pd.DataFrame) -> np.ndarray:
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    return (((df["PHO"] - df["Large"]) * lf - df["Wide"]) / L).values


def fit_beta_per_det(df: pd.DataFrame) -> dict:
    """Per (box, det), β = Σ(base·Sci_obs) / Σ(base²) — LSQ minimum of (β·base − Sci_obs)²."""
    betas = {}
    base_all = compute_base(df)
    sci_all = df["Sci_1s"].astype("float64").values
    for box in "ABC":
        for det in range(6):
            mask = ((df["box"] == box) & (df["det"] == det)).values
            base = base_all[mask]
            sci = sci_all[mask]
            betas[(box, det)] = float((base * sci).sum() / (base * base).sum())
    return betas


def _density_color_array(x, y, xb, yb):
    H, xedges, yedges = np.histogram2d(x, y, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xedges, x) - 1, 0, len(xedges) - 2)
    iy = np.clip(np.searchsorted(yedges, y) - 1, 0, len(yedges) - 2)
    density = H[ix, iy].astype(float)
    density[density < 1] = 1
    return density


def make_perdet_plot(df: pd.DataFrame, betas: dict, out_path: Path) -> None:
    df = df.copy().reset_index(drop=True)
    length = df["L_cycles"].astype("float32") * L_CYCLES_TO_SEC
    dt_frac = df["Dt"].astype("float32") / df["L_cycles"].astype("float32")
    live_frac = 1.0 - dt_frac
    base = ((df["PHO"] - df["Large"]) * live_frac - df["Wide"]) / length
    # Vectorized (box, det) → β lookup via merge.
    beta_map_df = pd.DataFrame(
        [(b, d, v) for (b, d), v in betas.items()],
        columns=["box", "det", "_beta_val"],
    )
    df = df.merge(beta_map_df, on=["box", "det"], how="left")
    # base was computed before merge — re-align via row-order (merge preserves left-row-order for how="left").
    beta_per_row = df["_beta_val"].astype("float32")
    df["sci_pred"] = (beta_per_row * base).astype("float32")
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
            b_d = betas[(box, col)]
            ax.text(0.96, 0.05,
                     f"β = {b_d:.3f}\nRMS = {rms:.0f}\nmedian = {resid_med:+.0f}\nN = {len(full_sub):,}",
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
        r"Hypothesis with per-det multiplicative $\beta$   $\mathrm{Sci}_{1\mathrm{s}} = "
        r"\beta_{(\mathrm{box},\mathrm{det})} \cdot \dfrac{(\mathrm{PHO}-\mathrm{Large}) \cdot (1 - \mathrm{DeadTime}/\mathrm{L_{cycles}}) - \mathrm{Wide}}"
        r"{\mathrm{L_{cycles}} \cdot 16\,\mu\mathrm{s}}$" + "\n"
        "β fitted independently per (box, det) by least-squares.   2020-H1 clean, PSD anomaly excluded.",
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
    df = pd.read_parquet(cache, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    n_before = len(df)
    df = exclude_psd_anomaly(df)
    print(f"  after PSD anomaly exclusion: {len(df):,} ({n_before - len(df):,} dropped)")

    print("Fitting per-(box, det) multiplicative β...")
    betas = fit_beta_per_det(df)
    print("  β per detector:")
    for box in "ABC":
        row = "  " + "  ".join(f"{box}-{d}: {betas[(box, d)]:.3f}" for d in range(6))
        print(row)
    b_vals = list(betas.values())
    print(f"  range: [{min(b_vals):.3f}, {max(b_vals):.3f}],  mean: {np.mean(b_vals):.3f},  std: {np.std(b_vals):.3f}")

    print(f"Generating plot at {out}...")
    make_perdet_plot(df, betas, out)
    print("Done.")


if __name__ == "__main__":
    main()
