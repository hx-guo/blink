#!/usr/bin/env python3
"""Residual structure plot WITH per-det additive constant C correction.

residual = (PHO − Large)·(1−dt)/L − Wide/L − Sci_1s − C(box, det)

For each (box, det), C is fitted as mean residual_{C=0} (same fit as
plot_pho_simple_perdet_const.py). Then we plot residual vs Sci_1s.

If a constant offset fully explains the original baseline, the residual cloud
should be centered on 0 AND remain flat (slope ≈ 0). Compare with the α-corrected
version which gave intercept ≈ 0 but introduced a fake negative slope: that's
the test for whether the additive model is the right physics.

Output: plots/pho_residual_structure_const.png (new file, doesn't overwrite).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DEFAULT_CACHE = Path("n_below_study/clean_2020H1.parquet")
DEFAULT_OUT = Path("plots/pho_residual_structure_const.png")

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]

PSD_ANOMALY_START = "2020-04-30"
PSD_ANOMALY_END = "2020-05-31"
L_CYCLES_TO_SEC = 16e-6

N_SCATTER_PER_DET = 30_000
N_BINS = 120


def exclude_psd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~((df["date"] >= PSD_ANOMALY_START) & (df["date"] <= PSD_ANOMALY_END))
    return df.loc[mask].copy()


def fit_const_per_det(df: pd.DataFrame) -> dict:
    consts = {}
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    resid_c0 = ((df["PHO"] - df["Large"]) * lf - df["Wide"]) / L - df["Sci_1s"]
    for box in "ABC":
        for det in range(6):
            mask = (df["box"] == box) & (df["det"] == det)
            consts[(box, det)] = float(resid_c0[mask].mean())
    return consts


def _density_color(x, y, xb, yb):
    H, xedges, yedges = np.histogram2d(x, y, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xedges, x) - 1, 0, len(xedges) - 2)
    iy = np.clip(np.searchsorted(yedges, y) - 1, 0, len(yedges) - 2)
    density = H[ix, iy].astype(float)
    density[density < 1] = 1
    return density


def make_plot(df: pd.DataFrame, consts: dict, out_path: Path) -> None:
    df = df.copy().reset_index(drop=True)  # streaming-subsampled caches have duplicate labels
    length = df["L_cycles"].astype("float32") * L_CYCLES_TO_SEC
    dt_frac = df["Dt"].astype("float32") / df["L_cycles"].astype("float32")
    live_frac = 1.0 - dt_frac
    # Vectorized (box, det) → C lookup via merge.
    const_map_df = pd.DataFrame(
        [(b, d, v) for (b, d), v in consts.items()],
        columns=["box", "det", "_const_val"],
    )
    df = df.merge(const_map_df, on=["box", "det"], how="left")
    const_row = df["_const_val"].astype("float32")
    df["sci_obs"] = df["Sci_1s"].astype("float32")
    df["residual"] = (
        (df["PHO"] - df["Large"]) * live_frac / length
        - df["Wide"] / length
        - df["sci_obs"]
        - const_row
    )

    x_lo, x_hi = 50.0, 1500.0
    y_lo, y_hi = -250.0, 250.0

    xb = np.linspace(x_lo, x_hi, N_BINS)
    yb = np.linspace(y_lo, y_hi, N_BINS)

    fig, axes = plt.subplots(3, 6, figsize=(24, 13), sharex=True, sharey=True)
    rng = np.random.RandomState(0)
    last_sc = None
    fit_coefs = []

    for row, box in enumerate("ABC"):
        for col in range(6):
            ax = axes[row, col]
            sub = df[(df["box"] == box) & (df["det"] == col)]
            x = sub["sci_obs"].values
            y = sub["residual"].values

            mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
            x_m, y_m = x[mask], y[mask]

            if len(x_m) > 0:
                density = _density_color(x_m, y_m, xb, yb)
                if len(x_m) > N_SCATTER_PER_DET:
                    idx = rng.choice(len(x_m), N_SCATTER_PER_DET, replace=False)
                else:
                    idx = np.arange(len(x_m))
                order = np.argsort(density[idx])
                sc = ax.scatter(x_m[idx][order], y_m[idx][order],
                                 c=density[idx][order], cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(density.max(), 2)),
                                 s=1.0, alpha=0.5, rasterized=True, edgecolor="none")
                last_sc = sc

            if len(sub) > 100:
                X = np.column_stack([np.ones(len(sub)), sub["sci_obs"].values])
                coef, *_ = np.linalg.lstsq(X, sub["residual"].values, rcond=None)
                b_fit, m_fit = coef
                xx = np.linspace(x_lo, x_hi, 50)
                yy = b_fit + m_fit * xx
                ax.plot(xx, yy, color="red", lw=1.0, label="lstsq fit")
                fit_coefs.append((box, col, b_fit, m_fit, len(sub), consts[(box, col)]))
                ax.text(0.04, 0.96,
                         f"C = {consts[(box, col)]:+.0f} cnt/s\nintercept = {b_fit:+.1f}\nslope k = {m_fit:+.4f}\nN = {len(sub):,}",
                         transform=ax.transAxes, ha="left", va="top",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor="white", alpha=0.75,
                                                edgecolor="none"))

            ax.axhline(0, color="black", lw=0.5, ls=":")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(f"{box}-{col}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Box {box}\nresidual [cnt/s]", fontsize=10)
            if row == 2:
                ax.set_xlabel("Sci_1s observed [cnt/s]", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.subplots_adjust(left=0.06, right=0.93, top=0.86, bottom=0.07,
                          hspace=0.22, wspace=0.08)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.07, 0.012, 0.79])
        fig.colorbar(last_sc, cax=cbar_ax, label="2020-H1 local density (log)")

    fig.suptitle(
        r"$\mathrm{Residual} = "
        r"\dfrac{(\mathrm{PHO}-\mathrm{Large}) \cdot (1 - \mathrm{DeadTime}/\mathrm{L_{cycles}}) - \mathrm{Wide}}"
        r"{\mathrm{L_{cycles}} \cdot 16\,\mu\mathrm{s}} - \mathrm{Sci}_{1\mathrm{s}} - C_{(\mathrm{box},\mathrm{det})}$" + "\n"
        "Residual vs Sci_1s observed   per (box, det)   (per-det additive C fitted, 2020-H1 clean, PSD anomaly excluded)",
        fontsize=13, fontweight="bold", y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nPer (box, det) — C + post-C residual fit:")
    print(f"  {'(box,det)':<10}{'C (cnt/s)':>12}{'intercept (cnt/s)':>20}{'slope k':>12}{'N':>10}")
    for box, det, b, m, n, c in fit_coefs:
        print(f"  {box}-{det}      {c:>+11.2f}{b:>+20.2f}{m:>+12.5f}{n:>10,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)

    print(f"Loading {cache}...")
    df = pd.read_parquet(cache, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    n_before = len(df)
    df = exclude_psd_anomaly(df)
    print(f"  after PSD anomaly exclusion: {len(df):,} ({n_before - len(df):,} dropped)")

    print("Fitting per-(box, det) additive C...")
    consts = fit_const_per_det(df)

    print(f"Generating plot at {out}...")
    make_plot(df, consts, out)
    print("Done.")


if __name__ == "__main__":
    main()
