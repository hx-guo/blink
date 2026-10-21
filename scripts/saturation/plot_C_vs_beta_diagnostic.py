#!/usr/bin/env python3
"""Diagnostic: at the wide rate range now available in 2020 relaxed cache,
does the PHO/Sci offset behave like an additive constant C or multiplicative β?

The two models predict very different residual behavior across rate:
- additive C:        residual = Sci_pred_base − Sci_obs ≈ C  (flat horizontal)
- multiplicative β:  residual ≈ (1/β − 1) · Sci_obs        (linear ramp through origin)

H1 strict (rate 50-1500) couldn't distinguish — both look similar in that narrow range.
Relaxed (50-10000) spans 2+ decades, giving the diagnostic power.

Reference values from H1 strict per-det median fits: C = +150, β = 0.84.

Output: plots/C_vs_beta_diagnostic.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_diagnostic.png")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

# H1-strict per-det medians (reference for "what would we predict at high rate")
C_H1 = 150.0
BETA_H1 = 0.84

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    L = df["L_cycles"].astype("float32") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float32") / df["L_cycles"].astype("float32")
    base = (((df["PHO"] - df["Large"]) * lf - df["Wide"]) / L).values.astype("float32")
    sci = df["Sci_1s"].values.astype("float32")
    pos = (base > 0) & (sci > 0)
    base, sci = base[pos], sci[pos]
    residual = base - sci
    print(f"  positive rows: {len(base):,}")

    # Subsample for scatter plotting
    N = min(300_000, len(base))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base), N, replace=False)
    base_s, sci_s = base[idx], sci[idx]

    LO, HI = 30.0, 10_000.0
    N_BINS = 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5))

    # --- Left panel: log-log Sci_pred_base vs Sci_obs ---
    xb = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb = np.logspace(np.log10(LO), np.log10(HI), 150)
    H, xe, ye = np.histogram2d(sci_s, base_s, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xe, sci_s) - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(ye, base_s) - 1, 0, len(ye) - 2)
    dens = H[ix, iy].astype(float)
    dens[dens < 1] = 1
    order = np.argsort(dens)
    sc = ax1.scatter(
        sci_s[order], base_s[order], c=dens[order],
        cmap="viridis", norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
        s=2, alpha=0.5, rasterized=True, edgecolor="none",
    )
    xx = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax1.plot(xx, xx, "k--", lw=1.5, label=r"$y = x$  (perfect)")
    ax1.plot(xx, xx + C_H1, "b-", lw=2.0, label=fr"$y = x + {C_H1:.0f}$  (additive C, H1 median)")
    ax1.plot(xx, xx / BETA_H1, "r-", lw=2.0, label=fr"$y = x \, / \, {BETA_H1:.2f} \approx 1.19\,x$  (multiplicative β, H1 median)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(LO, HI)
    ax1.set_ylim(LO, HI)
    ax1.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed  (cnt/s)", fontsize=11)
    ax1.set_ylabel(r"$\mathrm{Sci}_{\mathrm{pred,base}} = \dfrac{(\mathrm{PHO}-\mathrm{Large})(1-\mathrm{dt}) - \mathrm{Wide}}{L}$  (cnt/s)", fontsize=11)
    ax1.set_title("log-log Sci_pred_base vs Sci_obs\ncloud on blue → additive C correct  |  cloud on red → multiplicative β correct", fontsize=11)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # --- Right panel: binned median residual vs Sci_obs on log-x ---
    bins = np.logspace(np.log10(LO), np.log10(HI), N_BINS)
    centers, meds, q25, q75, counts = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i + 1])
        if m.sum() < 100:
            continue
        centers.append(float(np.sqrt(bins[i] * bins[i + 1])))
        meds.append(float(np.median(residual[m])))
        q25.append(float(np.quantile(residual[m], 0.25)))
        q75.append(float(np.quantile(residual[m], 0.75)))
        counts.append(int(m.sum()))
    centers = np.array(centers)
    meds = np.array(meds)
    q25 = np.array(q25)
    q75 = np.array(q75)

    ax2.fill_between(centers, q25, q75, color="gray", alpha=0.3, label="Q25-Q75 (per bin)")
    ax2.plot(centers, meds, "ko-", markersize=5, lw=1.5, label="median residual per bin")
    xx2 = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax2.axhline(C_H1, color="blue", ls="-", lw=2.0, label=fr"additive C = {C_H1:.0f}  (predicts flat)")
    ax2.plot(xx2, (1.0 / BETA_H1 - 1.0) * xx2, "r-", lw=2.0,
              label=fr"$(1/\beta - 1) \cdot \mathrm{{Sci}} = 0.19\,\mathrm{{Sci}}$  (β predicts ramp)")
    ax2.axhline(0, color="black", ls=":", lw=0.7)
    ax2.set_xscale("log")
    ax2.set_xlim(LO, HI)
    ax2.set_ylim(-100, 500)
    ax2.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed  (cnt/s)", fontsize=11)
    ax2.set_ylabel("residual = Sci_pred_base − Sci_obs  (cnt/s)", fontsize=11)
    ax2.set_title("binned median residual vs Sci_obs (y-range zoomed to expose low-Sci behavior)\nflat on blue → additive C correct  |  ramps along red → multiplicative β correct", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Additive C vs multiplicative β diagnostic — wide rate range (30 to 10000 cnt/s) from 2020 relaxed sample\n"
        "Reference values: H1 strict per-det medians (C = 150 cnt/s, β = 0.84). PSD anomaly month excluded.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")

    print("\nMedian residual per Sci_obs bin:")
    print(f"  {'Sci ≈ (cnt/s)':<16}{'median residual':>18}{'predict (additive)':>22}{'predict (β)':>16}{'N':>10}")
    for c, m, n in zip(centers, meds, counts):
        pred_add = C_H1
        pred_mul = (1.0 / BETA_H1 - 1.0) * c
        print(f"  {c:>10.0f}       {m:>+15.0f}    {pred_add:>+15.0f}      {pred_mul:>+10.0f}     {n:>10,}")


if __name__ == "__main__":
    main()
