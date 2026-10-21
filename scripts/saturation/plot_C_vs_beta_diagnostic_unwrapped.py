#!/usr/bin/env python3
"""C-vs-β diagnostic on 2020 relaxed sample, AFTER Large counter unwrap.

Identical to plot_C_vs_beta_diagnostic.py but applies scripts/unwrap_large.py
per-(box, det) to correct the 10-bit wrap before computing Sci_pred_base.

If wrap was the explanation for the relaxed cache anomalies (upper cloud,
slope +1, mean C +345), this plot should largely return to the H1 strict
phenomenology (C ≈ 150, slope ≈ -0.10, no upper cloud).

Output: plots/C_vs_beta_diagnostic_unwrapped.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_diagnostic_unwrapped.png")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

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

    # Per-(box, det) unwrap of Large
    large_corr = np.zeros(len(df), dtype=np.float64)
    print("Unwrapping Large per-(box, det)...")
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values
            pho_d = df.loc[m, "PHO"].values
            large_d = df.loc[m, "Large"].values
            large_corr[m] = unwrap_large(pho_d, large_d)
    n_wraps = ((large_corr - df["Large"].values) / 1024).round().astype(int)
    print(f"  n_wraps distribution: 0={np.sum(n_wraps==0):,}  1={np.sum(n_wraps==1):,}  "
          f"2={np.sum(n_wraps==2):,}  3+={np.sum(n_wraps>=3):,}")

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    base = (((df["PHO"].astype("float64") - large_corr) * lf - df["Wide"].astype("float64")) / L).values.astype("float32")
    sci = df["Sci_1s"].values.astype("float32")
    pos = (base > 0) & (sci > 0)
    base, sci = base[pos], sci[pos]
    residual = base - sci
    print(f"  positive rows after unwrap: {len(base):,}")

    N = min(300_000, len(base))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base), N, replace=False)
    base_s, sci_s = base[idx], sci[idx]

    LO, HI = 30.0, 10_000.0
    N_BINS = 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5))

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
    ax1.plot(xx, xx, "k--", lw=1.5, label=r"$y = x$")
    ax1.plot(xx, xx + C_H1, "b-", lw=2.0, label=fr"$y = x + {C_H1:.0f}$  (additive C, H1 median)")
    ax1.plot(xx, xx / BETA_H1, "r-", lw=2.0, label=fr"$y = x / {BETA_H1:.2f}$  (multiplicative β, H1 median)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(LO, HI)
    ax1.set_ylim(LO, HI)
    ax1.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed  (cnt/s)", fontsize=11)
    ax1.set_ylabel(r"$\mathrm{Sci}_{\mathrm{pred,base}}$ with $\mathrm{Large}_{\mathrm{unwrapped}}$  (cnt/s)", fontsize=11)
    ax1.set_title("log-log AFTER unwrap_large — upper cloud should be gone", fontsize=11)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

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
    ax2.set_ylabel("residual = Sci_pred_base − Sci_obs  (cnt/s, after unwrap)", fontsize=11)
    ax2.set_title("binned median residual AFTER unwrap — should track blue (additive C ≈ 150)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "C vs β diagnostic AFTER unwrap_large — does the cache return to H1 phenomenology?\n"
        "2020 relaxed sample (5%), PSD anomaly month excluded.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")

    print("\nMedian residual per Sci_obs bin (after unwrap):")
    print(f"  {'Sci ≈ (cnt/s)':<16}{'median residual':>18}{'predict additive':>20}{'predict β':>14}{'N':>10}")
    for c, m, n in zip(centers, meds, counts):
        pred_add = C_H1
        pred_mul = (1.0 / BETA_H1 - 1.0) * c
        print(f"  {c:>10.0f}       {m:>+15.0f}    {pred_add:>+15.0f}    {pred_mul:>+10.0f}     {n:>10,}")


if __name__ == "__main__":
    main()
