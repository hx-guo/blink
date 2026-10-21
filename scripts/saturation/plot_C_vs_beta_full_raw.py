#!/usr/bin/env python3
"""C-vs-β diagnostic on 2020 relaxed sample with NO date exclusion AND NO Large unwrap.

Raw view — same as plot_C_vs_beta_full_no_excl.py but uses Large_raw (wrapped)
directly without unwrap correction.

Output: plots/C_vs_beta_full_raw.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_full_raw.png")
L_CYCLES_TO_SEC = 16e-6

C_H1 = 150.0
BETA_H1 = 0.84

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")
    print(f"  NO date exclusion, NO Large unwrap — RAW data view")

    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    # NO unwrap — use Large as-is from cache
    base = (((df["PHO"].astype("float64") - df["Large"].astype("float64")) * lf - df["Wide"].astype("float64")) / L).values.astype("float32")
    sci = df["Sci_1s"].values.astype("float32")
    pos = (base > 0) & (sci > 0)
    base, sci = base[pos], sci[pos]
    residual = base - sci
    print(f"  positive rows: {len(base):,}")

    N = min(300_000, len(base))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base), N, replace=False)
    base_s, sci_s = base[idx], sci[idx]

    LO, HI = 30.0, 10_000.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7.5))

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
    ax1.plot(xx, xx + C_H1, "b-", lw=2.0, label=fr"$y = x + {C_H1:.0f}$ (additive C, H1 median)")
    ax1.plot(xx, xx / BETA_H1, "r-", lw=2.0, label=fr"$y = x / {BETA_H1:.2f}$ (multiplicative β, H1 median)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(LO, HI)
    ax1.set_ylim(LO, HI)
    ax1.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed (cnt/s)", fontsize=11)
    ax1.set_ylabel(r"$\mathrm{Sci}_{\mathrm{pred,base}}$ (Large RAW, no unwrap) (cnt/s)", fontsize=11)
    ax1.set_title("log-log Sci_pred_base vs Sci_obs — ALL data RAW (no unwrap)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Right panel: density scatter of residual vs Sci_obs (NOT binned median)
    # log-x, linear-y, density-colored
    Y_LO, Y_HI = -500, 3000   # wide enough to see the upper cloud's raw residual peaks
    xb2 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb2 = np.linspace(Y_LO, Y_HI, 150)

    # Use a fresh subsample for the residual plot
    in_range = (sci >= LO) & (sci <= HI) & (residual >= Y_LO) & (residual <= Y_HI)
    sci_r = sci[in_range]
    resid_r = residual[in_range]
    N2 = min(300_000, len(sci_r))
    idx2 = rng.choice(len(sci_r), N2, replace=False)
    sci_r_s = sci_r[idx2]
    resid_r_s = resid_r[idx2]

    H2, xe2, ye2 = np.histogram2d(sci_r_s, resid_r_s, bins=[xb2, yb2])
    ix2 = np.clip(np.searchsorted(xe2, sci_r_s) - 1, 0, len(xe2) - 2)
    iy2 = np.clip(np.searchsorted(ye2, resid_r_s) - 1, 0, len(ye2) - 2)
    dens2 = H2[ix2, iy2].astype(float)
    dens2[dens2 < 1] = 1
    order2 = np.argsort(dens2)
    sc2 = ax2.scatter(
        sci_r_s[order2], resid_r_s[order2], c=dens2[order2],
        cmap="viridis", norm=LogNorm(vmin=1, vmax=max(dens2.max(), 2)),
        s=2, alpha=0.5, rasterized=True, edgecolor="none",
    )

    xx2 = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax2.axhline(C_H1, color="blue", ls="-", lw=2.0, label=fr"additive C = {C_H1:.0f} (predicts flat)")
    ax2.plot(xx2, (1.0 / BETA_H1 - 1.0) * xx2, "r-", lw=2.0,
              label=fr"$(1/\beta - 1) \cdot \mathrm{{Sci}} = 0.19\,\mathrm{{Sci}}$ (β predicts ramp)")
    ax2.axhline(0, color="black", ls=":", lw=0.7)
    ax2.set_xscale("log")
    ax2.set_xlim(LO, HI)
    ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed (cnt/s)", fontsize=11)
    ax2.set_ylabel("residual = Sci_pred_base − Sci_obs (cnt/s, RAW)", fontsize=11)
    ax2.set_title("density scatter: residual vs Sci_obs — linear Y", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    # === Panel 3: log-log residual (|residual| with sign-encoded color) ===
    # For log Y axis: take abs and note dominant sign
    LOG_Y_LO, LOG_Y_HI = 5.0, 5000.0
    abs_resid = np.abs(residual)
    in_range3 = (sci >= LO) & (sci <= HI) & (abs_resid >= LOG_Y_LO) & (abs_resid <= LOG_Y_HI)
    sci_3 = sci[in_range3]
    resid_3 = abs_resid[in_range3]
    N3 = min(300_000, len(sci_3))
    idx3 = rng.choice(len(sci_3), N3, replace=False)
    sci_3_s = sci_3[idx3]
    resid_3_s = resid_3[idx3]

    xb3 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb3 = np.logspace(np.log10(LOG_Y_LO), np.log10(LOG_Y_HI), 150)
    H3, xe3, ye3 = np.histogram2d(sci_3_s, resid_3_s, bins=[xb3, yb3])
    ix3 = np.clip(np.searchsorted(xe3, sci_3_s) - 1, 0, len(xe3) - 2)
    iy3 = np.clip(np.searchsorted(ye3, resid_3_s) - 1, 0, len(ye3) - 2)
    dens3 = H3[ix3, iy3].astype(float)
    dens3[dens3 < 1] = 1
    order3 = np.argsort(dens3)
    sc3 = ax3.scatter(
        sci_3_s[order3], resid_3_s[order3], c=dens3[order3],
        cmap="viridis", norm=LogNorm(vmin=1, vmax=max(dens3.max(), 2)),
        s=2, alpha=0.5, rasterized=True, edgecolor="none",
    )

    xx3 = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax3.axhline(C_H1, color="blue", ls="-", lw=2.0, label=fr"additive |C| = {C_H1:.0f}")
    ax3.plot(xx3, (1.0 / BETA_H1 - 1.0) * xx3, "r-", lw=2.0,
              label=fr"$(1/\beta - 1) \cdot \mathrm{{Sci}} = 0.19\,\mathrm{{Sci}}$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(LO, HI)
    ax3.set_ylim(LOG_Y_LO, LOG_Y_HI)
    ax3.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed (cnt/s)", fontsize=11)
    ax3.set_ylabel("|residual| = |base − Sci_obs| (cnt/s, log)", fontsize=11)
    ax3.set_title("density scatter: |residual| vs Sci_obs — log-log", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "C vs β diagnostic — ALL data, NO date exclusion, NO Large unwrap\n"
        "Pure raw cache view. The upper cloud (wrap artifact) will be visible here.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")

    # quick text summary
    print(f"\nresidual rows in [{Y_LO}, {Y_HI}]: {in_range.sum():,} ({in_range.mean()*100:.1f}% of positive rows)")


if __name__ == "__main__":
    main()
