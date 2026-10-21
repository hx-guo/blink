#!/usr/bin/env python3
"""Cross-validation: does the choice of unwrap predictor (additive vs multiplicative)
bias the conclusion of "additive C vs multiplicative β"?

Implements two predictors:
- Additive:        predicted_Large = PHO − (Wide + (Sci + C)·L) / (1−dt),  C = 150 cnt/s
- Multiplicative:  predicted_Large = PHO − (Wide + (Sci/β)·L) / (1−dt),    β = 0.84

For each:
1. Unwrap with that predictor
2. Compute base = (PHO − Large_corrected)·(1−dt)/L − Wide/L
3. Bin residual base − Sci_obs vs Sci_obs
4. Report median residual per bin

Then compare:
- Do the two predictors give different n_wraps?
- Do they give different residual structure?
- Which model (additive or β) does each version's residual support?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/unwrap_additive_vs_multiplicative.png")
L_CYCLES_TO_SEC = 16e-6
PSD_START, PSD_END = "2020-04-30", "2020-05-31"

C_H1 = 150.0
BETA_H1 = 0.84

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]


def unwrap_with_predictor(pho, large, wide, sci, l_cycles, dt, predicted_correction):
    """predicted_correction = the part subtracted from PHO in the conservation equation,
       i.e., (Wide + (Sci + C)·L)/(1−dt) for additive or (Wide + (Sci/β)·L)/(1−dt) for mul.
    """
    predicted = pho - predicted_correction
    n_wraps = np.round((predicted - large) / 1024.0).astype(int)
    n_wraps = np.maximum(n_wraps, 0)
    max_allowed = pho - wide
    large_corr = large + n_wraps * 1024.0
    over = large_corr > max_allowed
    if over.any():
        n_max = np.floor((max_allowed - large) / 1024.0).astype(int)
        n_max = np.maximum(n_max, 0)
        n_wraps = np.where(over, n_max, n_wraps)
        large_corr = large + n_wraps * 1024.0
    return large_corr, n_wraps


def binned_median(sci, residual, lo=30.0, hi=10000.0, n_bins=30, min_count=100):
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins)
    centers, meds, q25, q75, ns = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i + 1])
        if m.sum() < min_count:
            continue
        centers.append(float(np.sqrt(bins[i] * bins[i + 1])))
        meds.append(float(np.median(residual[m])))
        q25.append(float(np.quantile(residual[m], 0.25)))
        q75.append(float(np.quantile(residual[m], 0.75)))
        ns.append(int(m.sum()))
    return np.array(centers), np.array(meds), np.array(q25), np.array(q75), np.array(ns)


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    mask = ~((df["date"] >= PSD_START) & (df["date"] <= PSD_END))
    df = df.loc[mask].copy().reset_index(drop=True)
    print(f"  rows after PSD exclusion: {len(df):,}")

    pho = df["PHO"].astype("float64").values
    large = df["Large"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    sci = df["Sci_1s"].astype("float64").values
    lc = df["L_cycles"].astype("float64").values
    dtv = df["Dt"].astype("float64").values
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dtv / lc

    # Additive predictor
    print("\nUnwrap with ADDITIVE predictor (C = 150)...")
    corr_add = (wide + (sci + C_H1) * L) / lf
    large_add, nw_add = unwrap_with_predictor(pho, large, wide, sci, lc, dtv, corr_add)

    # Multiplicative predictor
    print(f"Unwrap with MULTIPLICATIVE predictor (β = {BETA_H1})...")
    corr_mul = (wide + (sci / BETA_H1) * L) / lf
    large_mul, nw_mul = unwrap_with_predictor(pho, large, wide, sci, lc, dtv, corr_mul)

    # Compare n_wraps
    diff = nw_mul - nw_add
    print(f"\nn_wraps distributions:")
    print(f"  {'k':<5}{'additive':>15}{'multiplicative':>17}")
    for k in sorted(set(nw_add) | set(nw_mul)):
        print(f"  {k:<5}{(nw_add == k).sum():>15,}{(nw_mul == k).sum():>17,}")

    print(f"\nPer-row n_wraps difference (mul - add):")
    for k in sorted(set(diff)):
        n = (diff == k).sum()
        if n > 0:
            print(f"  diff = {k:>+3}:  {n:>12,}  ({n / len(diff) * 100:.4f}%)")

    disagree = diff != 0
    print(f"\nDisagreement: {disagree.sum():,} rows ({disagree.sum() / len(diff) * 100:.4f}%)")

    # Compute base + residual for both
    base_add = ((pho - large_add) * lf - wide) / L
    base_mul = ((pho - large_mul) * lf - wide) / L
    resid_add = base_add - sci
    resid_mul = base_mul - sci

    # Filter positive Sci, valid base
    pos_add = (sci > 0) & (base_add > 0)
    pos_mul = (sci > 0) & (base_mul > 0)

    # Binned medians per Sci range
    print("\n=== Binned median residual vs Sci_obs ===")
    print(f"  Additive predicts FLAT at +{C_H1:.0f}")
    print(f"  Multiplicative predicts RAMP at (1/β-1)·Sci = {1/BETA_H1 - 1:.3f}·Sci")
    cen_a, med_a, q25_a, q75_a, n_a = binned_median(sci[pos_add], resid_add[pos_add])
    cen_m, med_m, q25_m, q75_m, n_m = binned_median(sci[pos_mul], resid_mul[pos_mul])
    print(f"\n  {'Sci ≈':<10}{'med_add':>10}{'med_mul':>10}{'add - mul':>14}{'add pred':>10}{'mul pred':>12}{'N':>10}")
    for c, ma, mm, n in zip(cen_a, med_a, med_m, n_a):
        pa = C_H1
        pm = (1.0 / BETA_H1 - 1.0) * c
        print(f"  {c:>8.0f}  {ma:>+9.0f}  {mm:>+9.0f}  {ma - mm:>+13.0f}  {pa:>+8.0f}  {pm:>+10.0f}  {n:>10,}")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, cen, med, q25, q75, label, c_val in [
        (axes[0], cen_a, med_a, q25_a, q75_a, "ADDITIVE predictor", "additive"),
        (axes[1], cen_m, med_m, q25_m, q75_m, "MULTIPLICATIVE predictor", "mul"),
    ]:
        ax.fill_between(cen, q25, q75, color="gray", alpha=0.3, label="Q25-Q75")
        ax.plot(cen, med, "ko-", markersize=5, label="median residual")
        xx = np.logspace(np.log10(30), np.log10(10000), 200)
        ax.axhline(C_H1, color="blue", lw=2, label=f"additive C = {C_H1:.0f} (predicts flat)")
        ax.plot(xx, (1.0 / BETA_H1 - 1.0) * xx, "r-", lw=2,
                 label=f"(1/β−1)·Sci = 0.19·Sci (multiplicative predicts ramp)")
        ax.axhline(0, color="k", ls=":", lw=0.7)
        ax.set_xscale("log"); ax.set_xlim(30, 10000); ax.set_ylim(-100, 600)
        ax.set_xlabel(r"$\mathrm{Sci}_{1\mathrm{s}}$ observed (cnt/s)")
        ax.set_ylabel("residual = base − Sci_obs (cnt/s)")
        ax.set_title(f"Unwrap with {label}", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Cross-validation: does choice of unwrap predictor bias the residual-structure conclusion?\n"
        "If both panels show same shape → unwrap is model-robust, conclusion stands. If different → biased.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
