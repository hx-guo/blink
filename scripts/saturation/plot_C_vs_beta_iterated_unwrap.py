#!/usr/bin/env python3
"""Iteratively refine unwrap_v2 by re-estimating per-det C from corrected data.

Loop:
  1. Apply v2 unwrap with current per-det C
  2. Compute residual = base − Sci_obs per row
  3. Re-estimate per-det C = mean(residual) on rows that:
     - Are HIGH confidence (no edge or cap issues)
     - Are NOT magnetar mode (Wide/PHO < 0.3)
  4. Iterate until per-det C stabilizes

Then plot 3-panel diagnostic:
  - log-log Sci_pred vs Sci_obs
  - linear-Y residual vs Sci_obs (density scatter)
  - log-log |residual| vs Sci_obs (density scatter)

Output: plots/C_vs_beta_iterated_unwrap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT = Path("plots/C_vs_beta_iterated_unwrap.png")
L_CYCLES_TO_SEC = 16e-6

C_INIT = 150.0
BETA_H1 = 0.84

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s"]

MAX_ITER = 10
TOL = 0.5   # cnt/s — max per-det C change for convergence


def estimate_per_det_C(df, large_corr, conf):
    """Per (box, det), C = mean residual on HIGH-confidence non-magnetar rows."""
    L = df["L_cycles"].astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - df["Dt"].astype("float64") / df["L_cycles"].astype("float64")
    pho = df["PHO"].astype("float64")
    wide = df["Wide"].astype("float64")
    sci = df["Sci_1s"].astype("float64")
    base = ((pho - large_corr) * lf - wide) / L
    residual = (base - sci).values
    wide_pho = (wide / np.maximum(pho, 1)).values
    is_high_conf = (conf > CONF_LOW)
    is_main = wide_pho < 0.3
    valid_mask = is_high_conf & is_main & np.isfinite(residual) & (sci.values > 100)

    consts = {}
    for box in "ABC":
        for det in range(6):
            m = ((df["box"] == box) & (df["det"] == det)).values & valid_mask
            if m.sum() < 100:
                consts[(box, det)] = 150.0
            else:
                consts[(box, det)] = float(np.mean(residual[m]))
    return consts


def map_C_to_rows(df, consts):
    C_arr = np.full(len(df), 150.0)
    for (b, d), v in consts.items():
        m = ((df["box"] == b) & (df["det"] == d)).values
        C_arr[m] = v
    return C_arr


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    pho = df["PHO"].values
    large_raw = df["Large"].values
    wide = df["Wide"].values
    sci = df["Sci_1s"].values
    lc = df["L_cycles"].values
    dtv = df["Dt"].values

    # Iteration
    C_per_row = np.full(len(df), C_INIT)
    last_C_per_det = None
    for it in range(MAX_ITER):
        large_corr, conf = unwrap_large_v2(
            pho, large_raw, wide, sci, lc, dtv,
            C=C_per_row, return_confidence=True,
        )
        consts = estimate_per_det_C(df, large_corr, conf)
        c_vals = list(consts.values())
        print(f"\nIter {it+1}: per-det C mean={np.mean(c_vals):.2f}, std={np.std(c_vals):.2f}, "
              f"range=[{min(c_vals):.1f}, {max(c_vals):.1f}]")
        for box in "ABC":
            line = "  " + " ".join(f"{box}{d}:{consts[(box,d)]:+5.0f}" for d in range(6))
            print(line)

        # Convergence check
        if last_C_per_det is not None:
            max_change = max(abs(consts[k] - last_C_per_det[k]) for k in consts)
            print(f"  max |ΔC| = {max_change:.2f} cnt/s (tol={TOL})")
            if max_change < TOL:
                print(f"  CONVERGED in {it+1} iterations")
                break
        last_C_per_det = consts
        C_per_row = map_C_to_rows(df, consts)

    # Final unwrap with converged C
    large_corr, conf = unwrap_large_v2(
        pho, large_raw, wide, sci, lc, dtv,
        C=C_per_row, return_confidence=True,
    )
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    print(f"\nFinal n_wraps distribution:")
    for k in sorted(set(n_wraps)):
        if k >= 0:
            print(f"  k={k:>2}: {(n_wraps==k).sum():>12,} ({(n_wraps==k).sum()/len(n_wraps)*100:>6.3f}%)")
    n_low_conf = (conf == CONF_LOW).sum()
    print(f"  LOW confidence rows: {n_low_conf:,} ({n_low_conf/len(conf)*100:.3f}%)")

    # Compute base/residual
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")
    base = (((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L).astype("float32")
    sci_f = sci.astype("float32")
    pos = (base > 0) & (sci_f > 0)
    base = base[pos]
    sci_f = sci_f[pos]
    residual = base - sci_f
    print(f"  positive rows after unwrap: {len(base):,}")

    # === Plot 3 panels ===
    N = min(300_000, len(base))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(base), N, replace=False)
    base_s, sci_s = base[idx], sci_f[idx]
    resid_s = residual[idx]

    LO, HI = 30.0, 10_000.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7.5))

    # Panel 1: Sci_pred vs Sci_obs log-log
    xb = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb = np.logspace(np.log10(LO), np.log10(HI), 150)
    H, xe, ye = np.histogram2d(sci_s, base_s, bins=[xb, yb])
    ix = np.clip(np.searchsorted(xe, sci_s) - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(ye, base_s) - 1, 0, len(ye) - 2)
    dens = H[ix, iy].astype(float); dens[dens<1]=1
    order = np.argsort(dens)
    ax1.scatter(sci_s[order], base_s[order], c=dens[order], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    xx = np.logspace(np.log10(LO), np.log10(HI), 200)
    avg_C = np.mean(list(last_C_per_det.values())) if last_C_per_det else C_INIT
    # Mixed model from diag_test_additive_on_nwraps0.py: fit on n_wraps=0 ONLY (clean)
    # residual = α + k·Sci, α = +37, k = +0.162
    # => Sci_pred = (1+k)·Sci + α
    k_mix, alpha_mix = 0.162, 37.0
    mixed_line = (1.0 + k_mix) * xx + alpha_mix
    ax1.plot(xx, xx, "k--", lw=1.5, label="y = x")
    ax1.plot(xx, xx + avg_C, "b-", lw=2.0, label=fr"y = x + {avg_C:.0f} (additive C, iterated)")
    ax1.plot(xx, xx / BETA_H1, "r-", lw=2.0, label=fr"y = x / {BETA_H1} (multiplicative β)")
    ax1.plot(xx, mixed_line, "g-", lw=2.0, label=fr"mixed: y = {1+k_mix:.3f}·x {alpha_mix:+.0f} (k={k_mix}, α={alpha_mix})")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(LO, HI); ax1.set_ylim(LO, HI)
    ax1.set_xlabel("Sci_1s observed (cnt/s)")
    ax1.set_ylabel("Sci_pred_base with iterated unwrap (cnt/s)")
    ax1.set_title("log-log Sci_pred vs Sci_obs", fontsize=11)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel 2: linear residual vs Sci_obs density scatter
    Y_LO, Y_HI = -500, 1500
    in_range2 = (sci_s >= LO) & (sci_s <= HI) & (resid_s >= Y_LO) & (resid_s <= Y_HI)
    sci_2 = sci_s[in_range2]; resid_2 = resid_s[in_range2]
    xb2 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb2 = np.linspace(Y_LO, Y_HI, 150)
    H2, xe2, ye2 = np.histogram2d(sci_2, resid_2, bins=[xb2, yb2])
    ix2 = np.clip(np.searchsorted(xe2, sci_2) - 1, 0, len(xe2) - 2)
    iy2 = np.clip(np.searchsorted(ye2, resid_2) - 1, 0, len(ye2) - 2)
    dens2 = H2[ix2, iy2].astype(float); dens2[dens2<1]=1
    order2 = np.argsort(dens2)
    ax2.scatter(sci_2[order2], resid_2[order2], c=dens2[order2], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens2.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    xx2 = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax2.axhline(avg_C, color="blue", lw=2.0, label=fr"additive C = {avg_C:.0f}")
    ax2.plot(xx2, (1.0/BETA_H1 - 1.0)*xx2, "r-", lw=2.0,
              label=r"$(1/\beta-1)\cdot \mathrm{Sci} = 0.19\,\mathrm{Sci}$")
    # Mixed: residual = α + k·Sci
    ax2.plot(xx2, alpha_mix + k_mix * xx2, "g-", lw=2.5,
              label=fr"mixed: residual = {alpha_mix:+.0f} + {k_mix:.3f}·Sci")
    ax2.axhline(0, color="k", ls=":", lw=0.7)
    ax2.set_xscale("log"); ax2.set_xlim(LO, HI); ax2.set_ylim(Y_LO, Y_HI)
    ax2.set_xlabel("Sci_1s observed (cnt/s)")
    ax2.set_ylabel("residual = base − Sci_obs (cnt/s)")
    ax2.set_title("density scatter: residual vs Sci_obs — linear Y", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3: log-log |residual| vs Sci_obs
    LOG_Y_LO, LOG_Y_HI = 5.0, 5000.0
    abs_resid = np.abs(resid_s)
    in_range3 = (sci_s >= LO) & (sci_s <= HI) & (abs_resid >= LOG_Y_LO) & (abs_resid <= LOG_Y_HI)
    sci_3 = sci_s[in_range3]; resid_3 = abs_resid[in_range3]
    xb3 = np.logspace(np.log10(LO), np.log10(HI), 150)
    yb3 = np.logspace(np.log10(LOG_Y_LO), np.log10(LOG_Y_HI), 150)
    H3, xe3, ye3 = np.histogram2d(sci_3, resid_3, bins=[xb3, yb3])
    ix3 = np.clip(np.searchsorted(xe3, sci_3) - 1, 0, len(xe3) - 2)
    iy3 = np.clip(np.searchsorted(ye3, resid_3) - 1, 0, len(ye3) - 2)
    dens3 = H3[ix3, iy3].astype(float); dens3[dens3<1]=1
    order3 = np.argsort(dens3)
    ax3.scatter(sci_3[order3], resid_3[order3], c=dens3[order3], cmap="viridis",
                 norm=LogNorm(vmin=1, vmax=max(dens3.max(), 2)),
                 s=2, alpha=0.5, rasterized=True, edgecolor="none")
    xx3 = np.logspace(np.log10(LO), np.log10(HI), 200)
    ax3.axhline(avg_C, color="blue", lw=2.0, label=fr"additive |C| = {avg_C:.0f}")
    ax3.plot(xx3, (1.0/BETA_H1 - 1.0)*xx3, "r-", lw=2.0, label=r"$0.19\cdot \mathrm{Sci}$")
    # Mixed: |α + k·Sci|
    mixed_abs = np.abs(alpha_mix + k_mix * xx3)
    ax3.plot(xx3, mixed_abs, "g-", lw=2.5, label=fr"mixed: |{alpha_mix:+.0f} + {k_mix:.3f}·Sci|")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlim(LO, HI); ax3.set_ylim(LOG_Y_LO, LOG_Y_HI)
    ax3.set_xlabel("Sci_1s observed (cnt/s)")
    ax3.set_ylabel("|residual| (cnt/s, log)")
    ax3.set_title("density scatter: |residual| vs Sci_obs — log-log", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"C vs β diagnostic with ITERATED unwrap (converged after iterations) — ALL data, no exclusion\n"
        f"Final per-det C: mean={avg_C:.1f}, range=[{min(last_C_per_det.values()):.0f}, {max(last_C_per_det.values()):.0f}]. Large unwrap applied.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
