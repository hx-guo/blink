#!/usr/bin/env python3
"""Diagnose the early-epoch underestimate visible in panel 4.

Compute and visualize the systematic residual (data - model) as a function of
mlat and time, focused on the early-mid mlat × early-time region.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"
a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = (
    202.60, 1.695, 44.46, 6.33, 0.152, 5.25, 1.00, -79.26)


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def model_2d(mlat, t):
    sm = sigm((mlat - mu_m) / k_m)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def main():
    z = np.load(NPZ)
    C_data = z["C_med"]
    n_data = z["C_n"]
    months = z["months"]
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t0 = np.datetime64("2017-06-22")
    t_years = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25
    mask = n_data > 200

    C_pred = model_2d(mlat_centers, t_years)
    resid = C_data - C_pred

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle("Why does panel 4 underestimate early years at mid-mlat?",
                 fontsize=13, fontweight='bold')

    # ─── Panel A: residual as a function of mlat at each year ───
    ax = axes[0, 0]
    date_picks = ["2017-09", "2018-09", "2019-09", "2020-09", "2021-09",
                  "2022-09", "2023-09", "2024-09", "2025-09"]
    idx = [list(months).index(d) for d in date_picks]
    cmap = plt.cm.viridis
    for k, ti in enumerate(idx):
        color = cmap(k / max(len(idx)-1, 1))
        y = resid[:, ti].copy()
        y[~mask[:, ti]] = np.nan
        ax.plot(mlat_centers, y, '-', lw=1.5, color=color, label=date_picks[k])
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.axvline(mu_m, ls=':', color='gray', alpha=0.4, label=f'μ_m={mu_m:.1f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("data − model (cnt/s)", fontsize=11)
    ax.set_title("A. Residual C(|mlat|) at each year — sign flips by epoch",
                 fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    # ─── Panel B: residual over time at fixed mlat (mid-mlat focus) ───
    ax = axes[0, 1]
    mlat_picks = [30, 35, 40, 45, 50, 55]
    cmap = plt.cm.plasma
    mi = [int(np.argmin(np.abs(mlat_centers - m))) for m in mlat_picks]
    for k, idxm in enumerate(mi):
        color = cmap(k / max(len(mi)-1, 1))
        y = resid[idxm, :].copy()
        y[~mask[idxm, :]] = np.nan
        ax.plot(month_dt, y, '-', lw=1.5, color=color,
                label=f"|mlat|={mlat_picks[k]}°")
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.axvline(t0 + np.timedelta64(int(mu_t*365.25), 'D'),
               ls=':', color='gray', alpha=0.5, label=f'μ_t')
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("data − model (cnt/s)", fontsize=11)
    ax.set_title("B. Residual C(t) at fixed mid-mlat — early +, late −?",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ─── Panel C: early-vs-late residual scatter as function of mlat ───
    ax = axes[1, 0]
    early_mask = (t_years >= 0) & (t_years < 1.5)
    late_mask = (t_years > 6.0)
    Cm = np.where(mask, C_data, np.nan)
    Cp = np.where(mask, C_pred, np.nan)
    with np.errstate(invalid='ignore'):
        d_early = np.nanmean(Cm[:, early_mask] - Cp[:, early_mask], axis=1)
        d_late  = np.nanmean(Cm[:, late_mask]  - Cp[:, late_mask],  axis=1)
    ax.plot(mlat_centers, d_early, 'o-', lw=1.5, ms=4, color='C3',
            label='2017-Jun..2018-Dec  (early avg)')
    ax.plot(mlat_centers, d_late, 'o-', lw=1.5, ms=4, color='C0',
            label='2023-Jul..2026-May  (late avg)')
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨data − model⟩ (cnt/s)", fontsize=11)
    ax.set_title("C. Early vs late residual marginal: clear mid-mlat sign-flip pattern",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel D: numerical summary of bias magnitude ───
    ax = axes[1, 1]
    # Time-binned mean residual at mid-mlat band (30-55°)
    band = (mlat_centers >= 30) & (mlat_centers <= 55)
    with np.errstate(invalid='ignore'):
        bias_t = np.nanmean(np.where(mask[band, :], resid[band, :], np.nan), axis=0)
    ax.plot(month_dt, bias_t, 'o-', ms=4, lw=1.2, color='black')
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.fill_between(month_dt, bias_t, 0,
                    where=(bias_t > 0), alpha=0.3, color='C3', label='underestimate')
    ax.fill_between(month_dt, bias_t, 0,
                    where=(bias_t < 0), alpha=0.3, color='C0', label='overestimate')
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("⟨data − model⟩ over |mlat| 30-55° (cnt/s)", fontsize=11)
    ax.set_title("D. Mid-mlat (30-55°) time-bias: how big and how systematic?",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_early_underestimate.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")

    # Numeric summary
    print("\n=== Early bias summary (2017-09 to 2018-09, |mlat| 30-55°) ===")
    early_idx = (t_years > 0.2) & (t_years < 1.3)
    for m in [30, 35, 40, 45, 50, 55]:
        mi = int(np.argmin(np.abs(mlat_centers - m)))
        vals = resid[mi, early_idx][mask[mi, early_idx]]
        if vals.size > 0:
            print(f"  |mlat|={m}°  mean bias={vals.mean():+6.2f}  "
                  f"(data {C_data[mi, early_idx].mean():.0f} vs model {C_pred[mi, early_idx].mean():.0f} cnt/s)")


if __name__ == "__main__":
    main()
