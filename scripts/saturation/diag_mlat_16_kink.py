#!/usr/bin/env python3
"""Diagnose the mlat=16° kink hypothesis.

Look at:
  - marginal C(|mlat|) data vs single-sigmoid model (zoomed 0-25°)
  - residual marginal C(|mlat|) — is there a kink?
  - derivative dC/dmlat — local slope vs mlat
  - early vs late comparison (does the kink appear at all times?)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"

# 8-param fit
a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = (
    202.60, 1.695, 44.46, 6.33, 0.152, 5.25, 1.00, -79.26)


def sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


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

    C_model = model_2d(mlat_centers, t_years)

    # Time-averaged marginals
    Cm = np.where(mask, C_data, np.nan)
    Cmm = np.where(mask, C_model, np.nan)
    with np.errstate(invalid='ignore'):
        C_mlat_data = np.nanmean(Cm, axis=1)
        C_mlat_model = np.nanmean(Cmm, axis=1)
        C_mlat_resid = C_mlat_data - C_mlat_model

    # Local slope (centered difference)
    dC_data = np.gradient(C_mlat_data, mlat_centers)
    dC_model = np.gradient(C_mlat_model, mlat_centers)

    # Split early / late to see if kink is time-stationary
    early_mask = (t_years < 2.0)   # first 2 years
    late_mask = (t_years > 7.0)    # last ~2 years
    with np.errstate(invalid='ignore'):
        C_early_data = np.nanmean(np.where(mask[:, early_mask], C_data[:, early_mask], np.nan), axis=1)
        C_early_model = np.nanmean(np.where(mask[:, early_mask], C_model[:, early_mask], np.nan), axis=1)
        C_late_data = np.nanmean(np.where(mask[:, late_mask], C_data[:, late_mask], np.nan), axis=1)
        C_late_model = np.nanmean(np.where(mask[:, late_mask], C_model[:, late_mask], np.nan), axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Marginal C(|mlat|) diagnostic — search for the 16° kink",
                 fontsize=13, fontweight='bold')

    # ─── Panel 1: full mlat range, data vs model + residual ───
    ax = axes[0, 0]
    ax.plot(mlat_centers, C_mlat_data, 'o-', lw=1.5, ms=4, color='black',
            label='data ⟨C⟩_t', zorder=3)
    ax.plot(mlat_centers, C_mlat_model, '-', lw=2, color='C3',
            label='8-param model')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1, label='|mlat|=16°')
    ax.axvline(mu_m, ls=':', color='gray', alpha=0.5, label=f'μ_m={mu_m:.1f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("1. Full range: marginal data vs model", fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel 2: zoomed 0-25° to look for 16° kink ───
    ax = axes[0, 1]
    mlow = mlat_centers <= 25
    ax.plot(mlat_centers[mlow], C_mlat_data[mlow], 'o-', lw=1.5, ms=5, color='black',
            label='data ⟨C⟩_t')
    ax.plot(mlat_centers[mlow], C_mlat_model[mlow], '-', lw=2, color='C3',
            label='8-param model')
    ax.plot(mlat_centers[mlow], C_early_data[mlow], 's--', lw=1, ms=4, color='C0',
            alpha=0.7, label='early (t<2yr)')
    ax.plot(mlat_centers[mlow], C_late_data[mlow], '^--', lw=1, ms=4, color='C1',
            alpha=0.7, label='late (t>7yr)')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩ (cnt/s)", fontsize=11)
    ax.set_title("2. Zoomed 0-25°: kink at 16°?  (also: early vs late)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel 3: residual marginal ───
    ax = axes[1, 0]
    ax.plot(mlat_centers, C_mlat_resid, 'o-', lw=1.5, ms=4, color='black')
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.axvline(mu_m, ls=':', color='gray', alpha=0.5, label=f'μ_m={mu_m:.1f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("data − model (cnt/s)", fontsize=11)
    ax.set_title("3. Marginal residual data − model — kink as sign change?",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 4: local slope dC/dmlat ───
    ax = axes[1, 1]
    ax.plot(mlat_centers, dC_data, 'o-', lw=1.5, ms=4, color='black',
            label='data: dC/dmlat')
    ax.plot(mlat_centers, dC_model, '-', lw=2, color='C3',
            label='model: dC/dmlat')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.axvline(mu_m, ls=':', color='gray', alpha=0.5, label=f'μ_m={mu_m:.1f}°')
    ax.axhline(0, ls='--', color='gray', alpha=0.3)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("dC/d|mlat| (cnt/s per deg)", fontsize=11)
    ax.set_title("4. Local slope — sharp transition at 16° would show as bump",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_mlat_16_kink.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")

    # Print numeric residual at low mlat
    print("\n=== Marginal residual at low |mlat| ===")
    for i, m in enumerate(mlat_centers):
        if m <= 25:
            print(f"  |mlat|={m:5.1f}°  data={C_mlat_data[i]:7.2f}  "
                  f"model={C_mlat_model[i]:7.2f}  resid={C_mlat_resid[i]:+6.2f}  "
                  f"slope_data={dC_data[i]:+5.2f}  slope_model={dC_model[i]:+5.2f}")


if __name__ == "__main__":
    main()
