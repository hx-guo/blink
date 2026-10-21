#!/usr/bin/env python3
"""Compare sigmoid (smooth) vs Heaviside (true step) at 16°.

The user observes a "jump" at ~16°, distinguishing from a smooth transition.
This script:
  - Shows the data at full 1° resolution in 0-25° (no fit, just data)
  - Fits 3 models and compares:
      A: dual-σ smooth (k_2 free)        — 11p
      B: Heaviside step (k=0.01° fixed)  — fit μ_step, amp
      C: Heaviside at μ=16° (fixed)      — fit amp only

A sigmoid is mathematically the smooth limit of a step. If the data is truly
discontinuous, fit will prefer k_2 → 0. If data is 1° wide, fit picks k_2 ~ 1°.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


def heaviside(mlat, mu, k=0.1):
    # k = 0.1° gives a very sharp transition (10%->90% in 0.4°), narrower
    # than the 1° bin grid → indistinguishable from a true step at this resolution.
    # k=0.01 was numerically unstable (overflow).
    z = np.clip((mlat - mu) / k, -50, 50)
    return sigm(z)


def base_model(p_base, mlat, t):
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = p_base
    sm = sigm((mlat - mu_m) / k_m)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def model_A_dual_sigmoid(params, mlat, t):
    """11p: base + 3 params (α₂, μ_2, k_2 free) — current dual-σ."""
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = params[:8]
    alpha2, mu2, k2 = params[8:11]
    sm = sigm((mlat - mu_m) / k_m)
    s2 = sigm((mlat - mu2) / k2)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm + alpha2 * s2
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def model_B_step_free_pos(params, mlat, t):
    """10p: base + 2 params (delta amplitude, μ_step free). k locked to 0.01°."""
    base = base_model(params[:8], mlat, t)
    delta, mu_step = params[8], params[9]
    step = heaviside(mlat, mu_step, k=0.01)   # true Heaviside
    return base + delta * step[:, None]


def model_C_step_at_16(params, mlat, t):
    """9p: base + 1 param (delta amplitude only). μ=16°, k=0.01° locked."""
    base = base_model(params[:8], mlat, t)
    delta = params[8]
    step = heaviside(mlat, 16.0, k=0.01)
    return base + delta * step[:, None]


def fit_eval(name, fn, p0, lo, hi, mlat, t, C_data, mask):
    C_clean = np.where(mask, C_data, 0.0)
    def resid(p):
        return ((fn(p, mlat, t) - C_clean) * mask).ravel()
    res = least_squares(resid, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    C_pred = fn(res.x, mlat, t)
    r = (C_data - C_pred) * mask
    valid = r[mask]
    std = np.std(valid)
    print(f"\n=== {name} (np={len(p0)}) ===")
    print(f"  cost={res.cost:.0f}, nfev={res.nfev}, std={std:.3f} cnt/s")
    print(f"  params: {[f'{x:.3f}' for x in res.x]}")
    return res.x, C_pred, std


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

    # Base p0/bounds (8p)
    p0_8 = [200.0, 1.7, 44.0, 6.0, 0.15, 5.2, 1.0, -80.0]
    lo_8 = [10.0, 0.1, 30.0, 0.5, 0.0, 0.5, 0.2, -300.0]
    hi_8 = [500.0, 10.0, 60.0, 15.0, 1.0, 8.0, 8.0, +300.0]

    pA, CmA, sA = fit_eval(
        "A: dual-σ smooth (k_2 free)", model_A_dual_sigmoid,
        p0_8 + [-0.07, 14.5, 1.3],
        lo_8 + [-2.0, 5.0, 0.1],
        hi_8 + [+2.0, 30.0, 10.0],
        mlat_centers, t_years, C_data, mask)
    pB, CmB, sB = fit_eval(
        "B: Heaviside step (μ_step free, k=0.01° locked)", model_B_step_free_pos,
        p0_8 + [-7.0, 16.0],
        lo_8 + [-30.0, 5.0],
        hi_8 + [+30.0, 30.0],
        mlat_centers, t_years, C_data, mask)
    pC, CmC, sC = fit_eval(
        "C: Heaviside at μ=16° (locked)", model_C_step_at_16,
        p0_8 + [-7.0],
        lo_8 + [-30.0],
        hi_8 + [+30.0],
        mlat_centers, t_years, C_data, mask)

    # Marginal data
    Cm = np.where(mask, C_data, np.nan)
    with np.errstate(invalid='ignore'):
        C_data_marg = np.nanmean(Cm, axis=1)
        models = {'A: smooth (μ=14.5°, k=1.3°)': CmA,
                  f'B: step (μ={pB[9]:.2f}°)':   CmB,
                  'C: step (μ=16°)':              CmC}
        Cm_models = {k: np.nanmean(np.where(mask, v, np.nan), axis=1)
                     for k, v in models.items()}

    # ─── 4-panel plot ───
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Sigmoid vs Heaviside step — testing if the 16° transition is truly discontinuous",
                 fontsize=13, fontweight='bold')

    # Panel 1: zoomed marginal 5-25° with data error-bar-ish (n)
    ax = axes[0, 0]
    ml = (mlat_centers >= 5) & (mlat_centers <= 25)
    ax.plot(mlat_centers[ml], C_data_marg[ml], 'o-', ms=6, lw=1, color='black',
            label='data ⟨C⟩_t  (1° bin)', zorder=5)
    colors = {'A: smooth (μ=14.5°, k=1.3°)': 'C0',
              f'B: step (μ={pB[9]:.2f}°)': 'C2',
              'C: step (μ=16°)': 'C3'}
    for k, v in Cm_models.items():
        ax.plot(mlat_centers[ml], v[ml], '-', lw=2, color=colors[k], label=k)
    ax.axvline(16, ls=':', color='gray', alpha=0.5, lw=1)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("1. Zoomed 5-25°: smooth vs step fit", fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # Panel 2: residuals
    ax = axes[0, 1]
    for k, v in Cm_models.items():
        ax.plot(mlat_centers[ml], C_data_marg[ml] - v[ml], 'o-', ms=4, lw=1.5,
                color=colors[k], label=k)
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.axvline(16, ls=':', color='gray', alpha=0.5, lw=1)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("data − model (cnt/s)", fontsize=11)
    ax.set_title("2. Marginal residual in 5-25°", fontsize=11)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.3)

    # Panel 3: raw values + numerical bin-by-bin
    ax = axes[1, 0]
    ax.plot(mlat_centers[ml], C_data_marg[ml], 'o-', ms=8, lw=1.5, color='black')
    for i, m in enumerate(mlat_centers):
        if 12 <= m <= 19:
            ax.annotate(f"{C_data_marg[i]:.1f}",
                        (m, C_data_marg[i]),
                        xytext=(0, -15), textcoords='offset points',
                        ha='center', fontsize=10, color='C3')
    # Also draw the bin boundaries
    for e in edges:
        if 5 <= e <= 25:
            ax.axvline(e, color='lightgray', alpha=0.5, lw=0.5)
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16° (user)')
    ax.axvline(pB[9], ls=':', color='C0', alpha=0.7, lw=1.5,
               label=f'fit step μ={pB[9]:.2f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("3. Data only — where does the jump actually sit? (1° bins shown)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(11, 21)
    ax.set_ylim(105, 116)

    # Panel 4: bar chart of std
    ax = axes[1, 1]
    names = ['A: smooth\n11p (k_2=1.3°)',
             f'B: step μ_step={pB[9]:.2f}°\n10p (k=0.01°)',
             'C: step μ=16°\n9p (k=0.01°)']
    stds = [sA, sB, sC]
    bars = ax.bar(names, stds, color=['C0', 'C2', 'C3'])
    ax.axhline(5.49, ls=':', color='black', alpha=0.5,
               label='TPS noise floor (5.49)')
    ax.axhline(6.49, ls=':', color='gray', alpha=0.5,
               label='8p α=β baseline (6.49)')
    for b, s in zip(bars, stds):
        ax.text(b.get_x()+b.get_width()/2, s+0.03, f'{s:.3f}',
                ha='center', fontsize=10)
    ax.set_ylabel("heatmap residual std (cnt/s)", fontsize=11)
    ax.set_title("4. Full 2D heatmap residual std",
                 fontsize=11)
    ax.set_ylim(5.0, 7.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_mlat_step_vs_sigmoid.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    print("\n=== Bin-level data 12-20° ===")
    for i, m in enumerate(mlat_centers):
        if 12 <= m <= 20:
            print(f"  bin center={m:5.1f}°  (range {edges[i]:.0f}-{edges[i+1]:.0f}°): "
                  f"⟨C⟩_t={C_data_marg[i]:7.2f}  ΔC={C_data_marg[i]-C_data_marg[i-1]:+.2f}")


if __name__ == "__main__":
    main()
