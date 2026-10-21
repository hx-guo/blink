#!/usr/bin/env python3
"""Extend the 8-param α=β model with a low-mlat step at ~16°.

Three variants:
  (A) 8 + 1 = 9 params: amp only, μ_step=16° and k_step=0.5° fixed
  (B) 8 + 3 = 11 params: free step (amp, μ_step, k_step)
  (C) 8 + 2 = 10 params: dual mlat sigmoid (second σ_m for low-mlat hump+kink)
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


def model_base(params, mlat, t):
    """Original 8-param α=β model."""
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = params
    sm = sigm((mlat - mu_m) / k_m)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def model_A(params, mlat, t):
    """9-param: base + step at fixed 16°, amplitude delta only."""
    base = model_base(params[:8], mlat, t)
    delta = params[8]
    step = sigm((mlat - 16.0) / 0.5)            # sharp step, ramp 10-90% in 2°
    return base + delta * step[:, None]


def model_B(params, mlat, t):
    """11-param: base + free step (delta, mu_step, k_step)."""
    base = model_base(params[:8], mlat, t)
    delta, mu_step, k_step = params[8], params[9], params[10]
    step = sigm((mlat - mu_step) / k_step)
    return base + delta * step[:, None]


def model_C(params, mlat, t):
    """10-param: replace single σ_m by sum of two σ_m (low + high)."""
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t = params[:10]
    C_0 = params[10] if len(params) > 10 else 0.0
    s1 = sigm((mlat - mu1) / k1)
    s2 = sigm((mlat - mu2) / k2)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha1 * s1 + alpha2 * s2
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def make_residual(model_fn, mlat, t_years, C_data_clean, mask):
    def residual(params):
        C_pred = model_fn(params, mlat, t_years)
        return ((C_pred - C_data_clean) * mask).ravel()
    return residual


def fit_and_eval(name, model_fn, p0, lo, hi, mlat, t_years, C_data, mask, n_data):
    C_data_clean = np.where(mask, C_data, 0.0)
    resid_fn = make_residual(model_fn, mlat, t_years, C_data_clean, mask)
    res = least_squares(resid_fn, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    C_pred = model_fn(res.x, mlat, t_years)
    resid = (C_data - C_pred) * mask
    valid = resid[mask]
    std = np.std(valid)
    rms_pct = np.sqrt(np.mean(valid**2))/np.mean(C_data[mask])*100
    max_abs = np.max(np.abs(valid))
    print(f"\n=== {name} (n_params={len(p0)}) ===")
    print(f"  fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    print(f"  std: {std:.3f} cnt/s,  RMS/mean: {rms_pct:.2f}%,  max|r|: {max_abs:.1f}")
    print(f"  params: {[f'{x:.3f}' for x in res.x]}")
    return res.x, C_pred, std, rms_pct, max_abs


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

    # ──── Reference: 8-param baseline ────
    print("=" * 60)
    p0_8 = [200.0, 1.7, 44.0, 6.0, 0.15, 5.2, 1.0, -80.0]
    lo_8 = [10.0, 0.1, 30.0, 0.5, 0.0, 0.5, 0.2, -300.0]
    hi_8 = [500.0, 10.0, 60.0, 15.0, 1.0, 8.0, 8.0, +300.0]
    p_8, Cm_8, std_8, rms_8, max_8 = fit_and_eval(
        "8-param baseline (α=β)", lambda p, m, t: model_base(p, m, t),
        p0_8, lo_8, hi_8, mlat_centers, t_years, C_data, mask, n_data)

    # ──── A: 9-param fixed step ────
    p0_A = p0_8 + [-7.0]
    lo_A = lo_8 + [-30.0]
    hi_A = hi_8 + [+30.0]
    p_A, Cm_A, std_A, rms_A, max_A = fit_and_eval(
        "A: 9-param fixed step (μ=16°, k=0.5° fixed)", model_A,
        p0_A, lo_A, hi_A, mlat_centers, t_years, C_data, mask, n_data)

    # ──── B: 11-param free step ────
    p0_B = p0_8 + [-7.0, 16.0, 1.0]
    lo_B = lo_8 + [-30.0, 5.0, 0.1]
    hi_B = hi_8 + [+30.0, 30.0, 10.0]
    p_B, Cm_B, std_B, rms_B, max_B = fit_and_eval(
        "B: 11-param free step", model_B,
        p0_B, lo_B, hi_B, mlat_centers, t_years, C_data, mask, n_data)

    # ──── C: 10-param dual mlat sigmoid ────
    # params: a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t  (no C0 → 10 params)
    p0_C = [200.0, 1.7, 44.0, 6.0, -0.05, 16.0, 1.5, 0.15, 5.2, 1.0]
    lo_C = [10.0, 0.1, 30.0, 0.5, -2.0, 5.0, 0.1, 0.0, 0.5, 0.2]
    hi_C = [500.0, 10.0, 60.0, 15.0, 2.0, 30.0, 10.0, 1.0, 8.0, 8.0]
    # Add C_0 too → 11 params
    p0_C = p0_C + [-80.0]
    lo_C = lo_C + [-300.0]
    hi_C = hi_C + [+300.0]
    p_C, Cm_C, std_C, rms_C, max_C = fit_and_eval(
        "C: 11-param dual mlat sigmoid (low+high)", model_C,
        p0_C, lo_C, hi_C, mlat_centers, t_years, C_data, mask, n_data)

    # ──── Plot marginal C(mlat) for each ────
    Cm = np.where(mask, C_data, np.nan)
    with np.errstate(invalid='ignore'):
        C_mlat_data = np.nanmean(Cm, axis=1)
        models = {'8p baseline': Cm_8, '9p fix-step': Cm_A,
                  '11p free-step': Cm_B, '11p dual-σ': Cm_C}
        C_mlat_models = {k: np.nanmean(np.where(mask, v, np.nan), axis=1)
                          for k, v in models.items()}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Variants targeting the 16° kink — marginal C(|mlat|) & residual",
                 fontsize=13, fontweight='bold')

    colors = {'8p baseline': 'C3', '9p fix-step': 'C0',
              '11p free-step': 'C2', '11p dual-σ': 'C1'}

    # Panel 1: full range marginal
    ax = axes[0, 0]
    ax.plot(mlat_centers, C_mlat_data, 'o', ms=4, color='black',
            label='data', zorder=4)
    for k, v in C_mlat_models.items():
        ax.plot(mlat_centers, v, '-', lw=1.8, color=colors[k], label=k)
    ax.axvline(16, ls='--', color='C2', alpha=0.5, lw=1)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("1. Marginal C(|mlat|) — all variants", fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # Panel 2: zoom 0-25°
    ax = axes[0, 1]
    ml = mlat_centers <= 25
    ax.plot(mlat_centers[ml], C_mlat_data[ml], 'o-', ms=5, lw=1, color='black',
            label='data', zorder=4)
    for k, v in C_mlat_models.items():
        ax.plot(mlat_centers[ml], v[ml], '-', lw=1.8, color=colors[k], label=k)
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("2. Zoomed 0-25°: does any variant capture the kink?",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # Panel 3: marginal residual
    ax = axes[1, 0]
    for k, v in C_mlat_models.items():
        ax.plot(mlat_centers, C_mlat_data - v, '-', lw=1.8,
                color=colors[k], label=k)
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.axvline(16, ls='--', color='C2', alpha=0.5, lw=1)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("data − model (cnt/s)", fontsize=11)
    ax.set_title("3. Marginal residual data − model",
                 fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)

    # Panel 4: bar chart of std
    ax = axes[1, 1]
    names = ['8p baseline', '9p fix-step', '11p free-step', '11p dual-σ']
    stds = [std_8, std_A, std_B, std_C]
    bars = ax.bar(names, stds, color=[colors[n] for n in names])
    ax.axhline(5.49, ls=':', color='black', alpha=0.5,
               label='TPS noise floor (5.49)')
    for b, s in zip(bars, stds):
        ax.text(b.get_x()+b.get_width()/2, s+0.05, f'{s:.2f}',
                ha='center', fontsize=10)
    ax.set_ylabel("residual std (cnt/s)", fontsize=11)
    ax.set_title("4. Heatmap residual std comparison", fontsize=11)
    ax.set_ylim(5.0, 7.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_mlat_16_step_variants.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    print("\n=== Summary ===")
    print(f"  8p baseline:     std={std_8:.2f} cnt/s  ({rms_8:.2f}% RMS/mean)")
    print(f"  9p fix-step:     std={std_A:.2f} cnt/s  ({rms_A:.2f}%)")
    print(f"  11p free-step:   std={std_B:.2f} cnt/s  ({rms_B:.2f}%)")
    print(f"  11p dual-σ:      std={std_C:.2f} cnt/s  ({rms_C:.2f}%)")
    print(f"  TPS noise floor: 5.49 cnt/s  (3.07%)")


if __name__ == "__main__":
    main()
