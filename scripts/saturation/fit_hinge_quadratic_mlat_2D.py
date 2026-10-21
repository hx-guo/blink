#!/usr/bin/env python3
"""Fit the hinge_quadratic_mlat 2D model to the C(t, |mlat|) heatmap.

Model (6 params):
    C(mlat, t) = (a + b·max(0, mlat − θ1)²) · (1 − amp·σ_t((t − μ_t)/k_t)) + C0
    σ_t(x) = 1 / (1 + exp(−x))

Parameters: a, b, theta1, amp, mu_t, k_t, C0   (6 total)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def model(params, mlat_grid, t_grid):
    """mlat_grid (n_mlat,), t_grid (n_t,). Returns (n_mlat, n_t) C."""
    a, b, theta1, amp, mu_t, k_t, C0 = params
    excess = np.maximum(0.0, mlat_grid - theta1)            # (n_mlat,)
    A = a + b * excess * excess                              # (n_mlat,)
    sigma_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))   # (n_t,)
    F = 1.0 - amp * sigma_t                                   # (n_t,)
    return A[:, None] * F[None, :] + C0                       # (n_mlat, n_t)


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
    print(f"valid bins: {mask.sum()}/{mask.size}")
    print(f"mlat range: {mlat_centers[0]:.1f}..{mlat_centers[-1]:.1f}")
    print(f"t range: {t_years[0]:.2f}..{t_years[-1]:.2f} yr")
    print(f"C range (valid): {C_data[mask].min():.1f}..{C_data[mask].max():.1f}")

    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # Initial guess:
    #   a ~ low-mlat plateau value (~80 cnt/s); b small positive quadratic coeff
    #   theta1 ~ 20 deg (hinge); amp ~ 0.6 (time decline ~60%)
    #   mu_t ~ 3 yr (mid time); k_t ~ 2 yr; C0 ~ 0
    p0 = [
        80.0,    # a
        0.3,     # b   (cnt/s per deg^2)
        20.0,    # theta1
        0.6,     # amp
        3.0,     # mu_t
        2.0,     # k_t
        0.0,     # C0
    ]
    lo = [10.0,  0.0,  5.0,  0.0,  0.5, 0.2, -100.0]
    hi = [400.0, 5.0,  45.0, 1.5,  8.0, 8.0, +100.0]

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"\nfit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, b, theta1, amp, mu_t, k_t, C0 = res.x
    print("\n=== Fitted parameters ===")
    print(f"  a        = {a:8.3f}   (low-mlat plateau, cnt/s)")
    print(f"  b        = {b:8.4f}   (quadratic coef, cnt/s/deg^2)")
    print(f"  theta1   = {theta1:8.2f}°  (hinge knee)")
    print(f"  amp      = {amp:8.3f}   (time decline amplitude)")
    print(f"  mu_t     = {mu_t:8.3f} yr")
    print(f"  k_t      = {k_t:8.3f} yr")
    print(f"  C0       = {C0:+8.3f}  (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_m = resid.copy()
    resid_m[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid[mask]
    mean_C = float(np.mean(C_data[mask]))
    resid_std = float(np.std(valid_resid))
    resid_max_abs = float(np.max(np.abs(valid_resid)))
    resid_std_pct = resid_std / mean_C * 100.0

    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:        {np.mean(valid_resid):+.3f} cnt/s")
    print(f"  median:      {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std:         {resid_std:.3f} cnt/s")
    print(f"  max |resid|: {resid_max_abs:.2f} cnt/s")
    print(f"  std / mean(C_data): {resid_std_pct:.2f}%")
    print(f"  fit_cost:    {res.cost:.4f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"hinge_quadratic_mlat 2D fit:  C = (a + b·max(0,mlat−θ1)²)·(1 − amp·σ_t((t−μ_t)/k_t)) + C0\n"
        f"a={a:.1f}, b={b:.3f}, θ1={theta1:.1f}°, amp={amp:.2f}, "
        f"μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C0={C0:+.1f}",
        fontsize=12, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data,
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel("|mlat| (deg)", fontsize=11)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_m, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA — mean C (cnt/s)')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — hinge_quadratic_mlat')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data − model (cnt/s)',
             '3. RESIDUAL — data − model (symmetric ±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_hinge_quadratic_mlat_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")

    # Print a machine-friendly summary for the harness
    print("\n=== SUMMARY ===")
    print(f"  model_name:      hinge_quadratic_mlat")
    print(f"  n_params:        7")
    print(f"  residual_std:    {resid_std:.4f}")
    print(f"  residual_std_pct: {resid_std_pct:.4f}")
    print(f"  residual_max_abs: {resid_max_abs:.4f}")
    print(f"  fit_cost:        {res.cost:.6f}")
    print(f"  plot_path:       {out}")


if __name__ == "__main__":
    main()
