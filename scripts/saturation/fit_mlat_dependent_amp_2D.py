#!/usr/bin/env python3
"""Fit the mlat_dependent_amp 2D model to the C(t, |mlat|) heatmap.

Model:
  C(mlat, t) = a · [1 + α · σ_m(mlat)] · [1 − amp0 · (1 + β · σ_m(mlat)) · σ_t(t)] + C_0

  σ_m(m) = 1 / (1 + exp(-(m - μ_m)/k_m))      mlat sigmoid
  σ_t(t) = 1 / (1 + exp(-(t - μ_t)/k_t))       time sigmoid

Parameters (10 globals): a, α, μ_m, k_m, amp0, β, μ_t, k_t, C_0  (note: σ_m used for both
amplitude AND time-decay-amplitude — single mlat sigmoid shared across both terms)

Idea: high-mlat bins decay more (or less) than low-mlat bins; controlled by β.
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
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))   # (n_mlat,)
    sigma_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))       # (n_t,)
    A = a * (1.0 + alpha * sigma_m)                              # (n_mlat,)
    amp_eff = amp0 * (1.0 + beta * sigma_m)                      # (n_mlat,)
    F = 1.0 - amp_eff[:, None] * sigma_t[None, :]                # (n_mlat, n_t)
    return A[:, None] * F + C_0                                  # (n_mlat, n_t)


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

    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # 9 params: a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0
    p0 = [
        100.0,    # a
        3.0,      # alpha (mlat amplitude scale)
        50.0,     # mu_m
        4.0,      # k_m
        0.5,      # amp0 (baseline time-decay amplitude at low mlat)
        0.5,      # beta (extra decay at high mlat)
        3.0,      # mu_t
        2.0,      # k_t
        0.0,      # C_0
    ]
    lo = [10.0, 0.5, 30.0, 0.5, 0.0, -2.0, 0.5, 0.2, -100.0]
    hi = [500.0, 20.0, 60.0, 15.0, 1.5, 5.0, 8.0, 8.0, +100.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = res.x
    print(f"\n=== Fitted parameters (9 free) ===")
    print(f"  a       = {a:8.2f}   (global amplitude)")
    print(f"  alpha   = {alpha:8.3f}   (mlat amplitude scale)")
    print(f"  mu_m    = {mu_m:8.2f}   (mlat inflection deg)")
    print(f"  k_m     = {k_m:8.2f}   (mlat width deg)")
    print(f"  amp0    = {amp0:8.3f}   (baseline time-decay amplitude)")
    print(f"  beta    = {beta:8.3f}   (extra decay at high mlat)")
    print(f"  mu_t    = {mu_t:8.2f}   (time inflection yr, ~{(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)})")
    print(f"  k_t     = {k_t:8.2f}   (time width yr)")
    print(f"  C_0     = {C_0:+8.2f}   (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy().astype(float)
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy().astype(float); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy().astype(float); C_pred_m[~mask] = np.nan

    valid_resid = resid[mask]
    mean_C = float(np.mean(C_data[mask]))
    std_r = float(np.std(valid_resid))
    max_abs_r = float(np.max(np.abs(valid_resid)))
    std_pct = std_r / mean_C * 100.0
    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:   {np.mean(valid_resid):+.3f} cnt/s")
    print(f"  median: {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std:    {std_r:.3f} cnt/s")
    print(f"  max:    {max_abs_r:.2f} cnt/s")
    print(f"  std/mean(C_data): {std_pct:.2f}%")

    # 3-panel plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "mlat_dependent_amp 2D fit:  C = a*(1+a*sm)*(1 - amp0*(1+b*sm)*st) + C0\n"
        f"a={a:.0f}, alpha={alpha:.2f}, mu_m={mu_m:.1f}, k_m={k_m:.1f}, "
        f"amp0={amp0:.2f}, beta={beta:.2f}, mu_t={mu_t:.1f}, k_t={k_t:.1f}, C0={C_0:.0f}\n"
        f"residual std = {std_r:.2f} cnt/s ({std_pct:.2f}% of mean)",
        fontsize=11, fontweight='bold')

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
             'C_model (cnt/s)', '2. MODEL — mlat_dependent_amp fit')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL — data - model (symmetric +/- 30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_mlat_dependent_amp_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    print("\n=== JSON summary ===")
    import json
    summary = {
        "model_name": "mlat_dependent_amp",
        "n_params": 9,
        "fit_cost": float(res.cost),
        "params": {
            "a": float(a), "alpha": float(alpha), "mu_m": float(mu_m),
            "k_m": float(k_m), "amp0": float(amp0), "beta": float(beta),
            "mu_t": float(mu_t), "k_t": float(k_t), "C_0": float(C_0),
        },
        "residual_std": std_r,
        "residual_std_pct": std_pct,
        "residual_max_abs": max_abs_r,
        "plot_path": out,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
