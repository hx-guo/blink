#!/usr/bin/env python3
"""Fit the sigmoid (mlat) x pure exponential (time) 2D model to C(t, |mlat|).

Model:
  C(mlat, t) = a * (1 + alpha * sigm((mlat - mu_m)/k_m)) * exp(-t/tau) + C0

  sigm(x) = 1 / (1 + exp(-x))

Parameters (6 globals): a, alpha, mu_m, k_m, tau, C0
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def model(params, mlat_grid, t_grid):
    """mlat_grid shape (n_mlat,), t_grid shape (n_t,). Returns (n_mlat, n_t) C."""
    a, alpha, mu_m, k_m, tau, C0 = params
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))     # (n_mlat,)
    A = a * (1.0 + alpha * sigma_m)                                # (n_mlat,)
    F = np.exp(-t_grid / tau)                                       # (n_t,)
    return A[:, None] * F[None, :] + C0                             # (n_mlat, n_t)


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

    # Initial guess
    # data range ~30..420 cnt/s; at large mlat early ~ a*(1+alpha) ~ 400
    # at low mlat early ~ a ~ 100
    # decay over 8.9 yr: pick tau ~ 8 yr so exp(-1) at the end
    p0 = [
        100.0,    # a
        3.0,      # alpha
        50.0,     # mu_m
        4.0,      # k_m
        8.0,      # tau (years)
        0.0,      # C0
    ]
    lo = [10.0, 0.1, 20.0, 0.5, 0.5, -100.0]
    hi = [500.0, 30.0, 60.0, 20.0, 100.0, 200.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_m, k_m, tau, C0 = res.x
    print(f"\n=== Fitted parameters ===")
    print(f"  a       = {a:8.3f}")
    print(f"  alpha   = {alpha:8.4f}")
    print(f"  mu_m    = {mu_m:8.3f} deg")
    print(f"  k_m     = {k_m:8.3f} deg")
    print(f"  tau     = {tau:8.3f} yr")
    print(f"  C0      = {C0:+8.3f}")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    # Stats
    valid_resid = resid[mask]
    mean_C = float(np.mean(C_data[mask]))
    std_r = float(np.std(valid_resid))
    max_abs = float(np.max(np.abs(valid_resid)))
    pct = std_r / mean_C * 100.0
    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:   {np.mean(valid_resid):+.3f} cnt/s")
    print(f"  median: {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std:    {std_r:.3f} cnt/s")
    print(f"  max:    {max_abs:.3f} cnt/s")
    print(f"  std / mean(C_data): {pct:.3f}%")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "sigmoid_x_exp 2D fit:  C = a*(1+alpha*sigm((mlat-mu_m)/k_m))*exp(-t/tau) + C0\n"
        f"a={a:.2f}, alpha={alpha:.3f}, mu_m={mu_m:.2f}, k_m={k_m:.2f}, "
        f"tau={tau:.2f}yr, C0={C0:.2f}  | std={std_r:.2f} cnt/s ({pct:.2f}%)",
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
             'C_data (cnt/s)', '1. DATA - mean C (cnt/s)')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL - sigmoid_x_exp fit')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL - data - model (symmetric +-30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_sigmoid_x_exp_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
