#!/usr/bin/env python3
"""Fit hill_x_logistic_floor model to C(t, |mlat|) heatmap.

Model:
  C(mlat, t) = a * mlat^n_m / (m50^n_m + mlat^n_m)
               * (a_inf + (1 - a_inf) / (1 + exp((t - mu_t) / k_t)))
               + C0

Parameters (7 globals): a, m50, n_m, a_inf, mu_t, k_t, C0
  - a    : amplitude of mlat shape
  - m50  : mlat at which Hill function is half-max (deg)
  - n_m  : Hill exponent (steepness of mlat rise)
  - a_inf: asymptotic time-floor fraction (0..1); time factor goes from 1 to a_inf
  - mu_t : logistic midpoint in years since 2017-06-22
  - k_t  : logistic width (years); positive k_t -> decline over time
  - C0   : constant baseline offset (cnt/s)
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
    a, m50, n_m, a_inf, mu_t, k_t, C0 = params
    # Hill function in mlat: a * mlat^n / (m50^n + mlat^n)
    mlat_pos = np.maximum(mlat_grid, 1e-6)  # avoid 0^n issues
    hill = a * (mlat_pos ** n_m) / (m50 ** n_m + mlat_pos ** n_m)   # (n_mlat,)
    # Logistic decline with non-zero floor a_inf:
    # at t<<mu_t: factor -> a_inf + (1 - a_inf) = 1
    # at t>>mu_t: factor -> a_inf
    z = (t_grid - mu_t) / k_t
    # clip to avoid overflow in exp
    z_clipped = np.clip(z, -50.0, 50.0)
    time_factor = a_inf + (1.0 - a_inf) / (1.0 + np.exp(z_clipped))   # (n_t,)
    return hill[:, None] * time_factor[None, :] + C0                  # (n_mlat, n_t)


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

    # Initial guess based on data inspection:
    #   - C peaks at high mlat (~400 cnt/s) and rises near mlat ~ 40-50 deg
    #   - declines roughly halfway through ~9 yr span
    p0 = [
        400.0,    # a
        45.0,     # m50
        6.0,      # n_m (steep Hill)
        0.5,      # a_inf (floor fraction)
        4.0,      # mu_t (years)
        2.0,      # k_t (years)
        20.0,     # C0
    ]
    lo = [50.0,  10.0, 1.0,  0.0,  0.0,  0.2, -100.0]
    hi = [2000.0, 60.0, 30.0, 1.0, 10.0, 10.0, +200.0]

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"\nfit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, m50, n_m, a_inf, mu_t, k_t, C0 = res.x
    print(f"\n=== Fitted parameters ===")
    print(f"  a       = {a:8.2f}   (amplitude of mlat Hill)")
    print(f"  m50     = {m50:8.2f}°  (mlat half-max)")
    print(f"  n_m     = {n_m:8.3f}   (Hill exponent)")
    print(f"  a_inf   = {a_inf:8.4f}   (time-factor floor fraction)")
    print(f"  mu_t    = {mu_t:8.3f} yr"
          f" = {(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t     = {k_t:8.3f} yr  (logistic width)")
    print(f"  C0      = {C0:+8.2f}  (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid[mask]
    mean_data = np.mean(C_data[mask])
    r_std = float(np.std(valid_resid))
    r_max = float(np.max(np.abs(valid_resid)))
    r_std_pct = float(r_std / mean_data * 100.0)
    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:   {np.mean(valid_resid):+.2f} cnt/s")
    print(f"  median: {np.median(valid_resid):+.2f} cnt/s")
    print(f"  std:    {r_std:.2f} cnt/s")
    print(f"  max:    {r_max:.1f} cnt/s")
    print(f"  std / mean(C_data): {r_std_pct:.2f}%")

    # ─── 3-panel plot ───
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "hill_x_logistic_floor 2D fit:  "
        "C = a·mlat^n/(m50^n+mlat^n)·(a_inf+(1-a_inf)/(1+exp((t-mu_t)/k_t))) + C0\n"
        f"a={a:.0f}, m50={m50:.2f}°, n_m={n_m:.2f}, a_inf={a_inf:.3f}, "
        f"mu_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C0={C0:.1f}\n"
        f"residual std = {r_std:.2f} cnt/s ({r_std_pct:.2f}% of mean C)",
        fontsize=11, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1])/2),
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
             'C_model (cnt/s)', '2. MODEL — hill_x_logistic_floor fit')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data − model (cnt/s)',
             '3. RESIDUAL — data − model (symmetric ±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_hill_x_logistic_floor_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # Final structured summary lines
    print("\n=== STRUCTURED RESULT ===")
    print(f"  model_name        : hill_x_logistic_floor")
    print(f"  n_params          : 7")
    print(f"  residual_std      : {r_std:.4f}")
    print(f"  residual_std_pct  : {r_std_pct:.4f}")
    print(f"  residual_max_abs  : {r_max:.4f}")
    print(f"  fit_cost          : {res.cost:.4f}")
    print(f"  params : a={a:.4f} m50={m50:.4f} n_m={n_m:.4f} "
          f"a_inf={a_inf:.4f} mu_t={mu_t:.4f} k_t={k_t:.4f} C0={C0:.4f}")


if __name__ == "__main__":
    main()
