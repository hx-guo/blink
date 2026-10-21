#!/usr/bin/env python3
"""Fit the sigmoid 2D model to the C(t, |mlat|) heatmap and look at residual.

Model:
  C(mlat, t) = A(mlat) · F(t) + C_0

  A(mlat) = a · [1 + α · σ_m(|mlat|)]
  σ_m(m) = 1 / (1 + exp(-(m - μ_m)/k_m))      mlat sigmoid

  F(t) = 1 - amp · σ_t(t)
  σ_t(t) = 1 / (1 + exp(-(t - μ_t)/k_t))       (note: positive slope inside,
                                                  so F decreases over time)

Parameters (8 globals):  a, α, μ_m, k_m, amp, μ_t, k_t, C_0
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def model(params, mlat_grid, t_grid):
    """mlat_grid shape (n_mlat,), t_grid shape (n_t,). Returns (n_mlat, n_t) C."""
    a, alpha, mu_m, k_m, amp, mu_t, k_t, C_0 = params
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))     # (n_mlat,)
    sigma_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))         # (n_t,)
    A = a * (1.0 + alpha * sigma_m)                                # (n_mlat,)
    F = 1.0 - amp * sigma_t                                         # (n_t,)
    return A[:, None] * F[None, :] + C_0                            # (n_mlat, n_t)


def main():
    z = np.load(NPZ)
    C_data = z["C_med"]           # mean now
    n_data = z["C_n"]
    months = z["months"]
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    # t in years since 2017-06-22 (v5t epoch); use month-mid for binning t
    t0 = np.datetime64("2017-06-22")
    t_years = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25

    mask = n_data > 200   # valid bins
    print(f"valid bins: {mask.sum()}/{mask.size}")

    # ─── Fit ───
    # Pre-clean C_data: NaN → 0 in invalid bins, mask weights them out
    C_data_clean = np.where(mask, C_data, 0.0)
    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask    # zero contribution for invalid bins
        return r.ravel()

    # Initial guess from inspection of heatmap
    p0 = [
        100.0,     # a (global amplitude scale)
        3.0,       # α (mlat sigmoid amplitude relative to a)
        50.0,      # μ_m (mlat inflection, around 50-55°)
        4.0,       # k_m (mlat sigmoid width)
        0.6,       # amp (time decline amplitude, 0..1)
        3.0,       # μ_t (time inflection in years since t0, ~2020 = year 3)
        2.0,       # k_t (time sigmoid width, ~2 years)
        0.0,       # C_0 (offset)
    ]
    lo = [10.0, 0.5, 30.0, 0.5, 0.0,  0.5, 0.2, -100.0]
    hi = [500.0, 20.0, 60.0, 15.0, 1.5, 8.0, 8.0, +100.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=2000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_m, k_m, amp, mu_t, k_t, C_0 = res.x
    print(f"\n=== Fitted parameters ===")
    print(f"  a       = {a:8.2f}   (global amplitude)")
    print(f"  α       = {alpha:8.3f}   (mlat sigmoid scale)")
    print(f"  μ_m     = {mu_m:8.2f}°  (mlat inflection)")
    print(f"  k_m     = {k_m:8.2f}°  (mlat width)")
    print(f"  amp     = {amp:8.3f}   (time decline amplitude)")
    print(f"  μ_t     = {mu_t:8.2f} yr = {(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t     = {k_t:8.2f} yr  (time width)")
    print(f"  C_0     = {C_0:+8.2f}  (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    # Stats
    valid_resid = resid[mask]
    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:   {np.mean(valid_resid):+.2f} cnt/s")
    print(f"  median: {np.median(valid_resid):+.2f} cnt/s")
    print(f"  std:    {np.std(valid_resid):.2f} cnt/s")
    print(f"  max:    {np.max(np.abs(valid_resid)):.1f} cnt/s")
    print(f"  RMS / mean(C_data): {np.sqrt(np.mean(valid_resid**2)) / np.mean(C_data[mask])*100:.1f}%")

    # ─── Plot 3-panel: data / model / residual ───
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"Sigmoid 2D fit:  C = a·[1 + α·σ_m(mlat)]·[1 − amp·σ_t(t)] + C_0\n"
        f"a={a:.0f}, α={alpha:.2f}, μ_m={mu_m:.1f}°, k_m={k_m:.1f}°, "
        f"amp={amp:.2f}, μ_t={mu_t:.1f}yr, k_t={k_t:.1f}yr, C_0={C_0:.0f}",
        fontsize=12, fontweight='bold')

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
             'C_model (cnt/s)', '2. MODEL — sigmoid fit')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data − model (cnt/s)', '3. RESIDUAL — data − model (symmetric ±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_sigmoid_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
