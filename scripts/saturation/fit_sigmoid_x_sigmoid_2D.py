#!/usr/bin/env python3
"""Fit the sigmoid x sigmoid 2D model to the C(t, |mlat|) heatmap.

Model:
  C(mlat, t) = a · [1 + alpha · sigm((mlat - mu_m)/k_m)]
                · [1 - amp · sigm((t - mu_t)/k_t)] + C_0

  Eight global parameters: a, alpha, mu_m, k_m, amp, mu_t, k_t, C_0.

  This is the baseline reference model. The residual is expected to show
  systematic structure (~7 cnt/s std) at high |mlat| with sign reversal
  across years; we re-run it here for a self-consistent ablation table.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def model(params, mlat_grid, t_grid):
    """mlat_grid shape (n_mlat,), t_grid shape (n_t,). Returns (n_mlat, n_t)."""
    a, alpha, mu_m, k_m, amp, mu_t, k_t, C_0 = params
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))   # (n_mlat,)
    sigma_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))      # (n_t,)
    A = a * (1.0 + alpha * sigma_m)                              # (n_mlat,)
    F = 1.0 - amp * sigma_t                                      # (n_t,)
    return A[:, None] * F[None, :] + C_0                         # (n_mlat, n_t)


def main():
    z = np.load(NPZ)
    C_data = z["C_med"]            # mean cnt/s
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

    # Initial guess + bounds (same as baseline script)
    p0 = [
        100.0,   # a
        3.0,     # alpha
        50.0,    # mu_m
        4.0,     # k_m
        0.6,     # amp
        3.0,     # mu_t
        2.0,     # k_t
        0.0,     # C_0
    ]
    lo = [10.0, 0.5, 30.0, 0.5, 0.0, 0.5, 0.2, -100.0]
    hi = [500.0, 20.0, 60.0, 15.0, 1.5, 8.0, 8.0, +100.0]

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.3f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_m, k_m, amp, mu_t, k_t, C_0 = res.x
    print("\n=== Fitted parameters ===")
    print(f"  a       = {a:8.3f}   (global amplitude)")
    print(f"  alpha   = {alpha:8.4f}  (mlat sigmoid scale)")
    print(f"  mu_m    = {mu_m:8.3f}   (mlat inflection, deg)")
    print(f"  k_m     = {k_m:8.3f}   (mlat sigmoid width, deg)")
    print(f"  amp     = {amp:8.4f}  (time decline amplitude)")
    print(f"  mu_t    = {mu_t:8.3f}   (time inflection, yr since t0)")
    print(f"  k_t     = {k_t:8.3f}   (time sigmoid width, yr)")
    print(f"  C_0     = {C_0:+8.3f}  (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid_full = (C_data - C_pred) * mask     # data - model
    resid_masked = resid_full.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid_full[mask]
    mean_C = float(np.mean(C_data[mask]))
    r_mean = float(np.mean(valid_resid))
    r_std = float(np.std(valid_resid))
    r_max = float(np.max(np.abs(valid_resid)))
    r_std_pct = r_std / mean_C * 100.0

    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:           {r_mean:+.3f} cnt/s")
    print(f"  median:         {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std:            {r_std:.3f} cnt/s")
    print(f"  max |resid|:    {r_max:.3f} cnt/s")
    print(f"  mean(C_data):   {mean_C:.3f} cnt/s")
    print(f"  std / mean(C):  {r_std_pct:.3f} %")
    print(f"  fit cost (0.5*sum r^2): {res.cost:.3f}")

    # 3-panel plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"sigmoid_x_sigmoid 2D fit:  "
        f"C = a*(1+alpha*sigm_m)*(1-amp*sigm_t) + C_0\n"
        f"a={a:.1f}, alpha={alpha:.2f}, mu_m={mu_m:.1f}, k_m={k_m:.1f}, "
        f"amp={amp:.2f}, mu_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C_0={C_0:.1f}  "
        f"|  resid std = {r_std:.2f} cnt/s ({r_std_pct:.2f}%)",
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
             'C_data (cnt/s)', '1. DATA - mean C (cnt/s)')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL - sigmoid x sigmoid')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data - model (cnt/s)',
             '3. RESIDUAL - data minus model (symmetric +/- 30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_sigmoid_x_sigmoid_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
