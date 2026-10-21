#!/usr/bin/env python3
"""Fit the mlat_quartic_x_sigmoid 2D model to the C(t, |mlat|) heatmap.

Model:
  C(mlat, t) = (a + b*mlat**2 + c*mlat**4) * (1 - amp * sigm((t - mu_t)/k_t)) + C0

  - Polynomial mlat (even powers only: 1, mlat^2, mlat^4)
  - Sigmoid time

Parameters (7): a, b, c, amp, mu_t, k_t, C0
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
    a, b, c, amp, mu_t, k_t, C0 = params
    # Polynomial in mlat (even powers)
    A = a + b * mlat_grid**2 + c * mlat_grid**4              # (n_mlat,)
    # Sigmoid time
    sigma_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))   # (n_t,)
    F = 1.0 - amp * sigma_t                                  # (n_t,)
    return A[:, None] * F[None, :] + C0                      # (n_mlat, n_t)


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

    # Pre-clean
    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # Initial guess
    #   At mlat=0, t=0: C ~ a + C0  (low-mlat early). Looking at data, ~120 cnt/s
    #   At mlat=55, t=0: C ~ a + b*55^2 + c*55^4 + C0, much larger (high-mlat, early)
    #   For "early-vs-late" decline, amp ~ 0.5, mu_t ~ 3 yr, k_t ~ 2 yr
    p0 = [
        100.0,      # a   (constant term)
        0.05,       # b   (mlat^2 coef)
        1e-5,       # c   (mlat^4 coef)
        0.5,        # amp (time decline)
        3.0,        # mu_t (yr)
        2.0,        # k_t  (yr)
        0.0,        # C0
    ]
    lo = [10.0,  -1.0,  -1e-3, 0.0,  0.5, 0.2, -100.0]
    hi = [500.0,  1.0,   1e-3, 1.5,  8.0, 8.0,  100.0]

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")

    a, b, c, amp, mu_t, k_t, C0 = res.x
    print("\n=== Fitted parameters ===")
    print(f"  a    = {a:10.4f}   (constant term)")
    print(f"  b    = {b:10.6f}   (mlat^2 coef)")
    print(f"  c    = {c:10.4e}   (mlat^4 coef)")
    print(f"  amp  = {amp:10.4f}   (time decline amplitude)")
    print(f"  mu_t = {mu_t:10.4f} yr "
          f"= {(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t  = {k_t:10.4f} yr (time sigmoid width)")
    print(f"  C0   = {C0:+10.4f}   (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid[mask]
    mean_C = np.mean(C_data[mask])
    std_r = float(np.std(valid_resid))
    max_abs_r = float(np.max(np.abs(valid_resid)))
    pct = std_r / mean_C * 100

    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:   {np.mean(valid_resid):+.3f} cnt/s")
    print(f"  median: {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std:    {std_r:.3f} cnt/s")
    print(f"  max:    {max_abs_r:.2f} cnt/s")
    print(f"  std / mean(C_data) = {pct:.2f}%")
    print(f"  fit_cost = {float(res.cost):.4f}")

    # 3-panel plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "mlat_quartic x sigmoid 2D fit:\n"
        "C = (a + b*mlat^2 + c*mlat^4) * (1 - amp*sigm((t-mu_t)/k_t)) + C0\n"
        f"a={a:.1f}, b={b:.4f}, c={c:.2e}, amp={amp:.3f}, "
        f"mu_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C0={C0:+.2f}",
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
             'C_model (cnt/s)', '2. MODEL - mlat quartic x sigmoid time')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data - model (cnt/s)',
             '3. RESIDUAL - data - model (symmetric +/-30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_mlat_quartic_x_sigmoid_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
