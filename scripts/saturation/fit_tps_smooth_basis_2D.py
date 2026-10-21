#!/usr/bin/env python3
"""Fit a tensor-product smooth-basis model to the C(t, |mlat|) heatmap.

Model:
  C(mlat, t) = sum_{i=1..5} sum_{j=1..5} w_ij * B_i(mlat) * B_j(t) + C_0

  B_i and B_j are sigmoid radial basis functions centered on fixed knots
  spanning the data range. The knot widths are fixed to the inter-knot
  spacing. Only the linear weights w_ij and the global offset C_0 are
  optimized (26 parameters total).

  This is a smooth-basis sanity check: it should approach the irreducible
  per-bin noise floor and therefore tells us the lower bound on the
  achievable residual std from cell-level statistics alone.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"

N_KNOTS_M = 5
N_KNOTS_T = 5


def make_basis(x, knots, width):
    """Sigmoid radial basis centered at each knot.
    Returns array of shape (len(x), len(knots)).
    """
    # logistic-shaped bump: derivative of sigmoid (sech^2 like).
    # Use a Gaussian-shaped sigmoid bump: 1 / (cosh((x-mu)/w)) is the radial bump.
    # Simpler: use Gaussian-style sigmoid basis = exp(-((x-mu)/w)^2).
    # But the spec says "radial basis (sigmoid)" -> use sech^2 which is the
    # derivative of a sigmoid (logistic) and is the standard "sigmoid bump".
    z = (x[:, None] - knots[None, :]) / width
    # sech^2(z) = 4 / (exp(z) + exp(-z))^2
    # numerically: 1/cosh(z)^2
    return 1.0 / np.cosh(z) ** 2


def model_2d(params, B_m, B_t):
    """Build C(mlat, t) given basis matrices and weight vector.
    B_m: (n_mlat, K_m)
    B_t: (n_t, K_t)
    params: (K_m * K_t + 1,) -> last is C_0
    """
    K_m = B_m.shape[1]
    K_t = B_t.shape[1]
    W = params[: K_m * K_t].reshape(K_m, K_t)
    C_0 = params[-1]
    # C(i,j) = sum_p sum_q B_m[i,p] W[p,q] B_t[j,q]
    return B_m @ W @ B_t.T + C_0


def main():
    z = np.load(NPZ)
    C_data = z["C_med"]            # mean cnt/s (n_mlat, n_t)
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

    # Knot placement: evenly across the data range.
    mlat_lo, mlat_hi = float(mlat_centers.min()), float(mlat_centers.max())
    t_lo, t_hi = float(t_years.min()), float(t_years.max())
    mlat_knots = np.linspace(mlat_lo, mlat_hi, N_KNOTS_M)
    t_knots = np.linspace(t_lo, t_hi, N_KNOTS_T)
    width_m = (mlat_knots[1] - mlat_knots[0])  # inter-knot spacing in deg
    width_t = (t_knots[1] - t_knots[0])        # inter-knot spacing in yr

    print(f"mlat knots (deg): {mlat_knots}")
    print(f"t knots (yr):    {t_knots}")
    print(f"width_m = {width_m:.3f} deg, width_t = {width_t:.3f} yr")

    B_m = make_basis(mlat_centers, mlat_knots, width_m)  # (n_mlat, K_m)
    B_t = make_basis(t_years, t_knots, width_t)          # (n_t, K_t)
    print(f"B_m shape {B_m.shape}, B_t shape {B_t.shape}")

    K_m = B_m.shape[1]
    K_t = B_t.shape[1]
    n_params = K_m * K_t + 1
    print(f"n_params = {n_params}")

    def residual(params):
        C_pred = model_2d(params, B_m, B_t)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # Initial guess: weights ~ data mean, offset 0.
    mean_C = float(np.mean(C_data[mask]))
    p0 = np.full(n_params, 0.0)
    # Set diagonal-ish initialization to roughly mean.
    p0[: K_m * K_t] = mean_C / (K_m * K_t / 5.0)
    p0[-1] = 0.0

    lo = np.full(n_params, -2000.0)
    hi = np.full(n_params, +2000.0)
    lo[-1] = -200.0
    hi[-1] = +200.0

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.3f}, nfev: {res.nfev}, success: {res.success}")

    W = res.x[: K_m * K_t].reshape(K_m, K_t)
    C_0 = res.x[-1]
    print("\n=== Fitted weights W (rows = mlat knots, cols = time knots) ===")
    for i in range(K_m):
        row = "  ".join(f"{W[i,j]:+8.2f}" for j in range(K_t))
        print(f"  mlat={mlat_knots[i]:5.1f}:  {row}")
    print(f"  C_0 = {C_0:+.3f}")

    C_pred = model_2d(res.x, B_m, B_t)
    resid_full = (C_data - C_pred) * mask
    resid_masked = resid_full.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid_full[mask]
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
        f"tps_smooth_basis 2D fit:  C = sum_ij w_ij * B_i(mlat) * B_j(t) + C_0   "
        f"(K_m={K_m}, K_t={K_t}, n_params={n_params})\n"
        f"resid std = {r_std:.2f} cnt/s ({r_std_pct:.2f}%), max |resid| = {r_max:.2f} cnt/s",
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
             'C_model (cnt/s)', '2. MODEL - tensor-product smooth basis')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data - model (cnt/s)',
             '3. RESIDUAL - data minus model (symmetric +/- 30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_tps_smooth_basis_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")

    # Return a small dict-like print for downstream collection.
    print("\n=== STRUCTURED ===")
    print(f"n_params={n_params}")
    print(f"residual_std={r_std:.6f}")
    print(f"residual_std_pct={r_std_pct:.6f}")
    print(f"residual_max_abs={r_max:.6f}")
    print(f"fit_cost={res.cost:.6f}")
    print(f"plot_path=plots/fit_tps_smooth_basis_2D.png")


if __name__ == "__main__":
    main()
