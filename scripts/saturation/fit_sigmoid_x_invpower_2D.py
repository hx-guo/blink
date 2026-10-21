#!/usr/bin/env python3
"""Fit candidate model: sigmoid (mlat) x inverse-power (time).

Model (7 params):
  C(mlat, t) = a * (1 + alpha * sigm((mlat - mu_m)/k_m)) / (1 + (t/tau)**p) + C0

where sigm(x) = 1/(1+exp(-x)).
Lorentzian-like inverse-power decline in time, sigmoid rise vs |mlat|.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def model(params, mlat_grid, t_grid):
    """mlat_grid (n_mlat,), t_grid (n_t,) -> (n_mlat, n_t)."""
    a, alpha, mu_m, k_m, tau, p, C0 = params
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))           # (n_mlat,)
    A = a * (1.0 + alpha * sigma_m)                                      # (n_mlat,)
    # protect t/tau when t=0; small epsilon
    t_safe = np.maximum(t_grid, 1e-6)
    F = 1.0 / (1.0 + (t_safe / tau) ** p)                                # (n_t,)
    return A[:, None] * F[None, :] + C0                                  # (n_mlat, n_t)


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
    print(f"t_years range: {t_years.min():.3f} .. {t_years.max():.3f}")
    print(f"mlat range: {mlat_centers.min():.1f} .. {mlat_centers.max():.1f}")
    print(f"C_data range over mask: {C_data[mask].min():.1f} .. {C_data[mask].max():.1f}")

    # Pre-clean: zero out invalid bins; mask zeros their contribution to residual
    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # Initial guess: a~100, alpha~3 (sigmoid up to ~4x at high mlat),
    # mu_m~50, k_m~4. tau ~3 yr (Lorentzian-like turnover), p~1.0, C0~30.
    p0 = [100.0, 3.0, 50.0, 4.0, 3.0, 1.0, 30.0]
    lo = [10.0,  0.5, 30.0, 0.5, 0.3, 0.2, -100.0]
    hi = [500.0, 20.0, 60.0, 15.0, 30.0, 8.0, +200.0]

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"\nfit cost: {res.cost:.2f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_m, k_m, tau, p, C0 = res.x
    print(f"\n=== Fitted parameters ===")
    print(f"  a       = {a:8.3f}   (global amplitude)")
    print(f"  alpha   = {alpha:8.3f}   (mlat sigmoid scale)")
    print(f"  mu_m    = {mu_m:8.3f} deg (mlat inflection)")
    print(f"  k_m     = {k_m:8.3f} deg (mlat width)")
    print(f"  tau     = {tau:8.3f} yr  (time inv-power turnover)")
    print(f"  p       = {p:8.3f}      (time inv-power exponent)")
    print(f"  C0      = {C0:+8.3f}    (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid_resid = resid[mask]
    mean_data = float(np.mean(C_data[mask]))
    resid_std = float(np.std(valid_resid))
    resid_max = float(np.max(np.abs(valid_resid)))
    resid_std_pct = resid_std / mean_data * 100.0

    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean(C_data)            = {mean_data:.2f} cnt/s")
    print(f"  mean(resid)             = {np.mean(valid_resid):+.3f} cnt/s")
    print(f"  median(resid)           = {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std(resid)              = {resid_std:.3f} cnt/s")
    print(f"  max |resid|             = {resid_max:.2f} cnt/s")
    print(f"  std/mean(C_data) pct    = {resid_std_pct:.3f} %")

    # ─── 3-panel plot ───
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan
    resid_m = resid.copy().astype(float); resid_m[~mask] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "Sigmoid(mlat) x inverse-power(t) fit:\n"
        f"C = a*(1 + alpha*sigm((mlat-mu_m)/k_m)) / (1 + (t/tau)**p) + C0\n"
        f"a={a:.1f}, alpha={alpha:.2f}, mu_m={mu_m:.2f}, k_m={k_m:.2f}, "
        f"tau={tau:.2f}yr, p={p:.2f}, C0={C0:+.1f}",
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
             'C_model (cnt/s)', '2. MODEL - sigmoid x inverse-power')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL - symmetric +-30 cnt/s')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_sigmoid_x_invpower_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # also emit machine-friendly summary
    print("\n=== SUMMARY ===")
    print(f"FIT_COST={res.cost}")
    print(f"RESID_STD={resid_std}")
    print(f"RESID_STD_PCT={resid_std_pct}")
    print(f"RESID_MAX={resid_max}")


if __name__ == "__main__":
    main()
