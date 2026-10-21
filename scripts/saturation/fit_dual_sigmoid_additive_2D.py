#!/usr/bin/env python3
"""Fit dual_sigmoid_additive model to C(|mlat|, t) heatmap.

Model (additive, 11 params):
  C(m, t) = a * Fb(t) + b * sigm((m - mu_m)/k_m) * Fm(t) + C0
  Fb(t)   = 1 - ampb * sigm((t - mu_tb)/k_tb)
  Fm(t)   = 1 - ampm * sigm((t - mu_tm)/k_tm)
  sigm(x) = 1 / (1 + exp(-x))

Parameters (11):
  a, b, mu_m, k_m, ampb, mu_tb, k_tb, ampm, mu_tm, k_tm, C0

Idea: baseline (mlat-independent) and mlat-driven excess have INDEPENDENT
time sigmoids, allowing different decay timescales.
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


def model(params, mlat_grid, t_grid):
    """mlat_grid (n_mlat,), t_grid (n_t,) -> C (n_mlat, n_t)."""
    a, b, mu_m, k_m, ampb, mu_tb, k_tb, ampm, mu_tm, k_tm, C0 = params
    sm = sigm((mlat_grid - mu_m) / k_m)              # (n_mlat,)
    Fb = 1.0 - ampb * sigm((t_grid - mu_tb) / k_tb)  # (n_t,)
    Fm = 1.0 - ampm * sigm((t_grid - mu_tm) / k_tm)  # (n_t,)
    baseline = a * Fb[None, :]                       # (1, n_t)
    mlat_term = (b * sm[:, None]) * Fm[None, :]      # (n_mlat, n_t)
    return baseline + mlat_term + C0


def main():
    z = np.load(NPZ)
    C_data = z["C_med"]            # (60, 108) mean cnt/s
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

    # Initial guess: baseline ~ 60 cnt/s at low mlat early times, decaying ~30%.
    # Mlat term contributes ~300 cnt/s at high mlat, also decaying.
    p0 = [
        60.0,    # a   baseline amplitude
        300.0,   # b   mlat sigmoid amplitude
        50.0,    # mu_m
        4.0,     # k_m
        0.3,     # ampb
        3.0,     # mu_tb
        2.0,     # k_tb
        0.5,     # ampm
        3.0,     # mu_tm
        2.0,     # k_tm
        0.0,     # C0
    ]
    lo = [  5.0,  50.0, 30.0, 0.5, 0.0, 0.5, 0.2, 0.0, 0.5, 0.2, -100.0]
    hi = [300.0, 800.0, 60.0, 15.0, 1.0, 8.0, 8.0, 1.0, 8.0, 8.0, +100.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.2f}, nfev: {res.nfev}, success: {res.success}")
    a, b, mu_m, k_m, ampb, mu_tb, k_tb, ampm, mu_tm, k_tm, C0 = res.x

    def y_at(tau):
        return (t0 + np.timedelta64(int(tau * 365.25), 'D')).astype(str)

    print("\n=== Fitted parameters ===")
    print(f"  a       = {a:8.2f} cnt/s   (baseline amplitude)")
    print(f"  b       = {b:8.2f} cnt/s   (mlat sigmoid amplitude)")
    print(f"  mu_m    = {mu_m:8.2f} deg   (mlat inflection)")
    print(f"  k_m     = {k_m:8.2f} deg   (mlat width)")
    print(f"  ampb    = {ampb:8.3f}        (baseline time decline)")
    print(f"  mu_tb   = {mu_tb:8.2f} yr    ~ {y_at(mu_tb)}")
    print(f"  k_tb    = {k_tb:8.2f} yr    (baseline time width)")
    print(f"  ampm    = {ampm:8.3f}        (mlat term time decline)")
    print(f"  mu_tm   = {mu_tm:8.2f} yr    ~ {y_at(mu_tm)}")
    print(f"  k_tm    = {k_tm:8.2f} yr    (mlat term time width)")
    print(f"  C0      = {C0:+8.2f} cnt/s   (offset)")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy(); resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid[mask]
    mean_C = float(np.mean(C_data[mask]))
    rstd = float(np.std(valid_resid))
    rmax = float(np.max(np.abs(valid_resid)))
    rstd_pct = rstd / mean_C * 100.0

    print(f"\n=== Residual stats over {mask.sum()} valid bins ===")
    print(f"  mean:   {np.mean(valid_resid):+.3f} cnt/s")
    print(f"  median: {np.median(valid_resid):+.3f} cnt/s")
    print(f"  std:    {rstd:.3f} cnt/s")
    print(f"  max:    {rmax:.2f} cnt/s")
    print(f"  std / mean(C_data) = {rstd_pct:.2f}%  (mean C = {mean_C:.1f})")

    # ─── Plot ───
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "dual_sigmoid_additive 2D fit  (11 params)\n"
        f"a={a:.1f}, b={b:.1f}, mu_m={mu_m:.1f}°, k_m={k_m:.1f}°, "
        f"ampb={ampb:.2f}, mu_tb={mu_tb:.2f}, k_tb={k_tb:.2f}, "
        f"ampm={ampm:.2f}, mu_tm={mu_tm:.2f}, k_tm={k_tm:.2f}, C0={C0:.1f}",
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
             'C_data (cnt/s)', '1. DATA — mean C')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — dual_sigmoid_additive')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data − model (cnt/s)', '3. RESIDUAL — data − model (±30)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_dual_sigmoid_additive_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    print("\n=== JSON-ish summary ===")
    print(f"residual_std={rstd:.4f}")
    print(f"residual_std_pct={rstd_pct:.4f}")
    print(f"residual_max_abs={rmax:.4f}")
    print(f"fit_cost={res.cost:.4f}")


if __name__ == "__main__":
    main()
