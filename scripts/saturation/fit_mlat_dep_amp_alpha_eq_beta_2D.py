#!/usr/bin/env python3
"""8-param simplification: α = β.

Model:
  C(mlat, t) = a · [1 + α·σ_m] · [1 − amp₀·(1 + α·σ_m)·σ_t] + C₀

  σ_m(m) = 1/(1+exp(-(m-μ_m)/k_m))
  σ_t(t) = 1/(1+exp(-(t-μ_t)/k_t))

Forces α = β based on the 9-param fit observation that both converge to ~1.695.

Equivalent rewrite (letting g = 1 + α·σ_m):
  C = a·g − (a·amp₀)·g²·σ_t + C₀
i.e. the time-decay term scales as g² (mlat shape squared).

8 globals: a, α, μ_m, k_m, amp₀, μ_t, k_t, C₀
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def model(params, mlat_grid, t_grid):
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = params
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))
    sigma_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))
    g = 1.0 + alpha * sigma_m                          # mlat shape
    amp_eff = amp0 * g                                 # tied: time-decay amp ~ g
    F = 1.0 - amp_eff[:, None] * sigma_t[None, :]
    return a * g[:, None] * F + C_0


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
        return ((C_pred - C_data_clean) * mask).ravel()

    p0 = [200.0, 1.7, 44.0, 6.0, 0.15, 5.2, 1.0, -80.0]
    lo = [10.0, 0.1, 30.0, 0.5, 0.0, 0.5, 0.2, -300.0]
    hi = [500.0, 10.0, 60.0, 15.0, 1.0, 8.0, 8.0, +300.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = res.x

    print("\n=== Fitted parameters (8-param, α tied to β) ===")
    print(f"  a      = {a:8.2f}")
    print(f"  α      = {alpha:8.3f}   (mlat sigmoid scale; also enhances time decay)")
    print(f"  μ_m    = {mu_m:8.2f}°")
    print(f"  k_m    = {k_m:8.2f}°")
    print(f"  amp₀   = {amp0:8.3f}")
    print(f"  μ_t    = {mu_t:8.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t    = {k_t:8.2f} yr")
    print(f"  C_0    = {C_0:+8.2f}")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_masked = resid.copy()
    resid_masked[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    valid_resid = resid[mask]
    print(f"\n=== Residual stats ===")
    print(f"  mean:   {np.mean(valid_resid):+.2f} cnt/s")
    print(f"  std:    {np.std(valid_resid):.2f} cnt/s")
    print(f"  max:    {np.max(np.abs(valid_resid)):.1f} cnt/s")
    print(f"  RMS/mean: {np.sqrt(np.mean(valid_resid**2)) / np.mean(C_data[mask])*100:.2f}%")
    print(f"\nCompare with 9-param: std=6.49 cnt/s (RMS/mean=3.61%)")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "8-param α=β simplification:  C = a·g·[1 − amp₀·g·σ_t] + C₀,  g = 1 + α·σ_m\n"
        f"a={a:.0f}, α={alpha:.2f}, μ_m={mu_m:.1f}°, k_m={k_m:.1f}°, "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C_0:+.0f}",
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
             'C_model (cnt/s)', '2. MODEL — α=β 8 params')
    plot_pcm(axes[2], resid_masked, -30, 30, 'RdBu_r',
             'data − model (cnt/s)', '3. RESIDUAL — data − model (symmetric ±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_mlat_dep_amp_alpha_eq_beta_2D.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
