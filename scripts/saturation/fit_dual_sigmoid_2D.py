#!/usr/bin/env python3
"""11-param dual-mlat-sigmoid model: refit + produce 3-panel heatmap +
6-panel breakdown (matching the styles of fit_mlat_dep_amp_2D.py and
plot_mlat_dep_amp_breakdown.py).

Model:
  g(|mlat|) = 1 + α₁·σ_m₁(|mlat|) + α₂·σ_m₂(|mlat|)
  C(mlat, t) = a·g · [1 − amp₀·g·σ_t(t)] + C₀

  σ_m_i(m) = 1/(1+exp(-(m-μ_i)/k_i))   i ∈ {1, 2}
  σ_t(t)   = 1/(1+exp(-(t-μ_t)/k_t))

11 globals: a, α₁, μ_1, k_1, α₂, μ_2, k_2, amp₀, μ_t, k_t, C₀
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


def model_2d(params, mlat, t):
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C_0 = params
    s1 = sigm((mlat - mu1) / k1)
    s2 = sigm((mlat - mu2) / k2)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha1 * s1 + alpha2 * s2
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def g_only(params, mlat):
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C_0 = params
    s1 = sigm((mlat - mu1) / k1)
    s2 = sigm((mlat - mu2) / k2)
    return 1.0 + alpha1 * s1 + alpha2 * s2


def fit_main(C_data, mask, mlat_centers, t_years):
    C_clean = np.where(mask, C_data, 0.0)
    def residual(p):
        return ((model_2d(p, mlat_centers, t_years) - C_clean) * mask).ravel()

    # Initial guess from round 2 dual_mlat_sigmoid winner; α₂ negative for low-mlat dip
    p0 = [200.0, 1.7, 44.0, 6.0, -0.07, 14.5, 1.3, 0.15, 5.2, 1.0, -80.0]
    lo = [10.0, 0.1, 30.0, 0.5, -2.0,   5.0,  0.1, 0.0, 0.5, 0.2, -300.0]
    hi = [500.0, 10.0, 60.0, 15.0, 2.0, 30.0, 10.0, 1.0, 8.0, 8.0, +300.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    return res.x, res


def plot_heatmap_3panel(params, C_data, mask, mlat_centers, t_years,
                        edges, month_dt, out_path):
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C_0 = params
    C_pred = model_2d(params, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    resid_m = resid.copy(); resid_m[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "Dual-σ 11-param fit:  "
        "C = a·g·[1 − amp₀·g·σ_t] + C₀,   g = 1 + α₁·σ_m1 + α₂·σ_m2\n"
        f"a={a:.0f}, α₁={alpha1:.2f}, μ_1={mu1:.1f}°, k_1={k1:.1f}°, "
        f"α₂={alpha2:+.3f}, μ_2={mu2:.1f}°, k_2={k2:.2f}°,  "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C_0:+.0f}",
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
             'C_model (cnt/s)', '2. MODEL — dual-σ (11 params)')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data − model (cnt/s)', '3. RESIDUAL — data − model (±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out_path}")


def plot_breakdown_6panel(params, C_data, n_data, mask,
                          mlat_centers, t_years, months, month_dt, out_path):
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C_0 = params
    t0 = np.datetime64("2017-06-22")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    fig.suptitle(
        "Dual-σ 11-param model breakdown:  "
        "C = a·g·[1 − amp₀·g·σ_t] + C₀,   g = 1 + α₁·σ_m1 + α₂·σ_m2\n"
        f"a={a:.0f}, α₁={alpha1:.2f}, μ_1={mu1:.1f}°, k_1={k1:.1f}°, "
        f"α₂={alpha2:+.3f}, μ_2={mu2:.1f}°, k_2={k2:.2f}°,  "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C_0:+.0f}",
        fontsize=11, fontweight='bold')

    # ─── Panel 1: g(mlat) and g²(mlat) ───
    ax = fig.add_subplot(gs[0, 0])
    m_fine = np.linspace(0, 60, 800)
    g_fine = g_only(params, m_fine)
    s1_fine = sigm((m_fine - mu1) / k1)
    s2_fine = sigm((m_fine - mu2) / k2)
    ax.plot(m_fine, g_fine, '-', lw=2.5, color='C0', label='g(mlat)')
    ax.plot(m_fine, g_fine**2, '-', lw=2.5, color='C3', label='g²(mlat) — decay weight')
    ax.plot(m_fine, 1 + alpha1*s1_fine, ':', lw=1.5, color='gray',
            alpha=0.7, label='1 + α₁·σ_m1 (high)')
    ax.plot(m_fine, 1 + alpha2*s2_fine, '--', lw=1.5, color='C2',
            alpha=0.7, label='1 + α₂·σ_m2 (low-mlat dip)')
    ax.axvline(mu1, ls='--', color='gray', alpha=0.4, lw=1)
    ax.axvline(mu2, ls='--', color='C2', alpha=0.4, lw=1)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("dimensionless", fontsize=11)
    ax.set_title(f"1. mlat shape decomposition:  "
                 f"g(0)={g_fine[0]:.2f}, g(60)={g_fine[-1]:.2f}",
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel 2: σ_t(t) ───
    ax = fig.add_subplot(gs[0, 1])
    t_fine = np.linspace(0, 9, 400)
    st_fine = sigm((t_fine - mu_t) / k_t)
    dt_fine = np.array([t0 + np.timedelta64(int(tt*365.25), 'D') for tt in t_fine])
    ax.plot(dt_fine, st_fine, '-', lw=2.5, color='C2')
    ax.axvline(t0 + np.timedelta64(int(mu_t*365.25), 'D'), ls='--', color='gray',
               alpha=0.5, label=f'μ_t={mu_t:.2f}yr')
    ax.axhline(0.5, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("σ_t(t)", fontsize=11)
    ax.set_title(f"2. Time decay σ_t(t):  k_t={k_t:.2f} yr (10%→90% in {4*k_t:.1f} yr)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ─── Panel 3: C(t) at fixed |mlat| ───
    ax = fig.add_subplot(gs[1, 0])
    mlat_picks = [3, 20, 40, 50, 57]
    cmap = plt.cm.plasma
    mlat_idx = [int(np.argmin(np.abs(mlat_centers - m))) for m in mlat_picks]
    C_model = model_2d(params, mlat_centers, t_years)
    for k, (mp, mi) in enumerate(zip(mlat_picks, mlat_idx)):
        color = cmap(k / max(len(mlat_picks)-1, 1))
        y_data = C_data[mi, :].copy()
        y_data[n_data[mi, :] < 200] = np.nan
        ax.plot(month_dt, y_data, '.', ms=3, color=color, alpha=0.5)
        ax.plot(month_dt, C_model[mi, :], '-', lw=2, color=color,
                label=f"|mlat|≈{mp}°")
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("C (cnt/s)", fontsize=11)
    ax.set_title("3. C(t) at fixed |mlat|  (dots=data, lines=model)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ─── Panel 4: C(|mlat|) at fixed dates ───
    ax = fig.add_subplot(gs[1, 1])
    date_picks = ["2017-09", "2019-06", "2021-06", "2023-06", "2025-06"]
    pick_idx_t = [list(months).index(m) for m in date_picks if m in months]
    cmap = plt.cm.viridis
    for k, ti in enumerate(pick_idx_t):
        color = cmap(k / max(len(pick_idx_t)-1, 1))
        y_data = C_data[:, ti].copy()
        y_data[n_data[:, ti] < 200] = np.nan
        ax.plot(mlat_centers, y_data, '.', ms=3, color=color, alpha=0.5)
        # Fine-grid model line
        g_f = g_only(params, m_fine)
        st_val = sigm((t_years[ti] - mu_t) / k_t)
        C_line = a * g_f * (1 - amp0 * g_f * st_val) + C_0
        ax.plot(m_fine, C_line, '-', lw=2, color=color,
                label=date_picks[k])
    ax.axvline(mu1, ls='--', color='gray', alpha=0.4)
    ax.axvline(mu2, ls='--', color='C2', alpha=0.4, lw=1)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("C (cnt/s)", fontsize=11)
    ax.set_title("4. C(|mlat|) at fixed dates  (dots=data, lines=model)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel 5: marginal C(|mlat|) ───
    ax = fig.add_subplot(gs[2, 0])
    C_model_full = model_2d(params, mlat_centers, t_years)
    Cm = np.where(mask, C_data, np.nan)
    Cmm = np.where(mask, C_model_full, np.nan)
    with np.errstate(invalid='ignore'):
        C_mlat_data = np.nanmean(Cm, axis=1)
        C_mlat_model = np.nanmean(Cmm, axis=1)
    ax.plot(mlat_centers, C_mlat_data, 'o-', lw=1.5, ms=4, color='black',
            label='data (mean over t)')
    ax.plot(mlat_centers, C_mlat_model, '-', lw=2.5, color='C3', label='model')
    ax.axvline(mu2, ls='--', color='C2', alpha=0.5, lw=1, label=f'μ_2={mu2:.1f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("5. Marginal C(|mlat|): time-averaged data vs model",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel 6: marginal C(t) ───
    ax = fig.add_subplot(gs[2, 1])
    with np.errstate(invalid='ignore'):
        C_t_data = np.nanmean(Cm, axis=0)
        C_t_model = np.nanmean(Cmm, axis=0)
    ax.plot(month_dt, C_t_data, 'o-', lw=1.5, ms=4, color='black',
            label='data (mean over mlat)')
    ax.plot(month_dt, C_t_model, '-', lw=2.5, color='C3', label='model')
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("⟨C⟩_mlat (cnt/s)", fontsize=11)
    ax.set_title("6. Marginal C(t): mlat-averaged data vs model",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out_path}")


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

    params, res = fit_main(C_data, mask, mlat_centers, t_years)
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C_0 = params

    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    print("\n=== Fitted parameters (11-param dual-σ) ===")
    print(f"  a      = {a:8.2f}")
    print(f"  α₁     = {alpha1:8.3f}   (high-mlat sigmoid scale)")
    print(f"  μ_1    = {mu1:8.2f}°    (high-mlat inflection)")
    print(f"  k_1    = {k1:8.2f}°    (high-mlat width)")
    print(f"  α₂     = {alpha2:+8.4f}  (low-mlat dip amplitude)")
    print(f"  μ_2    = {mu2:8.2f}°    (low-mlat dip center)")
    print(f"  k_2    = {k2:8.3f}°    (low-mlat dip width)")
    print(f"  amp₀   = {amp0:8.3f}")
    print(f"  μ_t    = {mu_t:8.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t    = {k_t:8.2f} yr")
    print(f"  C_0    = {C_0:+8.2f}")

    C_pred = model_2d(params, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid = resid[mask]
    print(f"\n=== Residual stats ===")
    print(f"  std:    {np.std(valid):.3f} cnt/s")
    print(f"  RMS/mean: {np.sqrt(np.mean(valid**2))/np.mean(C_data[mask])*100:.2f}%")
    print(f"  max|r|: {np.max(np.abs(valid)):.1f}")

    Path("plots").mkdir(exist_ok=True)
    plot_heatmap_3panel(params, C_data, mask, mlat_centers, t_years,
                        edges, month_dt, "plots/fit_dual_sigmoid_2D.png")
    plot_breakdown_6panel(params, C_data, n_data, mask, mlat_centers, t_years,
                          months, month_dt, "plots/fit_dual_sigmoid_breakdown.png")


if __name__ == "__main__":
    main()
