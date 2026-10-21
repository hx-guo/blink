#!/usr/bin/env python3
"""Visualize the 8-param α=β model — break it into pieces.

  C(mlat, t) = a·g − a·amp₀·g²·σ_t + C₀
  g(mlat) = 1 + α·σ_m(|mlat|)

Panels:
  1. g(mlat) and g²(mlat) — mlat shape (baseline) vs decay weight
  2. σ_t(t) — time decay function
  3. C(t) curves at fixed |mlat| (mlat-dependent decay rate)
  4. C(|mlat|) curves at fixed dates (shape evolution)
  5. data vs model: time-mean profile C̄(|mlat|) + space-mean profile C̄(t)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"

# 8-param fit
a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = (
    202.60, 1.695, 44.46, 6.33, 0.152, 5.25, 1.00, -79.26)


def sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


def model_2d(mlat, t):
    sm = sigm((mlat - mu_m) / k_m)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


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

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    fig.suptitle(
        "8-param α=β model breakdown:  "
        "C = a·g − a·amp₀·g²·σ_t + C₀,  g = 1 + α·σ_m\n"
        f"a={a:.0f}, α={alpha:.2f}, μ_m={mu_m:.1f}°, k_m={k_m:.1f}°, "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C_0:+.0f}",
        fontsize=12, fontweight='bold')

    # ─── Panel 1: g(mlat) and g²(mlat) ───
    ax = fig.add_subplot(gs[0, 0])
    m_fine = np.linspace(0, 60, 400)
    sm_fine = sigm((m_fine - mu_m) / k_m)
    g_fine = 1.0 + alpha * sm_fine
    ax.plot(m_fine, g_fine, '-', lw=2.5, color='C0', label='g(mlat) — baseline')
    ax.plot(m_fine, g_fine**2, '-', lw=2.5, color='C3', label='g²(mlat) — decay weight')
    ax.axvline(mu_m, ls='--', color='gray', alpha=0.5, label=f'μ_m={mu_m:.1f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("dimensionless", fontsize=11)
    ax.set_title(f"1. mlat shapes:  g(0)=1, g(60)={g_fine[-1]:.2f},  g²(0)=1, g²(60)={g_fine[-1]**2:.2f}",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Panel 2: σ_t(t) ───
    ax = fig.add_subplot(gs[0, 1])
    t_fine = np.linspace(0, 9, 400)
    st_fine = sigm((t_fine - mu_t) / k_t)
    dt_fine = np.array([t0 + np.timedelta64(int(tt*365.25), 'D') for tt in t_fine])
    ax.plot(dt_fine, st_fine, '-', lw=2.5, color='C2')
    ax.axvline(t0 + np.timedelta64(int(mu_t*365.25), 'D'), ls='--', color='gray',
               alpha=0.5, label=f'μ_t={mu_t:.2f}yr (2022-09)')
    ax.axhline(0.5, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("σ_t(t)", fontsize=11)
    ax.set_title(f"2. Time decay σ_t(t):  width k_t={k_t:.2f} yr (10%→90% in {4*k_t:.1f} yr)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ─── Panel 3: C(t) at fixed |mlat| — denser sampling ───
    ax = fig.add_subplot(gs[1, 0])
    mlat_picks = [3, 8, 14, 20, 25, 30, 35, 40, 45, 50, 55, 57]
    cmap = plt.cm.plasma
    mlat_idx_data = [int(np.argmin(np.abs(mlat_centers - m))) for m in mlat_picks]
    C_model = model_2d(mlat_centers, t_years)
    for k, (m_pick, mi) in enumerate(zip(mlat_picks, mlat_idx_data)):
        color = cmap(k / max(len(mlat_picks)-1, 1))
        y_data = C_data[mi, :].copy()
        y_data[n_data[mi, :] < 200] = np.nan
        ax.plot(month_dt, y_data, '.', ms=3, color=color, alpha=0.5)
        ax.plot(month_dt, C_model[mi, :], '-', lw=1.6, color=color,
                label=f"{m_pick}°")
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("C (cnt/s)", fontsize=11)
    ax.set_title(f"3. C(t) at fixed |mlat| ({len(mlat_picks)} slices)  — dots=data, lines=model",
                 fontsize=11)
    ax.legend(fontsize=8, loc='upper right', ncol=3, title='|mlat|', columnspacing=0.7)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ─── Panel 4: C(|mlat|) at fixed dates — denser sampling ───
    ax = fig.add_subplot(gs[1, 1])
    date_picks = ["2017-09", "2018-03", "2018-09", "2019-03", "2019-09",
                  "2020-03", "2020-09", "2021-03", "2021-09",
                  "2022-03", "2022-09", "2023-03", "2023-09",
                  "2024-03", "2024-09", "2025-03", "2025-09", "2026-03"]
    pick_idx_t = [list(months).index(m) for m in date_picks if m in months]
    cmap = plt.cm.viridis
    for k, ti in enumerate(pick_idx_t):
        color = cmap(k / max(len(pick_idx_t)-1, 1))
        y_data = C_data[:, ti].copy()
        y_data[n_data[:, ti] < 200] = np.nan
        ax.plot(mlat_centers, y_data, '.', ms=3, color=color, alpha=0.5)
        sm_fine = sigm((m_fine - mu_m) / k_m)
        g_fine_ = 1.0 + alpha * sm_fine
        st_val = sigm((t_years[ti] - mu_t) / k_t)
        C_line = a * g_fine_ * (1 - amp0 * g_fine_ * st_val) + C_0
        ax.plot(m_fine, C_line, '-', lw=1.6, color=color,
                label=months[ti])
    ax.axvline(mu_m, ls='--', color='gray', alpha=0.4)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("C (cnt/s)", fontsize=11)
    ax.set_title(f"4. C(|mlat|) at fixed dates ({len(date_picks)} slices) — dots=data, lines=model",
                 fontsize=11)
    ax.legend(fontsize=7, loc='upper left', ncol=3, title='date', columnspacing=0.6)
    ax.grid(alpha=0.3)

    # ─── Panel 5: marginal profiles (data vs model) ───
    ax = fig.add_subplot(gs[2, 0])
    C_model_full = model_2d(mlat_centers, t_years)
    # Time-mean C(mlat)
    Cm = np.where(mask, C_data, np.nan)
    Cmm = np.where(mask, C_model_full, np.nan)
    with np.errstate(invalid='ignore'):
        C_mlat_data = np.nanmean(Cm, axis=1)
        C_mlat_model = np.nanmean(Cmm, axis=1)
    ax.plot(mlat_centers, C_mlat_data, 'o-', lw=1.5, ms=4, color='black', label='data (mean over t)')
    ax.plot(mlat_centers, C_mlat_model, '-', lw=2.5, color='C3', label='model')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("⟨C⟩_t (cnt/s)", fontsize=11)
    ax.set_title("5. Marginal C(|mlat|): time-averaged data vs model",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    with np.errstate(invalid='ignore'):
        C_t_data = np.nanmean(Cm, axis=0)
        C_t_model = np.nanmean(Cmm, axis=0)
    ax.plot(month_dt, C_t_data, 'o-', lw=1.5, ms=4, color='black', label='data (mean over mlat)')
    ax.plot(month_dt, C_t_model, '-', lw=2.5, color='C3', label='model')
    ax.set_xlabel("date", fontsize=11)
    ax.set_ylabel("⟨C⟩_mlat (cnt/s)", fontsize=11)
    ax.set_title("6. Marginal C(t): mlat-averaged data vs model",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    Path("plots").mkdir(exist_ok=True)
    out = "plots/mlat_dep_amp_breakdown.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
