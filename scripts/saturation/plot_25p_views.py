#!/usr/bin/env python3
"""25-param model: render the two summary views.
  A: 3-panel data/model/residual heatmap (aggregated over 18 detectors)
  B: 6-panel breakdown + a_det bar chart

Both use n-weighted aggregate over detectors so apples-to-apples with 8p plots.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

NPZ = "n_below_study/v5_npz/C_det_mlat_t_heatmap.npz"
PARAMS = json.loads(Path("/tmp/per_det_25param.json").read_text())


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def per_det_model(mlat_centers, t_years):
    a_det = np.array(PARAMS["a_det"])             # (18,)
    alpha = PARAMS["alpha"]; mu_m = PARAMS["mu_m"]; k_m = PARAMS["k_m"]
    amp0 = PARAMS["amp0"]; mu_t = PARAMS["mu_t"]; k_t = PARAMS["k_t"]
    C0 = PARAMS["C0"]
    sm = sigm((mlat_centers - mu_m) / k_m)
    st = sigm((t_years - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return (a_det[:, None, None] * g[None, :, None]
            * (1.0 - amp0 * g[None, :, None] * st[None, None, :]) + C0)


def main():
    z = np.load(NPZ)
    C_per_det = z["C_mean"]    # (18, 60, 108)
    n_per_det = z["n"]
    months = z["months"]; edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t0 = np.datetime64("2017-06-22")
    t_years = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25
    mask_per_det = n_per_det > 50

    # ─── Aggregate over detectors (n-weighted) ───
    C_model_per_det = per_det_model(mlat_centers, t_years)
    n_total = n_per_det.sum(axis=0)                                  # (60, 108)
    mask_agg = n_total > 500
    with np.errstate(invalid='ignore', divide='ignore'):
        C_data_agg = np.where(mask_per_det, n_per_det * C_per_det, 0.0).sum(axis=0)
        C_data_agg = np.where(mask_agg, C_data_agg / n_total, np.nan)
        C_model_agg = np.where(mask_per_det, n_per_det * C_model_per_det, 0.0).sum(axis=0)
        C_model_agg = np.where(mask_agg, C_model_agg / n_total, np.nan)

    resid = C_data_agg - C_model_agg
    valid_resid = resid[mask_agg]
    print(f"Aggregate heatmap (mlat, t) — residual std: {np.std(valid_resid[np.isfinite(valid_resid)]):.2f} cnt/s")

    a_det = np.array(PARAMS["a_det"])
    alpha = PARAMS["alpha"]; mu_m = PARAMS["mu_m"]; k_m = PARAMS["k_m"]
    amp0 = PARAMS["amp0"]; mu_t = PARAMS["mu_t"]; k_t = PARAMS["k_t"]
    C0 = PARAMS["C0"]

    # =====================================================================
    # VIEW A: 3-panel heatmap
    # =====================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "25-param per-det model:  C(det,mlat,t) = a_det·g·[1 − amp₀·g·σ_t] + C₀, "
        "g = 1 + α·σ_m\n"
        f"(18 a_det, shared: α={alpha:.2f}, μ_m={mu_m:.1f}°, k_m={k_m:.1f}°, "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C0:+.0f}) "
        f"— aggregated (n-weighted) over 18 detectors",
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

    plot_pcm(axes[0], C_data_agg, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA — mean C aggregated over detectors')
    plot_pcm(axes[1], C_model_agg, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — 25p aggregate prediction')
    plot_pcm(axes[2], resid, -30, 30, 'RdBu_r',
             'data − model (cnt/s)', '3. RESIDUAL — data − model (±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out_A = "plots/fit_25p_2D.png"
    plt.savefig(out_A, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out_A}")

    # =====================================================================
    # VIEW B: 6-panel breakdown + a_det bar
    # =====================================================================
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.40, wspace=0.25,
                          height_ratios=[1, 1, 1, 0.7])
    fig.suptitle(
        f"25-param model breakdown:  C(det,mlat,t) = a_det·g·[1 − amp₀·g·σ_t] + C₀, g = 1 + α·σ_m\n"
        f"shared: α={alpha:.2f}, μ_m={mu_m:.1f}°, k_m={k_m:.1f}°, "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C0:+.0f};  "
        f"a_det range: {a_det.min():.0f}–{a_det.max():.0f} (median {np.median(a_det):.0f})",
        fontsize=12, fontweight='bold')

    # Panel 1: g(mlat), g²(mlat)
    ax = fig.add_subplot(gs[0, 0])
    m_fine = np.linspace(0, 60, 800)
    sm_fine = sigm((m_fine - mu_m) / k_m)
    g_fine = 1.0 + alpha * sm_fine
    ax.plot(m_fine, g_fine, '-', lw=2.5, color='C0', label='g(mlat) — baseline')
    ax.plot(m_fine, g_fine**2, '-', lw=2.5, color='C3', label='g²(mlat) — decay weight')
    ax.axvline(mu_m, ls='--', color='gray', alpha=0.5, label=f'μ_m={mu_m:.1f}°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("dimensionless", fontsize=11)
    ax.set_title(f"1. mlat shapes: g(0)=1, g(60)={g_fine[-1]:.2f}, g²(60)={g_fine[-1]**2:.2f}",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3)

    # Panel 2: σ_t(t)
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
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3: C(t) at fixed mlat — 12 slices, dots=data agg, line=model agg
    ax = fig.add_subplot(gs[1, 0])
    mlat_picks = [3, 8, 14, 20, 25, 30, 35, 40, 45, 50, 55, 57]
    mi = [int(np.argmin(np.abs(mlat_centers - m))) for m in mlat_picks]
    cmap = plt.cm.plasma
    for k, (mp, ii) in enumerate(zip(mlat_picks, mi)):
        color = cmap(k / max(len(mlat_picks)-1, 1))
        y_data = C_data_agg[ii, :].copy()
        y_data[~mask_agg[ii, :]] = np.nan
        y_model = C_model_agg[ii, :].copy()
        y_model[~mask_agg[ii, :]] = np.nan
        ax.plot(month_dt, y_data, '.', ms=3, color=color, alpha=0.5)
        ax.plot(month_dt, y_model, '-', lw=1.6, color=color, label=f"{mp}°")
    ax.set_xlabel("date", fontsize=11); ax.set_ylabel("C (cnt/s)", fontsize=11)
    ax.set_title(f"3. C(t) at fixed |mlat| ({len(mlat_picks)} slices)  — aggregate",
                 fontsize=11)
    ax.legend(fontsize=8, loc='upper right', ncol=3, title='|mlat|', columnspacing=0.7)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 4: C(|mlat|) at fixed dates — 18 slices
    ax = fig.add_subplot(gs[1, 1])
    date_picks = ["2017-09", "2018-03", "2018-09", "2019-03", "2019-09",
                  "2020-03", "2020-09", "2021-03", "2021-09",
                  "2022-03", "2022-09", "2023-03", "2023-09",
                  "2024-03", "2024-09", "2025-03", "2025-09", "2026-03"]
    pick_idx_t = [list(months).index(m) for m in date_picks if m in months]
    cmap = plt.cm.viridis
    # Compute aggregate model on fine mlat grid for smooth lines.
    # Since shape g(mlat) and σ_t(t) are shared, aggregate model is <a>·g·... with <a> = n-weighted.
    sm_fine = sigm((m_fine - mu_m) / k_m)
    g_fine_ = 1.0 + alpha * sm_fine
    for k, ti in enumerate(pick_idx_t):
        color = cmap(k / max(len(pick_idx_t)-1, 1))
        y_data = C_data_agg[:, ti].copy()
        y_data[~mask_agg[:, ti]] = np.nan
        ax.plot(mlat_centers, y_data, '.', ms=3, color=color, alpha=0.5)
        # n-weighted <a> at this time slice
        w = n_per_det[:, :, ti].astype(float)
        with np.errstate(invalid='ignore'):
            w_total_mlat = w.sum(axis=0)
            a_eff_mlat = (w * a_det[:, None]).sum(axis=0) / np.where(w_total_mlat>0, w_total_mlat, 1)
        # Use overall n-weighted a for fine line
        a_overall = (w * a_det[:, None]).sum() / w.sum()
        st_val = sigm((t_years[ti] - mu_t) / k_t)
        C_line = a_overall * g_fine_ * (1 - amp0 * g_fine_ * st_val) + C0
        ax.plot(m_fine, C_line, '-', lw=1.6, color=color, label=months[ti])
    ax.axvline(mu_m, ls='--', color='gray', alpha=0.4)
    ax.set_xlabel("|mlat| (deg)", fontsize=11); ax.set_ylabel("C (cnt/s)", fontsize=11)
    ax.set_title(f"4. C(|mlat|) at fixed dates ({len(date_picks)} slices) — aggregate",
                 fontsize=11)
    ax.legend(fontsize=7, loc='upper left', ncol=3, title='date', columnspacing=0.6)
    ax.grid(alpha=0.3)

    # Panel 5: marginal C(mlat)
    ax = fig.add_subplot(gs[2, 0])
    with np.errstate(invalid='ignore'):
        C_mlat_d = np.nanmean(np.where(mask_agg, C_data_agg, np.nan), axis=1)
        C_mlat_m = np.nanmean(np.where(mask_agg, C_model_agg, np.nan), axis=1)
    ax.plot(mlat_centers, C_mlat_d, 'o-', lw=1.5, ms=4, color='black', label='data (mean over t)')
    ax.plot(mlat_centers, C_mlat_m, '-', lw=2.5, color='C3', label='25p aggregate model')
    ax.set_xlabel("|mlat| (deg)"); ax.set_ylabel("⟨C⟩_t (cnt/s)")
    ax.set_title("5. Marginal C(|mlat|): time-averaged data vs model")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Panel 6: marginal C(t)
    ax = fig.add_subplot(gs[2, 1])
    with np.errstate(invalid='ignore'):
        C_t_d = np.nanmean(np.where(mask_agg, C_data_agg, np.nan), axis=0)
        C_t_m = np.nanmean(np.where(mask_agg, C_model_agg, np.nan), axis=0)
    ax.plot(month_dt, C_t_d, 'o-', lw=1.5, ms=4, color='black', label='data (mean over mlat)')
    ax.plot(month_dt, C_t_m, '-', lw=2.5, color='C3', label='25p aggregate model')
    ax.set_xlabel("date"); ax.set_ylabel("⟨C⟩_mlat (cnt/s)")
    ax.set_title("6. Marginal C(t): mlat-averaged data vs model")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 7: a_det bar chart — the unique 25p signal
    ax = fig.add_subplot(gs[3, :])
    labels = [f"{'ABC'[i//6]}{i%6}" for i in range(18)]
    box_colors = ['#1f77b4']*6 + ['#2ca02c']*6 + ['#d62728']*6
    bars = ax.bar(labels, a_det, color=box_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(a_det.mean(), ls='--', color='gray', alpha=0.5,
               label=f'mean = {a_det.mean():.1f}')
    ax.axhline(202.6, ls=':', color='C1', alpha=0.7,
               label='8p single a = 202.6')
    for b, v in zip(bars, a_det):
        ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}',
                ha='center', fontsize=8)
    ax.set_ylabel("a_det (cnt/s)", fontsize=11)
    ax.set_title("7. Per-detector amplitudes — the only 25p-vs-8p visual difference",
                 fontsize=11)
    ax.set_ylim(140, max(a_det)*1.05)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, axis='y')

    out_B = "plots/fit_25p_breakdown.png"
    plt.savefig(out_B, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out_B}")


if __name__ == "__main__":
    main()
