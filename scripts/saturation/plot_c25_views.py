#!/usr/bin/env python3
"""C25 model: render the two summary views with paper-grade LaTeX symbols.
  A: 3-panel data/model/residual heatmap (aggregated over 18 detectors)
  B: 7-panel breakdown + A_i bar chart

Symbol convention (paper-grade):
  S(x; mu, w) = 1/(1+exp(-(x-mu)/w))         sigmoid
  g(|m|) = 1 + alpha_m * S(|m|; mu_m, w_m)   mlat shape
  C(i, |m|, t) = A_i * g * [1 - alpha_t * g * S(t; mu_t, w_t)] + C_0

JSON field name mapping (kept for script compatibility):
  alpha → alpha_m,  amp0 → alpha_t,  k_m → w_m,  k_t → w_t
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

NPZ = "n_below_study/v5_npz/C_det_mlat_t_heatmap.npz"
PARAMS = json.loads(Path("/tmp/per_det_25param.json").read_text())


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def per_det_model(mlat_centers, t_years):
    A_i = np.array(PARAMS["a_det"])
    alpha_m, mu_m, w_m = PARAMS["alpha"], PARAMS["mu_m"], PARAMS["k_m"]
    alpha_t, mu_t, w_t = PARAMS["amp0"], PARAMS["mu_t"], PARAMS["k_t"]
    C_0 = PARAMS["C0"]
    sm = sigm((mlat_centers - mu_m) / w_m)
    st = sigm((t_years - mu_t) / w_t)
    g = 1.0 + alpha_m * sm
    return (A_i[:, None, None] * g[None, :, None]
            * (1.0 - alpha_t * g[None, :, None] * st[None, None, :]) + C_0)


def header_block():
    """Master equation header used by both views."""
    A_i = np.array(PARAMS["a_det"])
    alpha_m = PARAMS["alpha"]; mu_m = PARAMS["mu_m"]; w_m = PARAMS["k_m"]
    alpha_t = PARAMS["amp0"]; mu_t = PARAMS["mu_t"]; w_t = PARAMS["k_t"]
    C_0 = PARAMS["C0"]
    eq = (
        r"$S(x;\mu,w)\,=\,\dfrac{1}{1+e^{-(x-\mu)/w}}$ \quad\quad "
        r"$g(|m|)\,=\,1+\alpha_m\,S(|m|;\mu_m,w_m)$"
        "\n"
        r"$C(i,|m|,t)\,=\,A_i\cdot g(|m|)\cdot"
        r"\left[\,1-\alpha_t\,g(|m|)\,S(t;\mu_t,w_t)\,\right]+C_0$"
        "\n"
        fr"$\alpha_m={alpha_m:.2f},\ \mu_m={mu_m:.1f}^\circ,\ w_m={w_m:.1f}^\circ,\ "
        fr"\alpha_t={alpha_t:.3f},\ \mu_t={mu_t:.2f}\,\mathrm{{yr}},\ "
        fr"w_t={w_t:.2f}\,\mathrm{{yr}},\ C_0={C_0:+.0f};\ "
        fr"\{{A_i\}}\in[{A_i.min():.0f},\,{A_i.max():.0f}]$"
    )
    return eq


def main():
    z = np.load(NPZ)
    C_per_det = z["C_mean"]; n_per_det = z["n"]
    months = z["months"]; edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t0 = np.datetime64("2017-06-22")
    t_years = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25
    mask_per_det = n_per_det > 50

    # Aggregate over detectors
    C_model_per_det = per_det_model(mlat_centers, t_years)
    n_total = n_per_det.sum(axis=0)
    mask_agg = n_total > 500
    with np.errstate(invalid='ignore', divide='ignore'):
        C_data_agg = np.where(mask_per_det, n_per_det * C_per_det, 0.0).sum(axis=0)
        C_data_agg = np.where(mask_agg, C_data_agg / n_total, np.nan)
        C_model_agg = np.where(mask_per_det, n_per_det * C_model_per_det, 0.0).sum(axis=0)
        C_model_agg = np.where(mask_agg, C_model_agg / n_total, np.nan)
    resid = C_data_agg - C_model_agg

    A_i = np.array(PARAMS["a_det"])
    alpha_m = PARAMS["alpha"]; mu_m = PARAMS["mu_m"]; w_m = PARAMS["k_m"]
    alpha_t = PARAMS["amp0"]; mu_t = PARAMS["mu_t"]; w_t = PARAMS["k_t"]
    C_0 = PARAMS["C0"]

    # =================================================================
    # VIEW A: 3-panel heatmap
    # =================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("C25 model  —  aggregated ($n_i$-weighted) over 18 detectors\n" + header_block(),
                 fontsize=13, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1])/2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data,
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel(r"$|m|$ (deg)", fontsize=12)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_agg, 0, 400, 'viridis',
             r"$C_\mathrm{data}$ (cnt/s)",
             r"1.\ DATA  $-$  mean $C$ aggregated over detectors")
    plot_pcm(axes[1], C_model_agg, 0, 400, 'viridis',
             r"$C_\mathrm{model}$ (cnt/s)",
             r"2.\ MODEL  $-$  C25 aggregate prediction")
    plot_pcm(axes[2], resid, -30, 30, 'RdBu_r',
             r"data $-$ model (cnt/s)",
             r"3.\ RESIDUAL  $-$  data $-$ model  ($\pm 30$ cnt/s)")
    axes[2].set_xlabel("date", fontsize=12)
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out_A = "plots/c25_heatmap.png"
    plt.savefig(out_A, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out_A}")

    # =================================================================
    # VIEW B: 7-panel breakdown
    # =================================================================
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25,
                          height_ratios=[1, 1, 1, 0.7])
    fig.suptitle("C25 model breakdown\n" + header_block(),
                 fontsize=12, fontweight='bold')

    # Panel 1: g(|m|), g^2(|m|)
    ax = fig.add_subplot(gs[0, 0])
    m_fine = np.linspace(0, 60, 800)
    sm_fine = sigm((m_fine - mu_m) / w_m)
    g_fine = 1.0 + alpha_m * sm_fine
    ax.plot(m_fine, g_fine, '-', lw=2.5, color='C0',
            label=r"$g(|m|)=1+\alpha_m\,S(|m|;\mu_m,w_m)$")
    ax.plot(m_fine, g_fine**2, '-', lw=2.5, color='C3',
            label=r"$g^2(|m|)$  (decay weight)")
    ax.axvline(mu_m, ls='--', color='gray', alpha=0.5,
               label=fr"$\mu_m={mu_m:.1f}^\circ$")
    ax.set_xlabel(r"$|m|$ (deg)", fontsize=12)
    ax.set_ylabel("dimensionless", fontsize=12)
    ax.set_title(fr"1.\ mlat shapes:  $g(0)=1,\ g(60^\circ)={g_fine[-1]:.2f},"
                 fr"\ g^2(60^\circ)={g_fine[-1]**2:.2f}$",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3)

    # Panel 2: S(t; mu_t, w_t)
    ax = fig.add_subplot(gs[0, 1])
    t_fine = np.linspace(0, 9, 400)
    st_fine = sigm((t_fine - mu_t) / w_t)
    dt_fine = np.array([t0 + np.timedelta64(int(tt*365.25), 'D') for tt in t_fine])
    ax.plot(dt_fine, st_fine, '-', lw=2.5, color='C2',
            label=r"$S(t;\mu_t,w_t)$")
    ax.axvline(t0 + np.timedelta64(int(mu_t*365.25), 'D'), ls='--', color='gray',
               alpha=0.5, label=fr"$\mu_t={mu_t:.2f}\,\mathrm{{yr}}$")
    ax.axhline(0.5, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel("date", fontsize=12)
    ax.set_ylabel(r"$S(t;\mu_t,w_t)$", fontsize=12)
    ax.set_title(fr"2.\ Time sigmoid:  $w_t={w_t:.2f}\,\mathrm{{yr}}$  "
                 fr"(10\%$\to$90\% over $\sim{4*w_t:.1f}\,\mathrm{{yr}}$)",
                 fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3: C(t) at fixed |m|
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
        ax.plot(month_dt, y_model, '-', lw=1.6, color=color, label=f"${mp}^\\circ$")
    ax.set_xlabel("date", fontsize=12); ax.set_ylabel(r"$C$ (cnt/s)", fontsize=12)
    ax.set_title(fr"3.\ $C(t)$ at fixed $|m|$ ({len(mlat_picks)} slices)  "
                 r"$-$  aggregate", fontsize=11)
    ax.legend(fontsize=8, loc='upper right', ncol=3, title=r"$|m|$", columnspacing=0.7)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 4: C(|m|) at fixed dates
    ax = fig.add_subplot(gs[1, 1])
    date_picks = ["2017-09", "2018-03", "2018-09", "2019-03", "2019-09",
                  "2020-03", "2020-09", "2021-03", "2021-09",
                  "2022-03", "2022-09", "2023-03", "2023-09",
                  "2024-03", "2024-09", "2025-03", "2025-09", "2026-03"]
    pick_idx_t = [list(months).index(m) for m in date_picks if m in months]
    cmap = plt.cm.viridis
    sm_fine = sigm((m_fine - mu_m) / w_m)
    g_fine_ = 1.0 + alpha_m * sm_fine
    for k, ti in enumerate(pick_idx_t):
        color = cmap(k / max(len(pick_idx_t)-1, 1))
        y_data = C_data_agg[:, ti].copy()
        y_data[~mask_agg[:, ti]] = np.nan
        ax.plot(mlat_centers, y_data, '.', ms=3, color=color, alpha=0.5)
        w = n_per_det[:, :, ti].astype(float)
        a_overall = (w * A_i[:, None]).sum() / max(w.sum(), 1)
        st_val = sigm((t_years[ti] - mu_t) / w_t)
        C_line = a_overall * g_fine_ * (1 - alpha_t * g_fine_ * st_val) + C_0
        ax.plot(m_fine, C_line, '-', lw=1.6, color=color, label=months[ti])
    ax.axvline(mu_m, ls='--', color='gray', alpha=0.4)
    ax.set_xlabel(r"$|m|$ (deg)", fontsize=12); ax.set_ylabel(r"$C$ (cnt/s)", fontsize=12)
    ax.set_title(fr"4.\ $C(|m|)$ at fixed dates ({len(date_picks)} slices)  $-$  aggregate",
                 fontsize=11)
    ax.legend(fontsize=7, loc='upper left', ncol=3, title="date", columnspacing=0.6)
    ax.grid(alpha=0.3)

    # Panel 5: marginal C(|m|)
    ax = fig.add_subplot(gs[2, 0])
    with np.errstate(invalid='ignore'):
        C_mlat_d = np.nanmean(np.where(mask_agg, C_data_agg, np.nan), axis=1)
        C_mlat_m = np.nanmean(np.where(mask_agg, C_model_agg, np.nan), axis=1)
    ax.plot(mlat_centers, C_mlat_d, 'o-', lw=1.5, ms=4, color='black',
            label=r"data $\langle C\rangle_t$")
    ax.plot(mlat_centers, C_mlat_m, '-', lw=2.5, color='C3', label="C25 aggregate model")
    ax.set_xlabel(r"$|m|$ (deg)"); ax.set_ylabel(r"$\langle C\rangle_t$ (cnt/s)")
    ax.set_title(r"5.\ Marginal $C(|m|)$:  time-averaged data vs model")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Panel 6: marginal C(t)
    ax = fig.add_subplot(gs[2, 1])
    with np.errstate(invalid='ignore'):
        C_t_d = np.nanmean(np.where(mask_agg, C_data_agg, np.nan), axis=0)
        C_t_m = np.nanmean(np.where(mask_agg, C_model_agg, np.nan), axis=0)
    ax.plot(month_dt, C_t_d, 'o-', lw=1.5, ms=4, color='black',
            label=r"data $\langle C\rangle_{|m|}$")
    ax.plot(month_dt, C_t_m, '-', lw=2.5, color='C3', label="C25 aggregate model")
    ax.set_xlabel("date"); ax.set_ylabel(r"$\langle C\rangle_{|m|}$ (cnt/s)")
    ax.set_title(r"6.\ Marginal $C(t)$:  mlat-averaged data vs model")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 7: A_i bar chart
    ax = fig.add_subplot(gs[3, :])
    labels = [f"{'ABC'[i//6]}{i%6}" for i in range(18)]
    box_colors = ['#1f77b4']*6 + ['#2ca02c']*6 + ['#d62728']*6
    bars = ax.bar(labels, A_i, color=box_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(A_i.mean(), ls='--', color='gray', alpha=0.5,
               label=fr"$\langle A_i\rangle = {A_i.mean():.1f}$")
    ax.axhline(202.6, ls=':', color='C1', alpha=0.7,
               label=r"8p single $A=202.6$ (for reference)")
    for b, v in zip(bars, A_i):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.0f}", ha='center', fontsize=8)
    ax.set_ylabel(r"$A_i$ (cnt/s)", fontsize=12)
    ax.set_title(r"7.\ Per-detector amplitudes $A_i$  (the only C25-vs-8p visual difference)",
                 fontsize=11)
    ax.set_ylim(140, max(A_i)*1.05)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, axis='y')

    out_B = "plots/c25_breakdown.png"
    plt.savefig(out_B, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out_B}")


if __name__ == "__main__":
    main()
