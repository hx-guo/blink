#!/usr/bin/env python3
"""Fit 8-param α=β sigmoid model on the C(ACD, t) heatmap.

Same form as the mlat-based fit, but with sigmoid in log10(ACD) instead of mlat:
  g(A) = 1 + α · σ((log10(A) - μ_A) / k_A)
  C(A, t) = a · g · [1 − amp₀ · g · σ_t(t)] + C₀

8 globals: a, α, μ_A, k_A, amp₀, μ_t, k_t, C₀
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_ACD_t_heatmap.npz"


def sigm(x):
    z = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def model_2d(params, log_acd, t):
    a, alpha, mu_A, k_A, amp0, mu_t, k_t, C_0 = params
    sm = sigm((log_acd - mu_A) / k_A)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g[:, None] * (1.0 - amp0 * g[:, None] * st[None, :]) + C_0


def main():
    z = np.load(NPZ)
    C_data = z["C_mean"]
    n_data = z["n"]
    months = z["months"]
    acd_edges = z["acd_edges"]
    acd_centers = np.sqrt(acd_edges[:-1] * acd_edges[1:])
    log_acd = np.log10(acd_centers)
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t0 = np.datetime64("2017-06-22")
    t_years = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25
    mask = n_data > 200
    print(f"valid bins: {mask.sum()}/{mask.size}")

    C_clean = np.where(mask, C_data, 0.0)
    def residual(p):
        return ((model_2d(p, log_acd, t_years) - C_clean) * mask).ravel()

    # log10(ACD) range ~ 0..3.3
    p0 = [100.0, 2.0, 2.3, 0.3, 0.15, 5.2, 1.0, 50.0]
    lo = [10.0, 0.1, 0.0, 0.05, 0.0, 0.5, 0.2, -300.0]
    hi = [500.0, 20.0, 4.0, 2.0, 1.0, 8.0, 8.0, +300.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    a, alpha, mu_A, k_A, amp0, mu_t, k_t, C_0 = res.x

    print("\n=== Fitted parameters (8-param ACD-based) ===")
    print(f"  a      = {a:8.2f}")
    print(f"  α      = {alpha:8.3f}")
    print(f"  μ_A    = {mu_A:8.3f}   (log10 ACD, → {10**mu_A:.1f} cnt/s)")
    print(f"  k_A    = {k_A:8.3f}")
    print(f"  amp₀   = {amp0:8.3f}")
    print(f"  μ_t    = {mu_t:8.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t    = {k_t:8.2f} yr")
    print(f"  C_0    = {C_0:+8.2f}")

    C_pred = model_2d(res.x, log_acd, t_years)
    resid = (C_data - C_pred) * mask
    valid = resid[mask]
    print(f"\n=== Residual stats ===")
    print(f"  std:    {np.std(valid):.3f} cnt/s")
    print(f"  RMS/mean: {np.sqrt(np.mean(valid**2))/np.mean(C_data[mask])*100:.2f}%")
    print(f"  max|r|: {np.max(np.abs(valid)):.1f}")
    print(f"\nCompare: 8p mlat-based α=β baseline std = 6.49 cnt/s")
    print(f"         11p dual-σ mlat std            = 5.79 cnt/s")
    print(f"         TPS noise floor (mlat)         = 5.49 cnt/s")

    # Plot 3-panel heatmap
    resid_m = resid.copy(); resid_m[~mask] = np.nan
    C_data_m = C_data.copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy(); C_pred_m[~mask] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "ACD-based 8-param α=β fit:  C = a·g·[1 − amp₀·g·σ_t] + C₀,  "
        "g = 1 + α·σ((log10(ACD) − μ_A)/k_A)\n"
        f"a={a:.0f}, α={alpha:.2f}, μ_A=log10({10**mu_A:.0f}), k_A={k_A:.2f}, "
        f"amp₀={amp0:.3f}, μ_t={mu_t:.2f}yr, k_t={k_t:.2f}yr, C₀={C_0:+.0f}",
        fontsize=11, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1])/2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])
    # ACD y axis: log scale
    y_edges = acd_edges

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, y_edges, data,
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel("ACD lookup (cnt/s)", fontsize=11)
        ax.set_yscale('log')
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_m, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA — mean C (cnt/s) over (ACD, month) bins')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — ACD-based 8 params')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data − model (cnt/s)', '3. RESIDUAL — data − model (±30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_C_ACD_t_8param.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # Persist params for downstream row-level evaluation
    import json
    Path("/tmp/acd_fit_params.json").write_text(json.dumps(list(res.x)))
    print("Saved /tmp/acd_fit_params.json")


if __name__ == "__main__":
    main()
