#!/usr/bin/env python3
"""Fit 25-param model: per-detector a_det × 18 + shared 7.

Model:
  g(mlat) = 1 + α · σ_m(mlat)
  C(det, mlat, t) = a_det · g · [1 − amp₀ · g · σ_t(t)] + C₀

25 globals:
  - a_det[18]: per-detector amplitude
  - α, μ_m, k_m, amp₀, μ_t, k_t, C₀: 7 shared
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_det_mlat_t_heatmap.npz"


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def model(params, mlat_centers, t_years):
    # params layout: [a_det[18], α, μ_m, k_m, amp0, μ_t, k_t, C₀]
    a_det = params[:18]                          # (18,)
    alpha, mu_m, k_m, amp0, mu_t, k_t, C0 = params[18:25]
    sm = sigm((mlat_centers - mu_m) / k_m)       # (n_mlat,)
    st = sigm((t_years - mu_t) / k_t)             # (n_t,)
    g = 1.0 + alpha * sm                          # (n_mlat,)
    # Shape: (det, mlat, t) = a_det[:,None,None] · g[None,:,None] · (1 - amp0·g[None,:,None]·st[None,None,:]) + C0
    return (a_det[:, None, None]
            * g[None, :, None]
            * (1.0 - amp0 * g[None, :, None] * st[None, None, :])
            + C0)


def main():
    z = np.load(NPZ)
    C_data = z["C_mean"]      # (18, 60, 108)
    n_data = z["n"]
    months = z["months"]
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t0 = np.datetime64("2017-06-22")
    t_years = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25
    mask = n_data > 50
    print(f"valid bins: {mask.sum()}/{mask.size}")

    C_clean = np.where(mask, C_data, 0.0)
    def residual(p):
        return ((model(p, mlat_centers, t_years) - C_clean) * mask).ravel()

    # Initial: a_det all 200, shared from 8p fit
    p0 = np.concatenate([
        np.full(18, 200.0),                # a_det
        [1.7, 44.0, 6.0, 0.15, 5.2, 1.0, -80.0],   # α, μ_m, k_m, amp₀, μ_t, k_t, C₀
    ])
    lo = np.concatenate([
        np.full(18, 10.0),
        [0.1, 30.0, 0.5, 0.0, 0.5, 0.2, -300.0],
    ])
    hi = np.concatenate([
        np.full(18, 500.0),
        [10.0, 60.0, 15.0, 1.0, 8.0, 8.0, 300.0],
    ])
    print(f"Fitting {len(p0)} params on {mask.sum()} bins...")
    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=8000)
    print(f"  cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")

    a_det = res.x[:18]
    alpha, mu_m, k_m, amp0, mu_t, k_t, C0 = res.x[18:25]
    print("\n=== Fitted parameters (25-param per-det a) ===")
    print(f"  a_det (18):  min={a_det.min():.1f}, median={np.median(a_det):.1f}, max={a_det.max():.1f}")
    for i, a in enumerate(a_det):
        box = "ABC"[i//6]; det = i % 6
        print(f"    {box}{det}: a={a:.1f}")
    print(f"\n  α      = {alpha:.3f}")
    print(f"  μ_m    = {mu_m:.2f}°")
    print(f"  k_m    = {k_m:.2f}°")
    print(f"  amp₀   = {amp0:.3f}")
    print(f"  μ_t    = {mu_t:.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t    = {k_t:.2f} yr")
    print(f"  C_0    = {C0:+.2f}")

    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid = resid[mask]
    print(f"\n=== Heatmap residual stats ===")
    print(f"  std:    {np.std(valid):.3f} cnt/s")
    print(f"  RMS/mean: {np.sqrt(np.mean(valid**2))/np.mean(C_data[mask])*100:.2f}%")
    print(f"  max|r|: {np.max(np.abs(valid)):.1f}")
    print("\n  cf. 8p (single a) heatmap std on (mlat, t):     6.49 cnt/s")
    print("      11p dual-σ heatmap std:                       5.79 cnt/s")
    print("      TPS smooth basis noise floor (mlat, t):       5.49 cnt/s")

    # Plot per-det residual heatmaps (4 boxes worth: 18 small panels)
    fig, axes = plt.subplots(3, 6, figsize=(22, 11), sharex=True, sharey=True)
    fig.suptitle(
        f"25-param per-det model: residuals per detector (heatmap std={np.std(valid):.2f} cnt/s)",
        fontsize=13, fontweight='bold')
    import matplotlib.dates as mdates
    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1])/2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])
    for i in range(18):
        ax = axes[i//6, i%6]
        r = resid[i].copy()
        r[~mask[i]] = np.nan
        pcm = ax.pcolormesh(x_edges, edges, r, cmap='RdBu_r', vmin=-30, vmax=30, shading='flat')
        box = "ABC"[i//6]; det = i % 6
        ax.set_title(f"{box}{det}  a={a_det[i]:.0f}", fontsize=9)
        if i//6 == 2:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
            ax.set_xlabel("date")
        if i%6 == 0:
            ax.set_ylabel("|mlat|")
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/fit_per_det_25param_residual.png"
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # Save params for downstream row-level eval
    Path("/tmp/per_det_25param.json").write_text(json.dumps({
        "a_det": a_det.tolist(), "alpha": alpha, "mu_m": mu_m, "k_m": k_m,
        "amp0": amp0, "mu_t": mu_t, "k_t": k_t, "C0": C0,
    }))
    print("Saved /tmp/per_det_25param.json")


if __name__ == "__main__":
    main()
