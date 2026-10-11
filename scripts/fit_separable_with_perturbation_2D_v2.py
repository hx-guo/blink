#!/usr/bin/env python3
"""Fit the separable_with_perturbation 2D model to C(mlat, t).

Model (10 free params):
    C(m, t) = a * sm(m) * st(t) + delta * sm(m) * (t - mu_t) * exp(-(t-mu_t)^2 / (2*sigma_t^2)) + C_0

  sm(m)  = 1 / (1 + exp(-(m - mu_m)/k_m))     mlat sigmoid (shared envelope)
  st(t)  = 1 - amp0 * sig_t(t),   sig_t(t)=1/(1+exp(-(t-mu_t_s)/k_t_s))
           (separable monotonic time decay)

Parameters:
  a, mu_m, k_m, amp0, mu_t_s, k_t_s, delta, mu_t, sigma_t, C_0

The perturbation term is a Gaussian-modulated linear pulse: a localized mid-mission
bump that crosses zero at t = mu_t, peaks/troughs at mu_t +/- sigma_t, and dies away
on a timescale ~sigma_t. It only modulates high-mlat region via sm(m).

Then runs a self-consistent unwrap using the fitted C(mlat,t) on sampled cache rows.
"""
from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"
GRID = "n_below_study/aacgm_grid_2020.npz"
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
L = 16e-6   # cycle period seconds
T0 = np.datetime64("2017-06-22")


# ============== MODEL ==============
def model(params, mlat_grid, t_grid):
    """C(mlat, t).
    mlat_grid (n_mlat,), t_grid (n_t,). Returns (n_mlat, n_t).
    """
    a, mu_m, k_m, amp0, mu_t_s, k_t_s, delta, mu_t, sigma_t, C_0 = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))          # (n_mlat,)
    sig_t = 1.0 / (1.0 + np.exp(-(t_grid - mu_t_s) / k_t_s))      # (n_t,)
    st = 1.0 - amp0 * sig_t                                       # (n_t,)
    # Gaussian-modulated linear perturbation in time
    pert = delta * (t_grid - mu_t) * np.exp(-(t_grid - mu_t) ** 2 / (2.0 * sigma_t ** 2))  # (n_t,)
    out = a * sm[:, None] * st[None, :] + sm[:, None] * pert[None, :] + C_0
    return out


def model_pointwise(params, mlat_vec, t_vec):
    """Pointwise C for cache rows."""
    a, mu_m, k_m, amp0, mu_t_s, k_t_s, delta, mu_t, sigma_t, C_0 = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_vec - mu_m) / k_m))
    sig_t = 1.0 / (1.0 + np.exp(-(t_vec - mu_t_s) / k_t_s))
    st = 1.0 - amp0 * sig_t
    pert = delta * (t_vec - mu_t) * np.exp(-(t_vec - mu_t) ** 2 / (2.0 * sigma_t ** 2))
    return a * sm * st + sm * pert + C_0


# ============== UNWRAP ==============
def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    pho = np.asarray(pho, float); large = np.asarray(large, float)
    wide = np.asarray(wide, float); sci = np.asarray(sci, float)
    LL = np.asarray(lc, float) * L
    lf = 1.0 - np.asarray(dt, float) / np.asarray(lc, float)
    pred = pho - (wide + (sci + C) * LL) / lf
    n = np.maximum(np.round((pred - large) / 1024.).astype(int), 0)
    mx = pho - wide
    out = large + n * 1024.
    ov = out > mx
    if ov.any():
        nm = np.maximum(np.floor((mx - large) / 1024.).astype(int), 0)
        out = large + np.where(ov, nm, n) * 1024.
    return out


def pipeline_lvfinal(pho, lg, wd, sci, lc, dtv, C):
    """unwrap_v2 + event-balance cap (slack=0). Returns lv_final."""
    LL = lc * L
    lf = 1.0 - dtv / lc
    lv1 = unwrap_v2(pho, lg, wd, sci, lc, dtv, C)
    mle = pho - ((sci + 0.0) * LL + wd) / lf
    n1 = np.round((lv1 - lg) / 1024).astype(int)
    nmax = np.maximum(np.floor((mle - lg) / 1024.).astype(int), 0)
    lv_final = lg + np.where(n1 > nmax, nmax, n1) * 1024.
    return lv_final


# ============== MAIN ==============
def main():
    # ----- load heatmap -----
    z = np.load(NPZ)
    C_data = z["C_med"]              # (60, 108)
    n_data = z["C_n"]
    months = z["months"]             # strings "YYYY-MM"
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])

    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t_years = ((month_dt - T0).astype("timedelta64[D]").astype(float)) / 365.25
    mask = n_data > 200
    print(f"valid bins: {mask.sum()}/{mask.size}")

    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # initial: borrow from mlat_dependent_amp where applicable, add perturbation.
    # Try a few seeds to escape weak local minima.
    # a, mu_m, k_m, amp0, mu_t_s, k_t_s, delta, mu_t, sigma_t, C_0
    seeds = [
        [200.0, 45.0, 6.0, 0.4, 5.0, 1.5,  20.0, 5.0, 0.8, -20.0],
        [200.0, 45.0, 6.0, 0.2, 4.0, 1.0,  30.0, 4.0, 0.8,  10.0],
        [202.6, 44.5, 6.3, 0.15, 5.25, 1.0, 30.0, 5.0, 1.0,  -79.3],
        [220.0, 46.0, 7.0, 0.3, 5.5, 1.5, -20.0, 5.0, 1.2,  -40.0],
        [200.0, 45.0, 6.0, 0.4, 5.0, 1.5,  40.0, 3.5, 0.8,   0.0],
    ]
    lo = [10.0,  30.0,  0.5, 0.0,  0.5, 0.2, -400.0, 1.0, 0.2, -300.0]
    hi = [500.0, 60.0, 15.0, 1.5,  9.0, 8.0, +400.0, 9.0, 4.0, +300.0]

    best = None
    for i, p0 in enumerate(seeds):
        r_ = least_squares(residual, p0, bounds=(lo, hi),
                           method='trf', max_nfev=5000)
        print(f"seed {i}: cost={r_.cost:.0f}, nfev={r_.nfev}, success={r_.success}")
        if best is None or r_.cost < best.cost:
            best = r_
    res = best
    print(f"BEST fit cost: {res.cost:.0f}, nfev: {res.nfev}")

    (a, mu_m, k_m, amp0, mu_t_s, k_t_s, delta, mu_t, sigma_t, C_0) = res.x
    print("=== Fitted parameters (10 free) ===")
    print(f"  a        = {a:8.2f}")
    print(f"  mu_m     = {mu_m:8.2f}")
    print(f"  k_m      = {k_m:8.2f}")
    print(f"  amp0     = {amp0:8.3f}")
    print(f"  mu_t_s   = {mu_t_s:8.2f}")
    print(f"  k_t_s    = {k_t_s:8.2f}")
    print(f"  delta    = {delta:+8.2f}")
    print(f"  mu_t     = {mu_t:8.2f}")
    print(f"  sigma_t  = {sigma_t:8.2f}")
    print(f"  C_0      = {C_0:+8.2f}")

    # heatmap residual stats
    C_pred = model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid = resid[mask]
    C_residual_std = float(np.std(valid))
    C_residual_max = float(np.max(np.abs(valid)))
    mean_C = float(np.mean(C_data[mask]))
    print(f"\n=== Heatmap residual ({mask.sum()} bins) ===")
    print(f"  mean : {np.mean(valid):+.3f} cnt/s")
    print(f"  std  : {C_residual_std:.3f} cnt/s  ({C_residual_std/mean_C*100:.2f}% of mean)")
    print(f"  max  : {C_residual_max:.2f}")

    # ----- self-consistent eval on cache -----
    grid = np.load(GRID)
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)

    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))
    print(f"\n=== Self-consistent unwrap on {len(files)} parquet files (rg=0 each) ===")

    resid_chunks = []
    nchanged = 0
    ntotal = 0

    for f in files:
        pf = pq.ParquetFile(f)
        df = pf.read_row_group(0, columns=NEEDED).to_pandas()
        pho = df["PHO"].astype(float).values
        lg = df["Large"].astype(float).values
        wd = df["Wide"].astype(float).values
        sci = df["Sci_1s"].astype(float).values
        lc = df["L_cycles"].astype(float).values
        dtv = df["Dt"].astype(float).values

        # |mlat| per row
        am = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
        am = np.where(np.isnan(am), 0.0, am)

        # time in years since T0
        d_arr = np.array([np.datetime64(d) for d in df["date"].values])
        ty = (d_arr - T0).astype("timedelta64[D]").astype(float) / 365.25

        # C from model
        C_pred_row = model_pointwise(res.x, am, ty)

        # pipeline with fitted C
        lv_fit = pipeline_lvfinal(pho, lg, wd, sci, lc, dtv, C_pred_row)
        # baseline pipeline with C = 150
        lv_base = pipeline_lvfinal(pho, lg, wd, sci, lc, dtv, 150.0)

        # Sci_rec from fitted pipeline
        LL = lc * L
        lf = 1.0 - dtv / lc
        Sci_rec = (pho - lv_fit) * lf / LL - wd / LL - C_pred_row
        r = Sci_rec - sci

        ok = np.isfinite(r) & (sci > 50.0) & (np.abs(r) < 1000.0)
        resid_chunks.append(r[ok])

        # unwrap change count (using the SAME ok filter)
        nchanged += (lv_fit[ok] != lv_base[ok]).sum()
        ntotal += int(ok.sum())

        print(f"  {os.path.basename(f)}: rows={len(df):,}  ok={ok.sum():,}  "
              f"r_std={np.std(r[ok]):.2f}", flush=True)

    resid_all = np.concatenate(resid_chunks)
    Sci_rec_residual_std = float(np.std(resid_all))
    Sci_rec_residual_max = float(np.max(np.abs(resid_all)))
    unwrap_change_pct = 100.0 * nchanged / max(ntotal, 1)
    print(f"\n=== Cache residual (n={len(resid_all):,}) ===")
    print(f"  std : {Sci_rec_residual_std:.3f} cnt/s")
    print(f"  max : {Sci_rec_residual_max:.2f} cnt/s")
    print(f"  unwrap_change_pct vs C=150 : {unwrap_change_pct:.4f}%")

    # ----- plots: 4-panel -----
    resid_masked = (C_data - C_pred).astype(float)
    resid_masked[~mask] = np.nan
    C_data_m = C_data.astype(float).copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.astype(float).copy(); C_pred_m[~mask] = np.nan

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.32)
    fig.suptitle(
        f"separable_with_perturbation  (10 params)\n"
        f"C = a*sm*[1-amp0*sig_t] + delta*sm*(t-mu_t)*exp(-(t-mu_t)^2/(2 sigma_t^2)) + C0\n"
        f"a={a:.0f}  mu_m={mu_m:.1f}  k_m={k_m:.1f}  amp0={amp0:.2f}  "
        f"mu_t_s={mu_t_s:.2f}  k_t_s={k_t_s:.2f}\n"
        f"delta={delta:+.2f}  mu_t={mu_t:.2f}  sigma_t={sigma_t:.2f}  C0={C_0:+.0f}\n"
        f"C_resid_std={C_residual_std:.2f}    "
        f"Sci_rec_resid_std={Sci_rec_residual_std:.2f}    "
        f"unwrap_change={unwrap_change_pct:.3f}%",
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

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    plot_pcm(ax0, C_data_m, 0, 400, 'viridis',
             'C (cnt/s)', '1. DATA — mean C from heatmap')
    plot_pcm(ax1, C_pred_m, 0, 400, 'viridis',
             'C (cnt/s)', '2. MODEL — separable_with_perturbation')
    plot_pcm(ax2, resid_masked, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. C residual (+/- 30 cnt/s)')

    ax3.hist(resid_all, bins=300, range=(-200, 200), color='steelblue',
             edgecolor='none', log=True)
    ax3.axvline(0.0, color='k', lw=0.7, ls='--')
    ax3.set_xlim(-200, 200)
    ax3.set_xlabel("Sci_rec - Sci_obs (cnt/s)")
    ax3.set_ylabel("count (log)")
    ax3.set_title(f"4. Self-consistent Sci_rec residual on cache  "
                  f"(std={Sci_rec_residual_std:.2f}, n={len(resid_all):,})")
    ax3.grid(True, alpha=0.3)

    Path("plots").mkdir(exist_ok=True)
    out_png = "plots/fit_separable_with_perturbation_2D_v2.png"
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out_png}")

    # ----- emit JSON-ish summary -----
    import json
    params = {
        "a": float(a), "mu_m": float(mu_m), "k_m": float(k_m),
        "amp0": float(amp0), "mu_t_s": float(mu_t_s), "k_t_s": float(k_t_s),
        "delta": float(delta), "mu_t": float(mu_t), "sigma_t": float(sigma_t),
        "C_0": float(C_0),
    }
    summary = {
        "model_name": "separable_with_perturbation",
        "formula": "C = a*sm*[1-amp0*sig_t] + delta*sm*(t-mu_t)*exp(-(t-mu_t)^2/(2 sigma_t^2)) + C0",
        "n_params": 10,
        "params": params,
        "C_residual_std": C_residual_std,
        "C_residual_max": C_residual_max,
        "Sci_rec_residual_std": Sci_rec_residual_std,
        "Sci_rec_residual_max": Sci_rec_residual_max,
        "unwrap_change_pct": unwrap_change_pct,
        "plot_path": out_png,
        "n_cache_rows": int(len(resid_all)),
    }
    print("\n=== SUMMARY JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
