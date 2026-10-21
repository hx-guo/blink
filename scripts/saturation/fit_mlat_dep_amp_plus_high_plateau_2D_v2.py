#!/usr/bin/env python3
"""Round 2 candidate: mlat_dep_amp_plus_high_plateau (11 params).

Adds a linear high-mlat plateau extension to the round-1 winner
(mlat_dependent_amp) to capture scattered residuals above ~50 deg.

Model (phenomenological, no PMT/solar/v5t labels):
  sm(mlat) = 1 / (1 + exp(-(mlat - mu_m)/k_m))
  st(t)    = 1 / (1 + exp(-(t   - mu_t)/k_t))
  hi(mlat) = max(0, mlat - theta_hi)
  A(mlat)  = a*(1 + alpha*sm) + c*hi                    # amplitude with plateau
  amp_eff  = amp0*(1 + beta*sm)                         # time-decay amplitude
  C(mlat,t)= A(mlat) * (1 - amp_eff*st) + C0

11 params: a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C0, c, theta_hi.

Pipeline:
  1) Fit on n_below_study/v5_npz/C_2D_heatmap.npz (least_squares, trf).
  2) Self-consistent eval on yearly cache row-group 0 samples:
     - C_pred per row via fitted model evaluated at (|mlat|, t_years)
     - unwrap_v2 + event-balance cap with C_pred
     - Sci_rec_residual = Sci_rec - Sci_obs (the WINNING metric)
     - unwrap_change_pct vs C=150 baseline
  3) 4-panel plot to plots/fit_<model>_2D_v2.png.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"
CACHE_DIR = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
AACGM = "n_below_study/aacgm_grid_2020.npz"
L_PER_CYCLE = 16e-6
T0 = np.datetime64("2017-06-22")
NEEDED = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s",
          "L_cycles", "Dt", "Lat", "Lon"]
MODEL_NAME = "mlat_dep_amp_plus_high_plateau"
PLOT_PATH = f"plots/fit_{MODEL_NAME}_2D_v2.png"


# ---------------- model ----------------

def model(params, mlat_grid, t_grid):
    """Return C with shape (n_mlat, n_t).

    mlat_grid: (n_mlat,) absolute mlat in deg
    t_grid:    (n_t,) time in years since T0
    """
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C0, c, theta_hi = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))  # (n_mlat,)
    st = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))      # (n_t,)
    hi = np.maximum(0.0, mlat_grid - theta_hi)              # (n_mlat,)
    A = a * (1.0 + alpha * sm) + c * hi                     # (n_mlat,)
    amp_eff = amp0 * (1.0 + beta * sm)                      # (n_mlat,)
    F = 1.0 - amp_eff[:, None] * st[None, :]                # (n_mlat, n_t)
    return A[:, None] * F + C0


def model_per_row(params, mlat_abs, t_years):
    """Vectorised pointwise evaluation for cache rows."""
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C0, c, theta_hi = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_abs - mu_m) / k_m))
    st = 1.0 / (1.0 + np.exp(-(t_years - mu_t) / k_t))
    hi = np.maximum(0.0, mlat_abs - theta_hi)
    A = a * (1.0 + alpha * sm) + c * hi
    amp_eff = amp0 * (1.0 + beta * sm)
    return A * (1.0 - amp_eff * st) + C0


# ---------------- unwrap_v2 + event balance cap ----------------

def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    pho = np.asarray(pho, float); large = np.asarray(large, float)
    wide = np.asarray(wide, float); sci = np.asarray(sci, float)
    LL = np.asarray(lc, float) * L_PER_CYCLE
    lf = 1.0 - np.asarray(dt, float) / np.asarray(lc, float)
    pred = pho - (wide + (sci + C) * LL) / lf
    n = np.maximum(np.round((pred - large) / 1024.0).astype(int), 0)
    mx = pho - wide
    out = large + n * 1024.0
    ov = out > mx
    if ov.any():
        nm = np.maximum(np.floor((mx - large) / 1024.0).astype(int), 0)
        out = large + np.where(ov, nm, n) * 1024.0
    return out


def sci_rec_pipeline(pho, lg, wd, sci, lc, dtv, C_pred):
    """Return (Sci_rec, lv_final). Implements unwrap_v2 + event-balance cap."""
    LL = lc * L_PER_CYCLE; lf = 1.0 - dtv / lc
    lv1 = unwrap_v2(pho, lg, wd, sci, lc, dtv, C_pred)
    mle = pho - ((sci + 0.0) * LL + wd) / lf
    n1 = np.round((lv1 - lg) / 1024.0).astype(int)
    nmax = np.maximum(np.floor((mle - lg) / 1024.0).astype(int), 0)
    lv_final = lg + np.where(n1 > nmax, nmax, n1) * 1024.0
    Sci_rec = (pho - lv_final) * lf / LL - wd / LL - C_pred
    return Sci_rec, lv_final


# ---------------- main ----------------

def main():
    Path("plots").mkdir(exist_ok=True)

    # ---- Step 1: fit on heatmap ----
    z = np.load(NPZ)
    C_data = z["C_med"]
    n_data = z["C_n"]
    months = z["months"]
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t_years = ((month_dt - T0).astype("timedelta64[D]").astype(float)) / 365.25

    mask = n_data > 200
    print(f"valid heatmap bins: {mask.sum()}/{mask.size}")
    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model(params, mlat_centers, t_years)
        return ((C_pred - C_data_clean) * mask).ravel()

    #   a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t,   C0,    c,  theta_hi
    p0 = [100.0, 3.0, 50.0, 4.0,  0.5, 0.5,  3.0,  2.0,   0.0, 5.0,  50.0]
    lo = [ 10.0, 0.5, 30.0, 0.5,  0.0, -2.0, 0.5,  0.2, -100.0, 0.0,  40.0]
    hi = [500.0,20.0, 60.0,15.0,  1.5,  5.0, 8.0,  8.0, +100.0, 60.0, 58.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    (a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C0, c, theta_hi) = res.x
    print(f"\n=== Fitted parameters (11) ===")
    print(f"  a        = {a:8.2f}")
    print(f"  alpha    = {alpha:8.3f}")
    print(f"  mu_m     = {mu_m:8.2f}")
    print(f"  k_m      = {k_m:8.2f}")
    print(f"  amp0     = {amp0:8.3f}")
    print(f"  beta     = {beta:8.3f}")
    print(f"  mu_t     = {mu_t:8.2f}")
    print(f"  k_t      = {k_t:8.2f}")
    print(f"  C0       = {C0:+8.2f}")
    print(f"  c        = {c:8.3f}")
    print(f"  theta_hi = {theta_hi:8.2f}")

    C_pred_grid = model(res.x, mlat_centers, t_years)
    resid_grid = (C_data - C_pred_grid) * mask
    valid_resid = resid_grid[mask]
    C_residual_std = float(np.std(valid_resid))
    C_residual_max = float(np.max(np.abs(valid_resid)))
    mean_C = float(np.mean(C_data[mask]))
    print(f"\n=== Heatmap residual stats ===")
    print(f"  std: {C_residual_std:.3f} cnt/s  max: {C_residual_max:.2f} cnt/s")
    print(f"  std/mean(C_data): {C_residual_std / mean_C * 100.0:.2f}%")

    # ---- Step 2: self-consistent eval on cache (one rg per yearly file) ----
    grid_az = np.load(AACGM)
    interp = RegularGridInterpolator(
        (grid_az["lat_grid"], grid_az["lon_grid"]),
        grid_az["mlat"], bounds_error=False, fill_value=np.nan,
    )
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "clean_relaxed_20*.parquet")))
    print(f"\n=== Step 2: cache eval, {len(files)} yearly files, rg0 each ===")

    all_resid = []
    all_resid_baseline = []
    n_total = 0; n_changed_unwrap = 0; n_eval = 0
    for f in files:
        pf = pq.ParquetFile(f)
        df = pf.read_row_group(0, columns=NEEDED).to_pandas()
        if len(df) == 0:
            continue
        pho = df["PHO"].astype(float).values
        lg = df["Large"].astype(float).values
        wd = df["Wide"].astype(float).values
        sci = df["Sci_1s"].astype(float).values
        lc = df["L_cycles"].astype(float).values
        dtv = df["Dt"].astype(float).values

        # AACGM mlat for each row
        lat = df["Lat"].astype(float).values
        lon = df["Lon"].astype(float).values
        mlat = interp(np.column_stack([lat, lon]))
        mlat_abs = np.abs(mlat)
        mlat_abs = np.where(np.isnan(mlat_abs), 0.0, mlat_abs)

        # date -> t_years
        d_arr = np.array([np.datetime64(d) for d in df["date"].values])
        t_years_row = (d_arr - T0).astype("timedelta64[D]").astype(float) / 365.25

        C_pred = model_per_row(res.x, mlat_abs, t_years_row)

        # New pipeline (with predicted C)
        Sci_rec_new, lv_new = sci_rec_pipeline(pho, lg, wd, sci, lc, dtv, C_pred)
        # Baseline (C=150) for unwrap_change_pct
        lv_base = unwrap_v2(pho, lg, wd, sci, lc, dtv, 150.0)
        # event-balance cap on baseline too, to compare apples-to-apples
        LL = lc * L_PER_CYCLE; lf = 1.0 - dtv / lc
        mle_b = pho - ((sci + 0.0) * LL + wd) / lf
        n_b = np.round((lv_base - lg) / 1024.0).astype(int)
        nmax_b = np.maximum(np.floor((mle_b - lg) / 1024.0).astype(int), 0)
        lv_base_capped = lg + np.where(n_b > nmax_b, nmax_b, n_b) * 1024.0

        resid_new = Sci_rec_new - sci
        # Filter
        keep = (sci > 50) & np.isfinite(resid_new) & (np.abs(resid_new) < 1000.0)
        all_resid.append(resid_new[keep])

        # unwrap_change_pct: rows where lv differs (within keep mask scope)
        changed = (lv_new != lv_base_capped)
        n_changed_unwrap += int(np.sum(changed[keep]))
        n_eval += int(np.sum(keep))
        n_total += int(len(df))
        print(f"  {os.path.basename(f)}: rows={len(df)}, kept={int(keep.sum())}, "
              f"changed={int(changed[keep].sum())}")

    resid_all = np.concatenate(all_resid) if all_resid else np.zeros(0)
    if resid_all.size:
        Sci_rec_residual_std = float(np.std(resid_all))
        Sci_rec_residual_max = float(np.max(np.abs(resid_all)))
    else:
        Sci_rec_residual_std = float("nan"); Sci_rec_residual_max = float("nan")
    unwrap_change_pct = 100.0 * n_changed_unwrap / max(1, n_eval)
    print(f"\n=== Cache eval stats ({n_eval} rows kept of {n_total}) ===")
    print(f"  Sci_rec_residual_std = {Sci_rec_residual_std:.3f} cnt/s")
    print(f"  Sci_rec_residual_max = {Sci_rec_residual_max:.2f} cnt/s")
    print(f"  unwrap_change_pct    = {unwrap_change_pct:.3f}%")

    # ---- Step 3: 4-panel plot ----
    fig, axes = plt.subplots(4, 1, figsize=(15, 18))
    fig.suptitle(
        f"{MODEL_NAME} 2D v2 fit (11 params)\n"
        f"a={a:.0f}, alpha={alpha:.2f}, mu_m={mu_m:.1f}, k_m={k_m:.1f}, "
        f"amp0={amp0:.2f}, beta={beta:.2f}, mu_t={mu_t:.1f}, k_t={k_t:.1f}, "
        f"C0={C0:.0f}, c={c:.2f}, theta_hi={theta_hi:.1f}\n"
        f"C_residual_std = {C_residual_std:.2f} cnt/s; "
        f"Sci_rec_residual_std = {Sci_rec_residual_std:.2f} cnt/s "
        f"(WINNING METRIC); unwrap_change = {unwrap_change_pct:.2f}%",
        fontsize=11, fontweight='bold')

    C_data_m = C_data.astype(float).copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred_grid.astype(float).copy(); C_pred_m[~mask] = np.nan
    resid_m = resid_grid.astype(float).copy(); resid_m[~mask] = np.nan

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel("|mlat| (deg)", fontsize=11)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_m, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA — mean C (cnt/s)')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — mlat_dep_amp_plus_high_plateau')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL — data - model (+/- 30 cnt/s)')
    axes[2].set_xlabel("date", fontsize=11)

    ax = axes[3]
    if resid_all.size:
        ax.hist(resid_all, bins=200, range=(-200, 200), color='steelblue',
                edgecolor='none')
    ax.set_yscale('log')
    ax.set_xlim(-200, 200)
    ax.set_xlabel('Sci_rec - Sci_obs (cnt/s)', fontsize=11)
    ax.set_ylabel('rows (log)', fontsize=11)
    ax.set_title(
        f"4. Sci_rec residual on cache (n={resid_all.size}); "
        f"std={Sci_rec_residual_std:.2f}, max={Sci_rec_residual_max:.1f}",
        fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {PLOT_PATH}")

    # ---- Print JSON summary ----
    import json
    summary = {
        "model_name": MODEL_NAME,
        "formula": ("C = (a*(1+alpha*sm) + c*max(0,mlat-theta_hi)) "
                    "* (1 - amp0*(1+beta*sm)*st) + C0"),
        "n_params": 11,
        "params": {
            "a": float(a), "alpha": float(alpha),
            "mu_m": float(mu_m), "k_m": float(k_m),
            "amp0": float(amp0), "beta": float(beta),
            "mu_t": float(mu_t), "k_t": float(k_t),
            "C0": float(C0), "c": float(c), "theta_hi": float(theta_hi),
        },
        "C_residual_std": C_residual_std,
        "Sci_rec_residual_std": Sci_rec_residual_std,
        "Sci_rec_residual_max": Sci_rec_residual_max,
        "unwrap_change_pct": unwrap_change_pct,
        "plot_path": PLOT_PATH,
    }
    print("\n=== JSON summary ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
