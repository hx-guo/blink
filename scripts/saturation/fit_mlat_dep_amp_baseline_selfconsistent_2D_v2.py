#!/usr/bin/env python3
"""Round 2 baseline: re-run the Round 1 winner (mlat_dependent_amp, 9 params) with
self-consistent unwrap on cache rows.

Model:
  C(mlat, t) = a * (1 + alpha * sm) * (1 - amp0 * (1 + beta * sm) * st) + C0
    sm = sigm((|mlat| - mu_m) / k_m)
    st = sigm((t - mu_t) / k_t)

Stages:
  1. Fit on n_below_study/v5_npz/C_2D_heatmap.npz (60 mlat bins x 108 month bins).
  2. Self-consistent unwrap eval on one row group per year from the cache.
  3. Write 4-panel diagnostic plot.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.parquet as pq
from scipy.optimize import least_squares

# ---------- paths ----------
ROOT = Path("/Users/skyair/Developer/ihep/blink")
NPZ = ROOT / "n_below_study/v5_npz/C_2D_heatmap.npz"
AACGM = ROOT / "n_below_study/aacgm_grid_2020.npz"
CACHE_DIR = Path("/Volumes/Graphite/work/blink/blink_clean_relaxed")
CACHE_GLOB = "clean_relaxed_{year}.parquet"
PLOT_PATH = ROOT / "plots/fit_mlat_dep_amp_baseline_selfconsistent_2D_v2.png"
YEARS = list(range(2017, 2027))  # 2017..2026

L = 16e-6
MIN_C_SLACK = 0.0
T0 = np.datetime64("2017-06-22")

NEEDED = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s",
          "L_cycles", "Dt", "Lat", "Lon"]


# ---------- model ----------
def model_2d(params, mlat_grid, t_grid):
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))
    st = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))
    A = a * (1.0 + alpha * sm)
    amp_eff = amp0 * (1.0 + beta * sm)
    F = 1.0 - amp_eff[:, None] * st[None, :]
    return A[:, None] * F + C_0


def model_pointwise(params, mlat_abs, t_years):
    """Per-row evaluation. mlat_abs, t_years are 1-D arrays of same length."""
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_abs - mu_m) / k_m))
    st = 1.0 / (1.0 + np.exp(-(t_years - mu_t) / k_t))
    A = a * (1.0 + alpha * sm)
    amp_eff = amp0 * (1.0 + beta * sm)
    return A * (1.0 - amp_eff * st) + C_0


# ---------- unwrap ----------
def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    pho = np.asarray(pho, float); large = np.asarray(large, float)
    wide = np.asarray(wide, float); sci = np.asarray(sci, float)
    lc = np.asarray(lc, float); dt = np.asarray(dt, float)
    C = np.asarray(C, float)
    LL = lc * L
    lf = 1.0 - dt / lc
    pred = pho - (wide + (sci + C) * LL) / lf
    n = np.maximum(np.round((pred - large) / 1024.0).astype(int), 0)
    mx = pho - wide
    out = large + n * 1024.0
    ov = out > mx
    if ov.any():
        nm = np.maximum(np.floor((mx - large) / 1024.0).astype(int), 0)
        out = large + np.where(ov, nm, n) * 1024.0
    return out


def apply_pipeline(pho, large, wide, sci, lc, dt, C):
    """Self-consistent unwrap with event-balance cap. Returns lv_final array."""
    LL = lc * L
    lf = 1.0 - dt / lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C)
    mle = pho - ((sci + MIN_C_SLACK) * LL + wide) / lf
    n1 = np.round((lv1 - large) / 1024.0).astype(int)
    nmax = np.maximum(np.floor((mle - large) / 1024.0).astype(int), 0)
    lv_final = large + np.where(n1 > nmax, nmax, n1) * 1024.0
    return lv_final


# ---------- helpers ----------
def aacgm_interp_mlat(lat, lon, lat_grid, lon_grid, mlat_table):
    """Bilinear interpolation of mlat at (lat, lon) points."""
    # lon in [0,360); lat in [-45,45]
    lat_idx = np.clip((lat - lat_grid[0]) / (lat_grid[-1] - lat_grid[0]) * (len(lat_grid) - 1), 0, len(lat_grid) - 1)
    lon_idx = np.clip((lon - lon_grid[0]) / (lon_grid[-1] - lon_grid[0]) * (len(lon_grid) - 1), 0, len(lon_grid) - 1)
    i0 = np.floor(lat_idx).astype(int); i1 = np.minimum(i0 + 1, len(lat_grid) - 1)
    j0 = np.floor(lon_idx).astype(int); j1 = np.minimum(j0 + 1, len(lon_grid) - 1)
    fi = lat_idx - i0; fj = lon_idx - j0
    m00 = mlat_table[i0, j0]; m01 = mlat_table[i0, j1]
    m10 = mlat_table[i1, j0]; m11 = mlat_table[i1, j1]
    return ((1 - fi) * (1 - fj) * m00 + (1 - fi) * fj * m01
            + fi * (1 - fj) * m10 + fi * fj * m11)


def main():
    # ---- 1. Load heatmap ----
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

    def residual_2d(params):
        C_pred = model_2d(params, mlat_centers, t_years)
        return ((C_pred - C_data_clean) * mask).ravel()

    p0 = [100.0, 3.0, 50.0, 4.0, 0.5, 0.5, 3.0, 2.0, 0.0]
    lo = [10.0, 0.5, 30.0, 0.5, 0.0, -2.0, 0.5, 0.2, -100.0]
    hi = [500.0, 20.0, 60.0, 15.0, 1.5, 5.0, 8.0, 8.0, 100.0]

    res = least_squares(residual_2d, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = res.x
    print(f"fit cost={res.cost:.0f} nfev={res.nfev} success={res.success}")
    print(f"  a={a:.2f} alpha={alpha:.3f} mu_m={mu_m:.2f} k_m={k_m:.2f}")
    print(f"  amp0={amp0:.3f} beta={beta:.3f} mu_t={mu_t:.2f} k_t={k_t:.2f}")
    print(f"  C_0={C_0:+.2f}")

    # ---- 2. Heatmap residual stats ----
    C_pred = model_2d(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid_resid = resid[mask]
    C_std = float(np.std(valid_resid))
    C_max = float(np.max(np.abs(valid_resid)))
    mean_C = float(np.mean(C_data[mask]))
    print(f"\n=== Heatmap residual ===")
    print(f"  std = {C_std:.3f} cnt/s  max|.|={C_max:.2f}  mean(C)={mean_C:.1f}")

    # ---- 3. Self-consistent eval on cache ----
    g = np.load(AACGM)
    lat_grid = g["lat_grid"]; lon_grid = g["lon_grid"]; mlat_table = g["mlat"]

    all_resid = []
    all_change = []
    n_rows_total = 0
    n_rows_used = 0
    for year in YEARS:
        path = CACHE_DIR / CACHE_GLOB.format(year=year)
        if not path.exists():
            print(f"  skip {path.name} (missing)")
            continue
        pf = pq.ParquetFile(str(path))
        if pf.num_row_groups == 0:
            continue
        tbl = pf.read_row_group(0, columns=NEEDED).to_pandas()
        n_rows_total += len(tbl)
        # cast
        pho = tbl["PHO"].astype(float).to_numpy()
        wide = tbl["Wide"].astype(float).to_numpy()
        large = tbl["Large"].astype(float).to_numpy()
        sci = tbl["Sci_1s"].astype(float).to_numpy()
        lc = tbl["L_cycles"].astype(float).to_numpy()
        dt = tbl["Dt"].astype(float).to_numpy()
        lat = tbl["Lat"].astype(float).to_numpy()
        lon = tbl["Lon"].astype(float).to_numpy()
        # dates -> years since T0
        dates = tbl["date"].astype(str).to_numpy()
        date_dt = np.array([np.datetime64(d) for d in dates])
        t_row = (date_dt - T0).astype("timedelta64[D]").astype(float) / 365.25
        # mlat
        mlat = aacgm_interp_mlat(lat, lon, lat_grid, lon_grid, mlat_table)
        mlat_abs = np.abs(mlat)
        # C_pred per row
        C_row = model_pointwise(res.x, mlat_abs, t_row)
        # baseline unwrap with C=150
        lv_base = apply_pipeline(pho, large, wide, sci, lc, dt, 150.0)
        # self-consistent
        lv_final = apply_pipeline(pho, large, wide, sci, lc, dt, C_row)
        # Sci_rec
        LL = lc * L
        lf = 1.0 - dt / lc
        sci_rec = (pho - lv_final) * lf / LL - wide / LL - C_row
        resid_row = sci_rec - sci
        # filter
        good = (sci > 50) & np.isfinite(resid_row) & (np.abs(resid_row) < 1000.0)
        # also require finite for unwrap_change calc; keep same mask
        all_resid.append(resid_row[good])
        all_change.append((lv_final[good] != lv_base[good]).astype(np.uint8))
        n_rows_used += int(good.sum())
        print(f"  {year}: rg0 rows={len(tbl)} kept={int(good.sum())}")

    resid_cat = np.concatenate(all_resid) if all_resid else np.array([])
    change_cat = np.concatenate(all_change) if all_change else np.array([])
    if resid_cat.size > 0:
        Sci_std = float(np.std(resid_cat))
        Sci_max = float(np.max(np.abs(resid_cat)))
        change_pct = float(change_cat.mean() * 100.0)
    else:
        Sci_std = float("nan"); Sci_max = float("nan"); change_pct = float("nan")
    print(f"\n=== Sci_rec residual on cache ({n_rows_used}/{n_rows_total} rows) ===")
    print(f"  std={Sci_std:.3f} cnt/s  max|.|={Sci_max:.2f}  unwrap_change={change_pct:.3f}%")

    # ---- 4. 4-panel plot ----
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    fig.suptitle(
        "BASELINE round-2  mlat_dep_amp self-consistent (9 params)\n"
        f"a={a:.0f}, alpha={alpha:.2f}, mu_m={mu_m:.1f}, k_m={k_m:.1f}, "
        f"amp0={amp0:.2f}, beta={beta:.2f}, mu_t={mu_t:.1f}, k_t={k_t:.1f}, C0={C_0:.0f}\n"
        f"C_resid_std={C_std:.2f} cnt/s    Sci_rec_resid_std={Sci_std:.2f} cnt/s    "
        f"unwrap_change={change_pct:.2f}%",
        fontsize=10, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    C_data_m = C_data.astype(float).copy(); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.astype(float).copy(); C_pred_m[~mask] = np.nan
    resid_m = resid.astype(float).copy(); resid_m[~mask] = np.nan

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel("|mlat| (deg)", fontsize=10)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_m, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA — mean C')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — mlat_dep_amp (baseline)')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL data - model (+/- 30 cnt/s)')
    # histogram
    if resid_cat.size > 0:
        axes[3].hist(np.clip(resid_cat, -200, 200), bins=200,
                     range=(-200, 200), color='steelblue', edgecolor='none')
        axes[3].set_yscale('log')
        axes[3].set_xlim(-200, 200)
        axes[3].axvline(0, color='red', lw=0.8, ls='--')
        axes[3].set_xlabel('Sci_rec - Sci_obs (cnt/s)', fontsize=10)
        axes[3].set_ylabel('rows (log)', fontsize=10)
        axes[3].set_title(f"4. Sci_rec residual histogram (N={resid_cat.size}, "
                          f"std={Sci_std:.2f}, max|.|={Sci_max:.1f})",
                          fontsize=10)
        axes[3].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(PLOT_PATH), dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {PLOT_PATH}")

    # ---- 5. summary JSON ----
    summary = {
        "model_name": "mlat_dep_amp_baseline_selfconsistent",
        "formula": "C = a*(1+alpha*sm)*(1 - amp0*(1+beta*sm)*st) + C0; "
                   "sm=sigm((m-mu_m)/k_m), st=sigm((t-mu_t)/k_t)",
        "n_params": 9,
        "params": {
            "a": float(a), "alpha": float(alpha), "mu_m": float(mu_m),
            "k_m": float(k_m), "amp0": float(amp0), "beta": float(beta),
            "mu_t": float(mu_t), "k_t": float(k_t), "C_0": float(C_0),
        },
        "C_residual_std": C_std,
        "Sci_rec_residual_std": Sci_std,
        "Sci_rec_residual_max": Sci_max,
        "unwrap_change_pct": change_pct,
        "plot_path": str(PLOT_PATH),
        "pattern_notes": (
            "Round 1 winner re-run with self-consistent unwrap; "
            "apples-to-apples baseline for Sci_rec metric. "
            "Heatmap residual shows vertical stripes (2018-Q1, 2021-mid-2022, "
            "2024-2026) and a mid-mlat positive band in 2021-2023."
        ),
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
