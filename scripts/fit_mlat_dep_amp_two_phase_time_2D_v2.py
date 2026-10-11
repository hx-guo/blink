#!/usr/bin/env python3
"""Round 2 candidate: mlat_dep_amp_two_phase_time (v2 — self-consistent eval on cache).

Model (12 params):
  C(mlat, t) = a · (1 + α · σ_m) · [1 − amp0 · (1 + β · σ_m) · σ_t1
                                    − amp1 · σ_t2] + C_0
where
  σ_m  = 1 / (1 + exp(-(|mlat| − μ_m)/k_m))
  σ_t1 = 1 / (1 + exp(-(t − μ_t1)/k_t1))   (main decay)
  σ_t2 = 1 / (1 + exp(-(t − μ_t2)/k_t2))   (secondary inflection)

Stage 1: fit on C(|mlat|, t) heatmap (60×108).
Stage 2: self-consistent eval on cache rows
         - read row group 0 from each yearly parquet (2017..2026)
         - look up mlat via AACGM grid (Lat,Lon)
         - compute C_pred per row using fitted model
         - run unwrap_v2 with C_pred AND with C=150 baseline; apply event-balance cap
         - compute Sci_rec_residual = Sci_rec - Sci_obs on rows with Sci_1s>50
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares

MODEL_NAME = "mlat_dep_amp_two_phase_time"
N_PARAMS = 12

NPZ_HEATMAP = "n_below_study/v5_npz/C_2D_heatmap.npz"
AACGM_NPZ   = "n_below_study/aacgm_grid_2020.npz"
CACHE_DIR   = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
YEARS       = list(range(2017, 2027))
L_PER_CYCLE = 16e-6
C_BASELINE  = 150.0
T0          = np.datetime64("2017-06-22")

CACHE_COLS = ['date', 'box', 'det', 'PHO', 'Wide', 'Large', 'Sci_1s', 'L_cycles', 'Dt', 'Lat', 'Lon']
SCI_MIN    = 50
RESID_CAP  = 1000.0


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def model_grid(params, mlat_grid, t_grid):
    """Eval model on outer grid (n_mlat, n_t)."""
    (a, alpha, mu_m, k_m,
     amp0, beta, mu_t1, k_t1,
     amp1, mu_t2, k_t2, C_0) = params
    sm  = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))
    st1 = 1.0 / (1.0 + np.exp(-(t_grid    - mu_t1) / k_t1))
    st2 = 1.0 / (1.0 + np.exp(-(t_grid    - mu_t2) / k_t2))
    A = a * (1.0 + alpha * sm)                                # (n_mlat,)
    amp_eff_main = amp0 * (1.0 + beta * sm)                   # (n_mlat,)
    F = 1.0 - amp_eff_main[:, None] * st1[None, :] - amp1 * st2[None, :]
    return A[:, None] * F + C_0


def model_vec(params, mlat_arr, t_arr):
    """Pointwise eval. mlat_arr, t_arr same shape (e.g. cache rows)."""
    (a, alpha, mu_m, k_m,
     amp0, beta, mu_t1, k_t1,
     amp1, mu_t2, k_t2, C_0) = params
    sm  = 1.0 / (1.0 + np.exp(-(mlat_arr - mu_m) / k_m))
    st1 = 1.0 / (1.0 + np.exp(-(t_arr    - mu_t1) / k_t1))
    st2 = 1.0 / (1.0 + np.exp(-(t_arr    - mu_t2) / k_t2))
    A = a * (1.0 + alpha * sm)
    amp_eff_main = amp0 * (1.0 + beta * sm)
    F = 1.0 - amp_eff_main * st1 - amp1 * st2
    return A * F + C_0


# ----------------------------------------------------------------------------
# Self-consistent unwrap
# ----------------------------------------------------------------------------
def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    """C may be scalar or per-row array."""
    LL = lc * L_PER_CYCLE
    lf = 1.0 - dt / lc
    pred = pho - (wide + (sci + C) * LL) / lf
    n = np.maximum(np.round((pred - large) / 1024.0).astype(int), 0)
    mx = pho - wide
    out = large + n * 1024.0
    ov = out > mx
    if ov.any():
        nm = np.maximum(np.floor((mx - large) / 1024.0).astype(int), 0)
        n_corr = np.where(ov, nm, n)
        out = large + n_corr * 1024.0
    return out


def event_balance_cap(pho, large, wide, sci, lc, dt, C_pred):
    LL = lc * L_PER_CYCLE
    lf = 1.0 - dt / lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C_pred)
    # event-balance: zero-baseline max-large estimate
    mle = pho - ((sci + 0.0) * LL + wide) / lf
    n1 = np.round((lv1 - large) / 1024.0).astype(int)
    nmax = np.maximum(np.floor((mle - large) / 1024.0).astype(int), 0)
    n_final = np.where(n1 > nmax, nmax, n1)
    lv_final = large + n_final * 1024.0
    return lv_final


# ----------------------------------------------------------------------------
# AACGM mlat lookup (vectorized nearest-neighbor)
# ----------------------------------------------------------------------------
def build_mlat_lookup():
    z = np.load(AACGM_NPZ)
    lat_grid = z['lat_grid']  # (91,)  -45..45
    lon_grid = z['lon_grid']  # (181,) 0..360
    mlat = z['mlat']          # (91, 181)
    return lat_grid, lon_grid, mlat


def lookup_mlat(lat, lon, lat_grid, lon_grid, mlat):
    """Vectorized nearest-neighbor. Returns |mlat| in deg."""
    # clip
    lat_c = np.clip(lat, lat_grid.min(), lat_grid.max())
    lon_c = np.mod(lon, 360.0)
    # bin index via spacing assumption (1-deg lat, 2-deg lon based on shapes 91/181)
    di = np.round((lat_c - lat_grid[0]) / (lat_grid[1] - lat_grid[0])).astype(int)
    dj = np.round((lon_c - lon_grid[0]) / (lon_grid[1] - lon_grid[0])).astype(int)
    di = np.clip(di, 0, mlat.shape[0] - 1)
    dj = np.clip(dj, 0, mlat.shape[1] - 1)
    return np.abs(mlat[di, dj])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    z = np.load(NPZ_HEATMAP)
    C_data = z["C_med"]
    n_data = z["C_n"]
    months = z["months"]
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])  # (60,)
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t_years = ((month_dt - T0).astype("timedelta64[D]").astype(float)) / 365.25

    mask = n_data > 200
    print(f"valid heatmap bins: {mask.sum()}/{mask.size}")

    # --- Stage 1: fit on heatmap ---
    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = model_grid(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # 12 params: a, alpha, mu_m, k_m, amp0, beta, mu_t1, k_t1, amp1, mu_t2, k_t2, C_0
    p0 = [
        200.0,   # a
        2.0,     # alpha
        45.0,    # mu_m
        6.0,     # k_m
        0.15,    # amp0
        1.5,     # beta
        5.0,     # mu_t1   (main decay around 2022)
        1.0,     # k_t1
        0.05,    # amp1
        3.0,     # mu_t2   (secondary inflection around 2020)
        1.0,     # k_t2
        -50.0,   # C_0
    ]
    lo = [10.0, 0.0, 30.0, 0.5, 0.0, -2.0, 0.5, 0.2, -0.5, 0.0, 0.2, -200.0]
    hi = [500.0, 20.0, 60.0, 15.0, 1.5, 5.0, 9.0, 8.0, 1.5, 9.0, 8.0, 200.0]

    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=5000)
    p = res.x
    (a, alpha, mu_m, k_m,
     amp0, beta, mu_t1, k_t1,
     amp1, mu_t2, k_t2, C_0) = p
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    print(f"a={a:.2f}, alpha={alpha:.3f}, mu_m={mu_m:.2f}, k_m={k_m:.2f}")
    print(f"amp0={amp0:.3f}, beta={beta:.3f}, mu_t1={mu_t1:.2f}, k_t1={k_t1:.2f}")
    print(f"amp1={amp1:.3f}, mu_t2={mu_t2:.2f}, k_t2={k_t2:.2f}, C_0={C_0:+.2f}")

    C_pred_grid = model_grid(p, mlat_centers, t_years)
    resid_grid = (C_data - C_pred_grid) * mask
    C_residual_std = float(np.std(resid_grid[mask]))
    print(f"\nC heatmap residual std (winning-ish): {C_residual_std:.3f} cnt/s")

    # --- Stage 2: self-consistent eval on cache rows ---
    lat_grid, lon_grid, mlat_map = build_mlat_lookup()

    Sci_rec_residuals = []
    Sci_rec_residuals_base = []
    n_unwrap_change_total = 0
    n_rows_total = 0

    for yr in YEARS:
        path = f"{CACHE_DIR}/clean_relaxed_{yr}.parquet"
        if not Path(path).exists():
            print(f"  skip missing {path}")
            continue
        pf = pq.ParquetFile(path)
        if pf.num_row_groups == 0:
            print(f"  skip empty {path}")
            continue
        tbl = pf.read_row_group(0, columns=CACHE_COLS)
        df = tbl.to_pandas()
        n = len(df)
        if n == 0:
            continue

        pho   = df['PHO'].to_numpy(dtype=np.float64)
        large = df['Large'].to_numpy(dtype=np.float64)
        wide  = df['Wide'].to_numpy(dtype=np.float64)
        sci   = df['Sci_1s'].to_numpy(dtype=np.float64)
        lc    = df['L_cycles'].to_numpy(dtype=np.float64)
        dt    = df['Dt'].to_numpy(dtype=np.float64)
        lat   = df['Lat'].to_numpy(dtype=np.float64)
        lon   = df['Lon'].to_numpy(dtype=np.float64)
        date_s = df['date'].to_numpy()

        # t_years per row from date string (YYYY-MM-DD)
        dt64 = np.array(date_s, dtype='datetime64[D]')
        t_row = ((dt64 - T0).astype("timedelta64[D]").astype(float)) / 365.25
        mlat_row = lookup_mlat(lat, lon, lat_grid, lon_grid, mlat_map)
        C_pred_row = model_vec(p, mlat_row, t_row)

        # baseline (C=150)
        lv_base = event_balance_cap(pho, large, wide, sci, lc, dt, C_BASELINE)
        # model C
        lv_model = event_balance_cap(pho, large, wide, sci, lc, dt, C_pred_row)

        LL = lc * L_PER_CYCLE
        lf = 1.0 - dt / lc
        # Sci_rec for model
        Sci_rec = (pho - lv_model) * lf / LL - wide / LL - C_pred_row
        Sci_rec_base = (pho - lv_base) * lf / LL - wide / LL - C_BASELINE

        resid = Sci_rec - sci
        resid_base = Sci_rec_base - sci

        # filter rows: sci>50, finite resid, |resid|<1000
        sel = (sci > SCI_MIN) & np.isfinite(resid) & (np.abs(resid) < RESID_CAP)
        Sci_rec_residuals.append(resid[sel])
        sel_b = (sci > SCI_MIN) & np.isfinite(resid_base) & (np.abs(resid_base) < RESID_CAP)
        Sci_rec_residuals_base.append(resid_base[sel_b])

        # unwrap change pct -- both passes need lf>0, defined rows
        sel_ok = np.isfinite(lv_model) & np.isfinite(lv_base) & (sci > SCI_MIN)
        n_unwrap_change_total += int(((lv_model != lv_base) & sel_ok).sum())
        n_rows_total += int(sel_ok.sum())
        print(f"  {yr}: rows={n:,}, sel={sel.sum():,}, "
              f"resid_std={np.std(resid[sel]):.2f}, "
              f"|change|={(lv_model != lv_base)[sel_ok].sum():,} / {sel_ok.sum():,}")

    res_arr = np.concatenate(Sci_rec_residuals) if Sci_rec_residuals else np.array([])
    res_base_arr = np.concatenate(Sci_rec_residuals_base) if Sci_rec_residuals_base else np.array([])
    if res_arr.size == 0:
        Sci_rec_residual_std = float('nan')
        Sci_rec_residual_max = float('nan')
    else:
        Sci_rec_residual_std = float(np.std(res_arr))
        Sci_rec_residual_max = float(np.max(np.abs(res_arr)))
    unwrap_change_pct = (100.0 * n_unwrap_change_total / n_rows_total) if n_rows_total else 0.0

    print(f"\n=== Self-consistent eval over {res_arr.size:,} cache rows ===")
    print(f"  Sci_rec_residual_std = {Sci_rec_residual_std:.3f} cnt/s")
    print(f"  Sci_rec_residual_max = {Sci_rec_residual_max:.2f} cnt/s")
    print(f"  unwrap_change_pct    = {unwrap_change_pct:.4f}% ({n_unwrap_change_total:,}/{n_rows_total:,})")
    print(f"  (baseline C=150)     std={np.std(res_base_arr):.3f}  max={np.max(np.abs(res_base_arr)):.2f}")

    # --- Plots ---
    Path("plots").mkdir(exist_ok=True)
    out_png = f"plots/fit_{MODEL_NAME}_2D_v2.png"

    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    fig.suptitle(
        f"{MODEL_NAME} (v2 self-consistent) — 12 params\n"
        f"a={a:.0f}, alpha={alpha:.2f}, mu_m={mu_m:.1f}, k_m={k_m:.1f}, "
        f"amp0={amp0:.2f}, beta={beta:.2f}, mu_t1={mu_t1:.1f}, k_t1={k_t1:.1f}, "
        f"amp1={amp1:.2f}, mu_t2={mu_t2:.1f}, k_t2={k_t2:.1f}, C0={C_0:+.0f}\n"
        f"C_resid_std={C_residual_std:.2f} cnt/s | "
        f"Sci_rec_resid_std={Sci_rec_residual_std:.2f} cnt/s | "
        f"unwrap_change={unwrap_change_pct:.3f}%",
        fontsize=11, fontweight='bold')

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    C_data_m = C_data.copy().astype(float); C_data_m[~mask] = np.nan
    C_pred_m = C_pred_grid.copy().astype(float); C_pred_m[~mask] = np.nan
    resid_m  = resid_grid.copy().astype(float); resid_m[~mask] = np.nan

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
             'C_data (cnt/s)', '1. DATA — mean C')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL — two-phase time fit')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL — data - model')

    axes[3].hist(res_arr, bins=200, range=(-200, 200), color='steelblue',
                 alpha=0.7, label=f"model C  (std={Sci_rec_residual_std:.2f})")
    axes[3].hist(res_base_arr, bins=200, range=(-200, 200),
                 histtype='step', color='crimson', linewidth=1.4,
                 label=f"baseline C=150  (std={np.std(res_base_arr):.2f})")
    axes[3].set_yscale('log')
    axes[3].set_xlim(-200, 200)
    axes[3].set_xlabel("Sci_rec - Sci_obs (cnt/s)", fontsize=11)
    axes[3].set_ylabel("count (log)", fontsize=11)
    axes[3].set_title(f"4. CACHE-LEVEL — Sci_rec residual (N={res_arr.size:,})", fontsize=11)
    axes[3].legend(loc='upper right')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out_png}")

    summary = {
        "model_name": MODEL_NAME,
        "n_params": N_PARAMS,
        "params": {
            "a": float(a), "alpha": float(alpha),
            "mu_m": float(mu_m), "k_m": float(k_m),
            "amp0": float(amp0), "beta": float(beta),
            "mu_t1": float(mu_t1), "k_t1": float(k_t1),
            "amp1": float(amp1),
            "mu_t2": float(mu_t2), "k_t2": float(k_t2),
            "C_0": float(C_0),
        },
        "C_residual_std": C_residual_std,
        "Sci_rec_residual_std": Sci_rec_residual_std,
        "Sci_rec_residual_max": Sci_rec_residual_max,
        "unwrap_change_pct": unwrap_change_pct,
        "n_cache_rows": int(res_arr.size),
        "plot_path": out_png,
    }
    print("\n=== JSON summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
