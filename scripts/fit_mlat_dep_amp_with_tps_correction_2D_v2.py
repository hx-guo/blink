#!/usr/bin/env python3
"""Round-2 candidate: mlat_dep_amp + tensor-product smooth-basis correction.

Model:
  C(mlat, t) = a*(1 + alpha*sigm((mlat-mu_m)/k_m))
               * (1 - amp0*(1 + beta*sigm((mlat-mu_m)/k_m))*sigm((t-mu_t)/k_t))
               + C0
               + sum_{i,j=1..3} w_ij * B_i(mlat) * B_j(t)

  9 base parameters + 9 correction weights = 18 params.
  B_i, B_j are sech^2 radial bumps centered on 3x3 fixed knots.

After fitting on the 2D C heatmap, the model is evaluated self-consistently on
sampled cache rows (one row group per yearly parquet) using unwrap_v2 with the
fitted C(mlat,t) and an event-balance cap, then compared to baseline C=150 to
report unwrap_change_pct and the Sci_rec residual stats.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
import pyarrow.parquet as pq

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"
AACGM_NPZ = "n_below_study/aacgm_grid_2020.npz"
CACHE_DIR = Path("/Volumes/Graphite/work/blink/blink_clean_relaxed")
PLOT_DIR = Path("plots")

L_CYCLES_TO_SEC = 16e-6
N_KNOTS_M = 3
N_KNOTS_T = 3
NEEDED_COLS = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s",
               "L_cycles", "Dt", "Lat", "Lon"]


def sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_basis(x, knots, width):
    """sech^2 radial bumps. Returns shape (len(x), n_knots)."""
    z = (x[:, None] - knots[None, :]) / width
    return 1.0 / np.cosh(z) ** 2


def model_C_full(params, mlat, t, B_m, B_t):
    """C model evaluated on grid (mlat_centers, t_centers).

    params = [a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0, w11..w33]
    """
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params[:9]
    W = params[9:].reshape(N_KNOTS_M, N_KNOTS_T)

    sm = sigm((mlat - mu_m) / k_m)              # (n_m,)
    st = sigm((t - mu_t) / k_t)                  # (n_t,)
    A = a * (1.0 + alpha * sm)
    amp_eff = amp0 * (1.0 + beta * sm)
    base = A[:, None] * (1.0 - amp_eff[:, None] * st[None, :]) + C_0
    corr = B_m @ W @ B_t.T                       # (n_m, n_t)
    return base + corr


def predict_C_rowwise(params, mlat_row, t_row, mlat_knots, t_knots, width_m, width_t):
    """Per-row C prediction (mlat_row, t_row 1D arrays of same length)."""
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params[:9]
    W = params[9:].reshape(N_KNOTS_M, N_KNOTS_T)

    sm = sigm((mlat_row - mu_m) / k_m)
    st = sigm((t_row - mu_t) / k_t)
    A = a * (1.0 + alpha * sm)
    amp_eff = amp0 * (1.0 + beta * sm)
    base = A * (1.0 - amp_eff * st) + C_0

    # B_m_row: shape (N, K_m); B_t_row: shape (N, K_t)
    B_m_row = make_basis(mlat_row, mlat_knots, width_m)
    B_t_row = make_basis(t_row, t_knots, width_t)
    # element-wise: sum_{i,j} w_ij * B_m[n,i] * B_t[n,j]
    corr = np.einsum("ni,ij,nj->n", B_m_row, W, B_t_row)
    return base + corr


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    LL = lc * L_CYCLES_TO_SEC
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


def unwrap_with_balance_cap(pho, large, wide, sci, lc, dt, C):
    """unwrap_v2 then event-balance cap (using sci=0 max-large estimate)."""
    LL = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dt / lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C)
    mle = pho - ((0.0) * LL + wide) / lf
    n1 = np.round((lv1 - large) / 1024.0).astype(int)
    nmax = np.maximum(np.floor((mle - large) / 1024.0).astype(int), 0)
    lv_final = large + np.where(n1 > nmax, nmax, n1) * 1024.0
    return lv_final


def fit_heatmap():
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
    print(f"valid bins: {mask.sum()}/{mask.size}")
    C_data_clean = np.where(mask, C_data, 0.0)

    mlat_lo, mlat_hi = float(mlat_centers.min()), float(mlat_centers.max())
    t_lo, t_hi = float(t_years.min()), float(t_years.max())
    mlat_knots = np.linspace(mlat_lo, mlat_hi, N_KNOTS_M)
    t_knots = np.linspace(t_lo, t_hi, N_KNOTS_T)
    width_m = mlat_knots[1] - mlat_knots[0]
    width_t = t_knots[1] - t_knots[0]
    print(f"mlat knots: {mlat_knots}, t knots: {t_knots}")
    print(f"width_m={width_m:.3f} deg, width_t={width_t:.3f} yr")

    B_m = make_basis(mlat_centers, mlat_knots, width_m)
    B_t = make_basis(t_years, t_knots, width_t)

    n_base = 9
    n_corr = N_KNOTS_M * N_KNOTS_T
    n_params = n_base + n_corr
    print(f"n_params = {n_params} (9 base + {n_corr} correction)")

    p0_base = [202.6, 1.69, 44.46, 6.33, 0.152, 1.69, 5.25, 1.00, -79.3]
    p0 = np.array(p0_base + [0.0] * n_corr, dtype=float)

    lo_base = [10.0, 0.5, 30.0, 0.5, 0.0, -2.0, 0.5, 0.2, -200.0]
    hi_base = [500.0, 20.0, 60.0, 15.0, 1.5, 5.0, 8.0, 8.0, 200.0]
    lo = np.array(lo_base + [-500.0] * n_corr)
    hi = np.array(hi_base + [+500.0] * n_corr)

    def residual(params):
        C_pred = model_C_full(params, mlat_centers, t_years, B_m, B_t)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    params_fit = res.x

    C_pred = model_C_full(params_fit, mlat_centers, t_years, B_m, B_t)
    valid_resid = (C_data - C_pred)[mask]
    C_residual_std = float(np.std(valid_resid))
    C_residual_max = float(np.max(np.abs(valid_resid)))
    mean_C = float(np.mean(C_data[mask]))
    print(f"\n=== C heatmap residual ===")
    print(f"  std = {C_residual_std:.3f} cnt/s ({C_residual_std/mean_C*100:.2f}% of mean)")
    print(f"  max |resid| = {C_residual_max:.2f}")

    base = params_fit[:9]
    W = params_fit[9:].reshape(N_KNOTS_M, N_KNOTS_T)
    print("\n=== Fitted base params ===")
    print(f"  a={base[0]:.2f} alpha={base[1]:.3f} mu_m={base[2]:.2f} k_m={base[3]:.2f}")
    print(f"  amp0={base[4]:.3f} beta={base[5]:.3f} mu_t={base[6]:.2f} k_t={base[7]:.2f} C0={base[8]:.2f}")
    print("=== Correction W (rows=mlat knots, cols=t knots) ===")
    for i in range(N_KNOTS_M):
        row = "  ".join(f"{W[i,j]:+8.3f}" for j in range(N_KNOTS_T))
        print(f"  mlat={mlat_knots[i]:5.1f}:  {row}")

    return {
        "params_fit": params_fit,
        "mlat_knots": mlat_knots,
        "t_knots": t_knots,
        "width_m": width_m,
        "width_t": width_t,
        "C_data": C_data,
        "C_pred": C_pred,
        "mask": mask,
        "edges": edges,
        "mlat_centers": mlat_centers,
        "month_dt": month_dt,
        "t_years": t_years,
        "C_residual_std": C_residual_std,
        "C_residual_max": C_residual_max,
        "mean_C": mean_C,
        "t0": t0,
    }


def evaluate_cache(params_fit, mlat_knots, t_knots, width_m, width_t, t0):
    """Self-consistent unwrap eval on sampled cache rows."""
    print("\n=== Self-consistent cache eval ===")
    aacgm = np.load(AACGM_NPZ)
    lat_grid = aacgm["lat_grid"]
    lon_grid = aacgm["lon_grid"]
    mlat_grid = np.abs(aacgm["mlat"])  # absolute |mlat|

    def lookup_mlat(lat, lon):
        # Clamp/wrap to grid range; Lon expected [0,360)
        lat = np.clip(lat, lat_grid[0], lat_grid[-1])
        lon = np.mod(lon, 360.0)
        # nearest indices
        i = np.clip(np.round((lat - lat_grid[0]) / (lat_grid[1] - lat_grid[0])).astype(int),
                    0, len(lat_grid) - 1)
        j = np.clip(np.round((lon - lon_grid[0]) / (lon_grid[1] - lon_grid[0])).astype(int),
                    0, len(lon_grid) - 1)
        return mlat_grid[i, j]

    all_resid = []
    all_sci = []
    n_total = 0
    n_changed = 0

    for year in range(2017, 2027):
        path = CACHE_DIR / f"clean_relaxed_{year}.parquet"
        if not path.exists():
            print(f"  [skip] {path.name} (missing)")
            continue
        pf = pq.ParquetFile(path)
        if pf.num_row_groups == 0:
            continue
        tbl = pf.read_row_group(0, columns=NEEDED_COLS)
        df = tbl.to_pandas()
        n_rows_raw = len(df)

        # Compute |mlat| from Lat/Lon
        lat = df["Lat"].to_numpy(dtype=np.float64)
        lon = df["Lon"].to_numpy(dtype=np.float64)
        mlat = lookup_mlat(lat, lon)

        # Compute t_years
        dates = df["date"].to_numpy()
        if dates.dtype.kind == "O":
            dates_dt = np.array([np.datetime64(str(d)) for d in dates])
        else:
            dates_dt = dates.astype("datetime64[D]")
        t_years = ((dates_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25

        pho = df["PHO"].to_numpy(dtype=np.float64)
        wide = df["Wide"].to_numpy(dtype=np.float64)
        large = df["Large"].to_numpy(dtype=np.float64)
        sci = df["Sci_1s"].to_numpy(dtype=np.float64)
        lc = df["L_cycles"].to_numpy(dtype=np.float64)
        dt_ = df["Dt"].to_numpy(dtype=np.float64)

        # Avoid bad rows: lc<=0 or dt>=lc
        good_phys = (lc > 0) & (dt_ < lc) & np.isfinite(pho) & np.isfinite(wide) \
                    & np.isfinite(large) & np.isfinite(sci) & np.isfinite(mlat) \
                    & np.isfinite(t_years)
        if good_phys.sum() == 0:
            print(f"  [{year}] no physically valid rows in row_group 0")
            continue

        # Subset for safety/speed
        pho_g = pho[good_phys]
        wide_g = wide[good_phys]
        large_g = large[good_phys]
        sci_g = sci[good_phys]
        lc_g = lc[good_phys]
        dt_g = dt_[good_phys]
        mlat_g = mlat[good_phys]
        t_g = t_years[good_phys]

        # Predict C per row
        C_pred = predict_C_rowwise(params_fit, mlat_g, t_g,
                                   mlat_knots, t_knots, width_m, width_t)

        # Self-consistent + event-balance cap
        lv_final = unwrap_with_balance_cap(pho_g, large_g, wide_g, sci_g, lc_g, dt_g, C_pred)
        # Baseline C=150
        lv_baseline = unwrap_with_balance_cap(pho_g, large_g, wide_g, sci_g, lc_g, dt_g, 150.0)
        changed = lv_final != lv_baseline

        # Compute Sci_rec
        LL = lc_g * L_CYCLES_TO_SEC
        lf = 1.0 - dt_g / lc_g
        Sci_rec = (pho_g - lv_final) * lf / LL - wide_g / LL - C_pred
        resid = Sci_rec - sci_g

        # Filter
        keep = (sci_g > 50) & np.isfinite(resid) & (np.abs(resid) < 1000.0)
        all_resid.append(resid[keep])
        all_sci.append(sci_g[keep])
        n_total += int(keep.sum())
        n_changed += int(changed[keep].sum())
        print(f"  [{year}] rg0 raw={n_rows_raw}, good_phys={good_phys.sum()}, "
              f"kept={keep.sum()}, changed={changed[keep].sum()} ({changed[keep].mean()*100:.2f}%)")

    if not all_resid or n_total == 0:
        print("  [warn] no kept rows across all years!")
        return 0.0, 0.0, 0.0

    all_resid = np.concatenate(all_resid)
    Sci_rec_residual_std = float(np.std(all_resid))
    Sci_rec_residual_max = float(np.max(np.abs(all_resid)))
    unwrap_change_pct = 100.0 * n_changed / max(n_total, 1)

    print(f"\n=== Cache eval ===")
    print(f"  n_total kept (sci>50, |resid|<1000): {n_total}")
    print(f"  Sci_rec residual std = {Sci_rec_residual_std:.3f} cnt/s")
    print(f"  Sci_rec residual max = {Sci_rec_residual_max:.3f} cnt/s")
    print(f"  unwrap_change_pct vs C=150: {unwrap_change_pct:.3f}%")

    return Sci_rec_residual_std, Sci_rec_residual_max, unwrap_change_pct, all_resid


def make_plot(fit_state, all_resid, params_fit, Sci_rec_residual_std,
              C_residual_std, unwrap_change_pct, out_path):
    C_data = fit_state["C_data"]
    C_pred = fit_state["C_pred"]
    mask = fit_state["mask"]
    edges = fit_state["edges"]
    month_dt = fit_state["month_dt"]

    C_data_m = C_data.copy().astype(float); C_data_m[~mask] = np.nan
    C_pred_m = C_pred.copy().astype(float); C_pred_m[~mask] = np.nan
    resid_m = (C_data - C_pred).astype(float); resid_m[~mask] = np.nan

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    title = ("mlat_dep_amp + tps_correction (18 params)\n"
             f"C_resid_std = {C_residual_std:.2f} cnt/s, "
             f"Sci_rec_resid_std = {Sci_rec_residual_std:.2f} cnt/s, "
             f"unwrap_change_pct = {unwrap_change_pct:.2f}%")
    fig.suptitle(title, fontsize=12, fontweight='bold')

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data,
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel("|mlat| (deg)", fontsize=10)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_m, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA - C (cnt/s)')
    plot_pcm(axes[1], C_pred_m, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL - mlat_dep_amp + tps')
    plot_pcm(axes[2], resid_m, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL +/- 30 cnt/s')
    axes[2].set_xlabel("date", fontsize=10)

    ax = axes[3]
    if all_resid is not None and len(all_resid) > 0:
        ax.hist(all_resid, bins=np.linspace(-200, 200, 201),
                color='steelblue', edgecolor='black', alpha=0.85)
        ax.set_yscale('log')
        ax.set_xlim(-200, 200)
        ax.set_xlabel("Sci_rec - Sci_obs (cnt/s)", fontsize=10)
        ax.set_ylabel("rows (log)", fontsize=10)
        ax.set_title(f"4. Sci_rec residual histogram (N={len(all_resid)}, "
                     f"std={Sci_rec_residual_std:.2f})", fontsize=10)
        ax.axvline(0, color='red', lw=1, ls='--')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no cache rows", ha='center', va='center')

    plt.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out_path}")


def main():
    fit_state = fit_heatmap()
    params_fit = fit_state["params_fit"]
    mlat_knots = fit_state["mlat_knots"]
    t_knots = fit_state["t_knots"]
    width_m = fit_state["width_m"]
    width_t = fit_state["width_t"]
    t0 = fit_state["t0"]

    sci_std, sci_max, unwrap_change_pct, all_resid = evaluate_cache(
        params_fit, mlat_knots, t_knots, width_m, width_t, t0)

    out_path = PLOT_DIR / "fit_mlat_dep_amp_with_tps_correction_2D_v2.png"
    make_plot(fit_state, all_resid, params_fit, sci_std,
              fit_state["C_residual_std"], unwrap_change_pct, out_path)

    print("\n=== Final summary ===")
    print(f"  C_residual_std       = {fit_state['C_residual_std']:.3f}")
    print(f"  Sci_rec_residual_std = {sci_std:.3f}  (WINNING METRIC)")
    print(f"  Sci_rec_residual_max = {sci_max:.3f}")
    print(f"  unwrap_change_pct    = {unwrap_change_pct:.3f}%")
    print(f"  plot_path            = {out_path}")


if __name__ == "__main__":
    main()
