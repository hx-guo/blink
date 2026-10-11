#!/usr/bin/env python3
"""Fit the mlat_dep_amp_dual_mlat_sigmoid 2D model with self-consistent unwrap eval.

Model (12 params):
  C(mlat, t) = a * (1 + alpha1*sigm((m-mu1)/k1) + alpha2*sigm((m-mu2)/k2))
                 * (1 - amp0*(1 + beta*sigm((m-mu1)/k1)) * sigm((t-mu_t)/k_t))
               + C_0

Two mlat sigmoids (one low-transition, one high-transition) for finer shape.
Time-decay amplitude only modulated by the low (primary) sigmoid via beta.

Free params: a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, beta, mu_t, k_t, C_0
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
import pyarrow.parquet as pq

NPZ = "/Users/skyair/Developer/ihep/blink/n_below_study/v5_npz/C_2D_heatmap.npz"
CACHE_DIR = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
AACGM_NPZ = "/Users/skyair/Developer/ihep/blink/n_below_study/aacgm_grid_2020.npz"
L_CYCLES_TO_SEC = 16e-6
T0 = np.datetime64("2017-06-22")
MODEL_NAME = "mlat_dep_amp_dual_mlat_sigmoid"


def sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


def C_model(params, mlat, t_years):
    """mlat, t_years either both 1D for outer product, or both already broadcast.

    If mlat shape (M,) and t_years shape (T,), returns (M, T).
    If both already broadcast / same shape, returns same shape (per-row mode).
    """
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, beta, mu_t, k_t, C_0 = params
    mlat = np.asarray(mlat, dtype=np.float64)
    t_years = np.asarray(t_years, dtype=np.float64)
    if mlat.ndim == 1 and t_years.ndim == 1 and mlat.shape != t_years.shape:
        # outer-product mode (heatmap grid)
        s1 = sigm((mlat - mu1) / k1)        # (M,)
        s2 = sigm((mlat - mu2) / k2)        # (M,)
        A = a * (1.0 + alpha1 * s1 + alpha2 * s2)      # (M,)
        amp_eff = amp0 * (1.0 + beta * s1)             # (M,)
        st = sigm((t_years - mu_t) / k_t)              # (T,)
        F = 1.0 - amp_eff[:, None] * st[None, :]       # (M, T)
        return A[:, None] * F + C_0                    # (M, T)
    # per-row mode (both same shape)
    s1 = sigm((mlat - mu1) / k1)
    s2 = sigm((mlat - mu2) / k2)
    st = sigm((t_years - mu_t) / k_t)
    A = a * (1.0 + alpha1 * s1 + alpha2 * s2)
    amp_eff = amp0 * (1.0 + beta * s1)
    return A * (1.0 - amp_eff * st) + C_0


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


def unwrap_with_eb_cap(pho, large, wide, sci, lc, dt, C):
    """Run unwrap_v2 and apply event-balance cap."""
    LL = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dt / lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C)
    mle = pho - ((sci + 0.0) * LL + wide) / lf  # max-large-event (sci-only contributes)
    n1 = np.round((lv1 - large) / 1024.0).astype(int)
    nmax = np.maximum(np.floor((mle - large) / 1024.0).astype(int), 0)
    lv_final = large + np.where(n1 > nmax, nmax, n1) * 1024.0
    return lv_final


def fit_heatmap():
    """Fit on the 2D heatmap; return params + residual stats + arrays for plotting."""
    z = np.load(NPZ)
    C_data = z["C_med"]                # (60, 108)
    n_data = z["C_n"]
    months = z["months"]
    edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t_years = ((month_dt - T0).astype("timedelta64[D]").astype(float)) / 365.25

    mask = n_data > 200
    print(f"valid bins: {mask.sum()}/{mask.size}")
    C_data_clean = np.where(mask, C_data, 0.0)

    def residual(params):
        C_pred = C_model(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # Initial guesses (informed by round-1 winner mu_m=44.5, k_m=6.3):
    # Place first sigmoid at low-transition (mid-mlat), second at high-mlat ramp.
    p0 = [
        180.0,    # a
        0.8,      # alpha1
        35.0,     # mu1 (lower transition)
        4.0,      # k1
        1.0,      # alpha2
        50.0,     # mu2 (higher transition)
        4.0,      # k2
        0.15,     # amp0
        1.5,      # beta
        5.0,      # mu_t
        1.0,      # k_t
        -50.0,    # C_0
    ]
    lo = [10.0, -2.0, 20.0, 0.5, -2.0, 40.0, 0.5, 0.0, -3.0, 0.5, 0.2, -200.0]
    hi = [500.0, 5.0, 45.0, 15.0, 5.0, 60.0, 15.0, 1.5, 5.0, 8.0, 8.0, 100.0]

    res = least_squares(residual, p0, bounds=(lo, hi),
                        method='trf', max_nfev=5000)
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")

    pnames = ["a", "alpha1", "mu1", "k1", "alpha2", "mu2", "k2",
              "amp0", "beta", "mu_t", "k_t", "C_0"]
    fitted = dict(zip(pnames, [float(v) for v in res.x]))
    print("\n=== Fitted parameters (12 free) ===")
    for k, v in fitted.items():
        print(f"  {k:8s} = {v:9.3f}")

    C_pred = C_model(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid_resid = resid[mask]
    mean_C = float(np.mean(C_data[mask]))
    std_r = float(np.std(valid_resid))
    max_abs_r = float(np.max(np.abs(valid_resid)))
    std_pct = std_r / mean_C * 100.0
    print(f"\n=== Heatmap residual ({mask.sum()} valid bins) ===")
    print(f"  std:    {std_r:.3f} cnt/s ({std_pct:.2f}% of mean {mean_C:.1f})")
    print(f"  max:    {max_abs_r:.2f} cnt/s")

    return {
        "params": res.x,
        "params_dict": fitted,
        "C_data": C_data, "C_pred": C_pred, "mask": mask,
        "month_dt": month_dt, "mlat_edges": edges, "mlat_centers": mlat_centers,
        "t_years": t_years,
        "C_residual_std": std_r, "C_residual_pct": std_pct,
        "C_residual_max": max_abs_r,
    }


def load_aacgm():
    z = np.load(AACGM_NPZ)
    return z["lat_grid"], z["lon_grid"], z["mlat"]


def mlat_lookup(lat, lon, lat_grid, lon_grid, mlat_table):
    """Nearest-neighbor lookup of |mlat| from lat/lon arrays."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64) % 360.0
    li = np.clip(np.searchsorted(lat_grid, lat), 1, len(lat_grid) - 1)
    lo = np.clip(np.searchsorted(lon_grid, lon), 1, len(lon_grid) - 1)
    # pick the closer of li-1 and li
    li_lo = li - 1
    pick_hi = np.abs(lat - lat_grid[li]) < np.abs(lat - lat_grid[li_lo])
    li_use = np.where(pick_hi, li, li_lo)
    lo_lo = lo - 1
    pick_hi = np.abs(lon - lon_grid[lo]) < np.abs(lon - lon_grid[lo_lo])
    lo_use = np.where(pick_hi, lo, lo_lo)
    return np.abs(mlat_table[li_use, lo_use])


def cache_eval(params):
    """Read one row group per year file, evaluate Sci_rec residual."""
    lat_grid, lon_grid, mlat_tab = load_aacgm()
    years = list(range(2017, 2027))
    cols = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s",
            "L_cycles", "Dt", "Lat", "Lon"]

    all_resid = []
    all_changed = []
    n_total = 0
    n_kept = 0

    for y in years:
        f = f"{CACHE_DIR}/clean_relaxed_{y}.parquet"
        if not Path(f).exists():
            print(f"  missing {f}, skip")
            continue
        pf = pq.ParquetFile(f)
        if pf.num_row_groups == 0:
            continue
        tbl = pf.read_row_group(0, columns=cols)
        df = tbl.to_pandas()
        n_total += len(df)

        # Compute t_years. date is a string like '2020-01-01'.
        dates = np.array(df["date"].astype(str).to_numpy(), dtype="datetime64[D]")
        t_years = ((dates - T0).astype("timedelta64[D]").astype(float)) / 365.25

        # mlat lookup
        lat = df["Lat"].to_numpy(dtype=np.float64)
        lon = df["Lon"].to_numpy(dtype=np.float64)
        mlat_abs = mlat_lookup(lat, lon, lat_grid, lon_grid, mlat_tab)

        # Per-row C_pred from fitted model
        C_pred = C_model(params, mlat_abs, t_years)

        pho = df["PHO"].to_numpy(dtype=np.float64)
        large = df["Large"].to_numpy(dtype=np.float64)
        wide = df["Wide"].to_numpy(dtype=np.float64)
        sci = df["Sci_1s"].to_numpy(dtype=np.float64)
        lc = df["L_cycles"].to_numpy(dtype=np.float64)
        dt = df["Dt"].to_numpy(dtype=np.float64)

        # Avoid divide-by-zero & non-physical rows
        valid = (lc > 0) & np.isfinite(pho) & np.isfinite(large) & np.isfinite(wide) & np.isfinite(sci) & np.isfinite(lc) & np.isfinite(dt) & np.isfinite(C_pred)
        if not valid.any():
            continue

        pho_v = pho[valid]; large_v = large[valid]; wide_v = wide[valid]
        sci_v = sci[valid]; lc_v = lc[valid]; dt_v = dt[valid]
        C_pred_v = C_pred[valid]
        LL = lc_v * L_CYCLES_TO_SEC
        lf = 1.0 - dt_v / lc_v

        # Self-consistent unwrap with event-balance cap
        lv_final = unwrap_with_eb_cap(pho_v, large_v, wide_v, sci_v, lc_v, dt_v, C_pred_v)
        Sci_rec = (pho_v - lv_final) * lf / LL - wide_v / LL - C_pred_v
        resid = Sci_rec - sci_v

        # Baseline: same unwrap with C=150 to measure change
        lv_baseline = unwrap_with_eb_cap(pho_v, large_v, wide_v, sci_v, lc_v, dt_v, 150.0)
        changed = lv_final != lv_baseline

        # Filter
        keep = (sci_v > 50) & np.isfinite(resid) & (np.abs(resid) < 1000.0)
        all_resid.append(resid[keep])
        all_changed.append(changed[keep])
        n_kept += int(keep.sum())
        print(f"  {y}: rows={len(df)}, valid={valid.sum()}, kept={int(keep.sum())}, "
              f"this-yr std={float(np.std(resid[keep])):.2f}, change%={100*changed[keep].mean():.2f}")

    if not all_resid:
        return {"Sci_rec_residual_std": float("nan"),
                "Sci_rec_residual_max": float("nan"),
                "unwrap_change_pct": float("nan"),
                "Sci_rec_residual_arr": np.array([])}
    R = np.concatenate(all_resid)
    Ch = np.concatenate(all_changed)
    return {
        "Sci_rec_residual_std": float(np.std(R)),
        "Sci_rec_residual_max": float(np.max(np.abs(R))),
        "Sci_rec_residual_mean": float(np.mean(R)),
        "unwrap_change_pct": float(100.0 * Ch.mean()),
        "Sci_rec_residual_arr": R,
        "n_kept": n_kept, "n_total": n_total,
    }


def make_plot(heatmap_out, cache_out, plot_path):
    C_data = heatmap_out["C_data"].copy().astype(float)
    C_pred = heatmap_out["C_pred"].copy().astype(float)
    mask = heatmap_out["mask"]
    C_data[~mask] = np.nan
    C_pred[~mask] = np.nan
    resid = (heatmap_out["C_data"] - heatmap_out["C_pred"]).copy().astype(float)
    resid[~mask] = np.nan

    month_dt = heatmap_out["month_dt"]
    edges = heatmap_out["mlat_edges"]
    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    p = heatmap_out["params_dict"]
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    title = (
        f"{MODEL_NAME} (12 params)  |  "
        f"C heatmap std = {heatmap_out['C_residual_std']:.2f} cnt/s  |  "
        f"Sci_rec std = {cache_out['Sci_rec_residual_std']:.2f} cnt/s\n"
        f"a={p['a']:.0f}, a1={p['alpha1']:.2f}, mu1={p['mu1']:.1f}, k1={p['k1']:.1f}, "
        f"a2={p['alpha2']:.2f}, mu2={p['mu2']:.1f}, k2={p['k2']:.1f}, "
        f"amp0={p['amp0']:.2f}, beta={p['beta']:.2f}, mu_t={p['mu_t']:.1f}, k_t={p['k_t']:.1f}, "
        f"C0={p['C_0']:+.0f}\n"
        f"unwrap change vs C=150 baseline: {cache_out['unwrap_change_pct']:.2f}%"
    )
    fig.suptitle(title, fontsize=10, fontweight='bold')

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_ylabel("|mlat| (deg)", fontsize=11)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data, 0, 400, 'viridis',
             'C_data (cnt/s)', '1. DATA  mean C (cnt/s)')
    plot_pcm(axes[1], C_pred, 0, 400, 'viridis',
             'C_model (cnt/s)', '2. MODEL  dual mlat sigmoid')
    plot_pcm(axes[2], resid, -30, 30, 'RdBu_r',
             'data - model (cnt/s)', '3. RESIDUAL (data - model)')
    axes[2].set_xlabel("date", fontsize=11)

    # Bottom: histogram of Sci_rec residual
    R = cache_out["Sci_rec_residual_arr"]
    ax = axes[3]
    if R.size > 0:
        bins = np.linspace(-200, 200, 401)
        ax.hist(R, bins=bins, color='#2266aa', alpha=0.85)
        ax.set_yscale('log')
        ax.set_xlim(-200, 200)
        ax.axvline(0, color='k', lw=0.7, alpha=0.6)
        ax.set_xlabel("Sci_rec - Sci_obs (cnt/s)", fontsize=11)
        ax.set_ylabel("count (log)", fontsize=11)
        ax.set_title(
            f"4. Sci_rec residual on sampled cache rows (n={R.size:,})  |  "
            f"std={cache_out['Sci_rec_residual_std']:.2f}, "
            f"max|.|={cache_out['Sci_rec_residual_max']:.1f}, "
            f"mean={cache_out['Sci_rec_residual_mean']:+.2f} cnt/s",
            fontsize=11)
    else:
        ax.text(0.5, 0.5, "no cache rows kept", ha='center', va='center')

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_path}")


def main():
    print("=" * 70)
    print(f"Fitting {MODEL_NAME} (12 params)")
    print("=" * 70)
    heatmap_out = fit_heatmap()
    print()
    print("=" * 70)
    print("Self-consistent cache evaluation (one row group per year)")
    print("=" * 70)
    cache_out = cache_eval(heatmap_out["params"])
    print(f"\n=== Cache Sci_rec residual ({cache_out.get('n_kept', 0):,} kept rows) ===")
    print(f"  std:           {cache_out['Sci_rec_residual_std']:.3f} cnt/s")
    print(f"  max|.|:        {cache_out['Sci_rec_residual_max']:.2f} cnt/s")
    print(f"  unwrap change: {cache_out['unwrap_change_pct']:.2f}% vs C=150 baseline")

    plot_path = f"/Users/skyair/Developer/ihep/blink/plots/fit_{MODEL_NAME}_2D_v2.png"
    make_plot(heatmap_out, cache_out, plot_path)

    # Summary JSON
    import json
    summary = {
        "model_name": MODEL_NAME,
        "n_params": 12,
        "params": heatmap_out["params_dict"],
        "C_residual_std": heatmap_out["C_residual_std"],
        "Sci_rec_residual_std": cache_out["Sci_rec_residual_std"],
        "Sci_rec_residual_max": cache_out["Sci_rec_residual_max"],
        "unwrap_change_pct": cache_out["unwrap_change_pct"],
        "plot_path": plot_path,
    }
    print("\n=== JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
