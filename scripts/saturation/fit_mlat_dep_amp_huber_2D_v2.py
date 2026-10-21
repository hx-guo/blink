#!/usr/bin/env python3
"""Round-2 candidate: mlat_dep_amp_huber.

Same 9-parameter functional form as Round-1 winner ``mlat_dependent_amp``::

    sigma_m(m) = 1/(1+exp(-(m-mu_m)/k_m))
    sigma_t(t) = 1/(1+exp(-(t-mu_t)/k_t))
    A(m)       = a*(1 + alpha*sigma_m(m))
    amp(m)     = amp0*(1 + beta*sigma_m(m))
    C(m,t)     = A(m)*(1 - amp(m)*sigma_t(t)) + C_0

The ONLY difference is the fit loss: ``loss='huber'`` with ``f_scale=10`` cnt/s
inside ``least_squares``.  This downweights monthly vertical-stripe anomalies
(which are real instrumental events, not part of the asymptotic 2-D shape) so
the smooth surface is determined by the bulk of the data rather than the few
high-residual months.

Pipeline (matches the bake-off harness):
  1. Fit on the 60x108 C(|mlat|, t) heatmap with Huber loss (delta=10).
  2. Self-consistent eval: read one row-group per yearly parquet, look up
     C_pred per row, run ``unwrap_v2`` + event-balance cap, compute
     ``Sci_rec - Sci_obs`` residual stats, and the unwrap-change fraction
     versus the C=150 baseline.
  3. Save the 4-panel diagnostic plot.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
import pyarrow.parquet as pq

HEATMAP_NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"
AACGM_NPZ = "n_below_study/aacgm_grid_2020.npz"
CACHE_DIR = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
CACHE_GLOB = "clean_relaxed_{year}.parquet"
YEARS = list(range(2017, 2027))

L_CYCLES_TO_SEC = 16e-6
C_BASELINE = 150.0
T0 = np.datetime64("2017-06-22")

NEEDED_COLS = [
    "date", "box", "det", "PHO", "Wide", "Large", "Sci_1s",
    "L_cycles", "Dt", "Lat", "Lon",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def model_grid(params, mlat_grid, t_grid):
    """Evaluate C model on a 2-D grid: returns (n_mlat, n_t)."""
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params
    sm = 1.0 / (1.0 + np.exp(-(mlat_grid - mu_m) / k_m))   # (n_mlat,)
    st = 1.0 / (1.0 + np.exp(-(t_grid - mu_t) / k_t))      # (n_t,)
    A = a * (1.0 + alpha * sm)
    amp_eff = amp0 * (1.0 + beta * sm)
    F = 1.0 - amp_eff[:, None] * st[None, :]
    return A[:, None] * F + C_0


def model_pointwise(params, mlat, t_years):
    """Per-row C evaluation (same params, broadcast on flat arrays)."""
    a, alpha, mu_m, k_m, amp0, beta, mu_t, k_t, C_0 = params
    mlat = np.abs(np.asarray(mlat, dtype=np.float64))
    t = np.asarray(t_years, dtype=np.float64)
    sm = 1.0 / (1.0 + np.exp(-(mlat - mu_m) / k_m))
    st = 1.0 / (1.0 + np.exp(-(t - mu_t) / k_t))
    A = a * (1.0 + alpha * sm)
    amp_eff = amp0 * (1.0 + beta * sm)
    return A * (1.0 - amp_eff * st) + C_0


# ---------------------------------------------------------------------------
# Unwrap (carbon copy of harness spec)
# ---------------------------------------------------------------------------
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


def unwrap_with_event_balance(pho, large, wide, sci, lc, dt, C):
    """Full unwrap including event-balance cap."""
    LL = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dt / lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C)
    mle = pho - ((sci + 0.0) * LL + wide) / lf
    n1 = np.round((lv1 - large) / 1024.0).astype(int)
    nmax = np.maximum(np.floor((mle - large) / 1024.0).astype(int), 0)
    return large + np.where(n1 > nmax, nmax, n1) * 1024.0


# ---------------------------------------------------------------------------
# Heatmap fit (Huber loss)
# ---------------------------------------------------------------------------
def fit_heatmap():
    z = np.load(HEATMAP_NPZ)
    C_data = z["C_med"]
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
        C_pred = model_grid(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    p0 = [100.0, 3.0, 50.0, 4.0, 0.5, 0.5, 3.0, 2.0, 0.0]
    lo = [10.0, 0.5, 30.0, 0.5, 0.0, -2.0, 0.5, 0.2, -100.0]
    hi = [500.0, 20.0, 60.0, 15.0, 1.5, 5.0, 8.0, 8.0, 100.0]

    # Huber robust loss; f_scale=10 cnt/s means deviations < ~10 cnt/s stay
    # quadratic, monthly anomalies get downweighted linearly.
    res = least_squares(
        residual, p0, bounds=(lo, hi), method="trf",
        loss="huber", f_scale=10.0, max_nfev=5000,
    )
    print(f"fit cost: {res.cost:.0f}, nfev: {res.nfev}, success: {res.success}")
    keys = ["a", "alpha", "mu_m", "k_m", "amp0", "beta", "mu_t", "k_t", "C_0"]
    params = dict(zip(keys, [float(x) for x in res.x]))
    for k, v in params.items():
        print(f"  {k:6s} = {v:+10.4f}")

    C_pred = model_grid(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred) * mask
    valid_resid = resid[mask]
    C_std = float(np.std(valid_resid))
    C_mean = float(np.mean(C_data[mask]))
    print(f"\nC residual std = {C_std:.3f} cnt/s ({C_std/C_mean*100:.2f}% of mean)")

    return {
        "params": params,
        "params_arr": res.x,
        "C_data": C_data,
        "C_pred": C_pred,
        "mask": mask,
        "edges": edges,
        "month_dt": month_dt,
        "mlat_centers": mlat_centers,
        "t_years": t_years,
        "C_std": C_std,
    }


# ---------------------------------------------------------------------------
# Self-consistent cache eval
# ---------------------------------------------------------------------------
def lookup_mlat(lat, lon, grid):
    """Bilinear interpolation of AACGM |mlat| from the 2020 grid.

    lat_grid: (91,) -45..45 in 1deg, lon_grid: (181,) 0..360 in 2deg,
    mlat: (91, 181).
    """
    lat_grid = grid["lat_grid"]
    lon_grid = grid["lon_grid"]
    mlat_map = grid["mlat"]

    lon_w = np.where(lon < 0, lon + 360.0, lon)
    lon_w = np.where(lon_w >= 360.0, lon_w - 360.0, lon_w)

    fi = np.interp(lat, lat_grid, np.arange(len(lat_grid)))
    fj = np.interp(lon_w, lon_grid, np.arange(len(lon_grid)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat_grid) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(lon_grid) - 2)
    di = fi - i0
    dj = fj - j0

    v00 = mlat_map[i0, j0]
    v01 = mlat_map[i0, j0 + 1]
    v10 = mlat_map[i0 + 1, j0]
    v11 = mlat_map[i0 + 1, j0 + 1]
    mlat = (
        v00 * (1 - di) * (1 - dj)
        + v01 * (1 - di) * dj
        + v10 * di * (1 - dj)
        + v11 * di * dj
    )
    return np.abs(mlat)


def date_to_years(date_arr):
    """Convert dates (string or datetime-like) to years since T0 (mid-month)."""
    dt = np.asarray(date_arr).astype("datetime64[D]")
    return (dt - T0).astype("timedelta64[D]").astype(float) / 365.25


def eval_cache(params_arr):
    grid = np.load(AACGM_NPZ)

    all_resid = []
    all_unwrap_changed = []
    n_rows_total = 0
    n_rows_used = 0

    for year in YEARS:
        path = Path(CACHE_DIR) / CACHE_GLOB.format(year=year)
        if not path.exists():
            print(f"  skip (missing): {path.name}")
            continue
        try:
            pf = pq.ParquetFile(str(path))
        except Exception as e:
            print(f"  skip ({e}): {path.name}")
            continue
        if pf.num_row_groups == 0:
            continue

        tbl = pf.read_row_group(0, columns=NEEDED_COLS)
        df = tbl.to_pandas()
        n_rows_total += len(df)

        pho = df["PHO"].to_numpy(dtype=np.float64)
        large = df["Large"].to_numpy(dtype=np.float64)
        wide = df["Wide"].to_numpy(dtype=np.float64)
        sci = df["Sci_1s"].to_numpy(dtype=np.float64)
        lc = df["L_cycles"].to_numpy(dtype=np.float64)
        dt_cnt = df["Dt"].to_numpy(dtype=np.float64)
        lat = df["Lat"].to_numpy(dtype=np.float64)
        lon = df["Lon"].to_numpy(dtype=np.float64)
        date = df["date"].to_numpy()

        finite = np.isfinite(pho) & np.isfinite(large) & np.isfinite(wide) \
            & np.isfinite(sci) & np.isfinite(lc) & np.isfinite(dt_cnt) \
            & np.isfinite(lat) & np.isfinite(lon) & (lc > 0)
        if not finite.any():
            continue

        pho = pho[finite]; large = large[finite]; wide = wide[finite]
        sci = sci[finite]; lc = lc[finite]; dt_cnt = dt_cnt[finite]
        lat = lat[finite]; lon = lon[finite]; date = date[finite]

        mlat_abs = lookup_mlat(lat, lon, grid)
        t_yrs = date_to_years(date)
        C_pred = model_pointwise(params_arr, mlat_abs, t_yrs)

        lv_pred = unwrap_with_event_balance(pho, large, wide, sci, lc, dt_cnt, C_pred)
        lv_base = unwrap_with_event_balance(
            pho, large, wide, sci, lc, dt_cnt,
            np.full_like(pho, C_BASELINE),
        )

        LL = lc * L_CYCLES_TO_SEC
        lf = 1.0 - dt_cnt / lc
        sci_rec = (pho - lv_pred) * lf / LL - wide / LL - C_pred
        resid = sci_rec - sci

        sel = (sci > 50) & np.isfinite(resid) & (np.abs(resid) < 1000)
        if not sel.any():
            continue

        all_resid.append(resid[sel])
        all_unwrap_changed.append((lv_pred[sel] != lv_base[sel]).astype(np.int8))
        n_rows_used += int(sel.sum())
        print(f"  {year}: rows={len(df):8d}, used={int(sel.sum()):8d}, "
              f"resid std={np.std(resid[sel]):.2f}, max={np.max(np.abs(resid[sel])):.1f}")

    if not all_resid:
        return {
            "sci_rec_std": float("nan"),
            "sci_rec_max": float("nan"),
            "unwrap_change_pct": float("nan"),
            "n_rows_total": n_rows_total,
            "n_rows_used": 0,
            "all_resid": np.array([]),
        }

    resid_all = np.concatenate(all_resid)
    unwrap_changed_all = np.concatenate(all_unwrap_changed)
    return {
        "sci_rec_std": float(np.std(resid_all)),
        "sci_rec_max": float(np.max(np.abs(resid_all))),
        "unwrap_change_pct": float(100.0 * unwrap_changed_all.mean()),
        "n_rows_total": n_rows_total,
        "n_rows_used": n_rows_used,
        "all_resid": resid_all,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(fit_out, eval_out, plot_path):
    C_data = fit_out["C_data"].astype(float).copy()
    C_pred = fit_out["C_pred"].astype(float).copy()
    mask = fit_out["mask"]
    edges = fit_out["edges"]
    month_dt = fit_out["month_dt"]
    params = fit_out["params"]
    C_std = fit_out["C_std"]

    C_data[~mask] = np.nan
    C_pred[~mask] = np.nan
    resid_grid = (fit_out["C_data"] - fit_out["C_pred"]).astype(float)
    resid_grid[~mask] = np.nan

    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    title = (
        "mlat_dep_amp_huber (9 params, Huber loss delta=10)\n"
        f"a={params['a']:.1f}, alpha={params['alpha']:.2f}, "
        f"mu_m={params['mu_m']:.2f}, k_m={params['k_m']:.2f}, "
        f"amp0={params['amp0']:.3f}, beta={params['beta']:.2f}, "
        f"mu_t={params['mu_t']:.2f}, k_t={params['k_t']:.2f}, C0={params['C_0']:+.1f}\n"
        f"C_residual_std={C_std:.2f} cnt/s   |   "
        f"Sci_rec_std={eval_out['sci_rec_std']:.2f} cnt/s "
        f"(N={eval_out['n_rows_used']}, unwrap-changed {eval_out['unwrap_change_pct']:.2f}%)"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title_):
        pcm = ax.pcolormesh(x_edges, edges, data, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading="flat")
        ax.set_ylabel("|mlat| (deg)", fontsize=10)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=9)
        ax.set_title(title_, fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data, 0, 400, "viridis",
             "C_data (cnt/s)", "1. DATA - C_med (cnt/s)")
    plot_pcm(axes[1], C_pred, 0, 400, "viridis",
             "C_model (cnt/s)", "2. MODEL - mlat_dep_amp_huber fit (Huber loss)")
    plot_pcm(axes[2], resid_grid, -30, 30, "RdBu_r",
             "data - model (cnt/s)", "3. RESIDUAL - data - model (+/- 30 cnt/s)")
    axes[2].set_xlabel("date", fontsize=10)

    # 4. Sci_rec residual histogram
    ax = axes[3]
    if eval_out["all_resid"].size > 0:
        bins = np.linspace(-200, 200, 161)
        ax.hist(eval_out["all_resid"], bins=bins, color="C0",
                edgecolor="none", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlim(-200, 200)
        med = float(np.median(eval_out["all_resid"]))
        ax.axvline(0, color="k", lw=0.7)
        ax.axvline(med, color="red", lw=1.0, ls="--",
                   label=f"median={med:+.2f}")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Sci_rec - Sci_obs (cnt/s)", fontsize=10)
    ax.set_ylabel("rows (log)", fontsize=10)
    ax.set_title(
        f"4. Sci_rec residual on sampled cache rows  "
        f"(std={eval_out['sci_rec_std']:.2f}, "
        f"max|.|={eval_out['sci_rec_max']:.1f}, "
        f"unwrap-changed vs C=150 baseline: {eval_out['unwrap_change_pct']:.2f}%)",
        fontsize=10,
    )

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {plot_path}")


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: Fit C(|mlat|, t) heatmap with Huber loss")
    print("=" * 70)
    fit_out = fit_heatmap()

    print("\n" + "=" * 70)
    print("STEP 2: Self-consistent eval on yearly cache (1 row group each)")
    print("=" * 70)
    eval_out = eval_cache(fit_out["params_arr"])
    print(f"\nSci_rec residual std = {eval_out['sci_rec_std']:.3f} cnt/s")
    print(f"Sci_rec residual max = {eval_out['sci_rec_max']:.3f} cnt/s")
    print(f"unwrap-change vs C=150: {eval_out['unwrap_change_pct']:.3f}%")

    plot_path = "plots/fit_mlat_dep_amp_huber_2D_v2.png"
    make_plot(fit_out, eval_out, plot_path)

    summary = {
        "model_name": "mlat_dep_amp_huber",
        "formula": (
            "C(m,t) = a*(1+alpha*sigm((m-mu_m)/k_m)) "
            "* (1 - amp0*(1+beta*sigm((m-mu_m)/k_m))*sigm((t-mu_t)/k_t)) + C0; "
            "fit with Huber loss delta=10 cnt/s"
        ),
        "n_params": 9,
        "params": fit_out["params"],
        "C_residual_std": fit_out["C_std"],
        "Sci_rec_residual_std": eval_out["sci_rec_std"],
        "Sci_rec_residual_max": eval_out["sci_rec_max"],
        "unwrap_change_pct": eval_out["unwrap_change_pct"],
        "plot_path": plot_path,
        "n_rows_used": eval_out["n_rows_used"],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
