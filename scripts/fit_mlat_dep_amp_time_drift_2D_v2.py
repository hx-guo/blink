#!/usr/bin/env python3
"""Round 2 candidate: mlat_dep_amp_time_drift.

Model (10 params):
    mu_m(t) = mu_m0 + delta_mu * t
    sigma_m(m,t) = 1 / (1 + exp(-(m - mu_m(t))/k_m))
    sigma_t(t)   = 1 / (1 + exp(-(t - mu_t)/k_t))
    C(m,t) = a * (1 + alpha*sigma_m(m,t)) *
             (1 - amp0*(1 + beta*sigma_m(m,t))*sigma_t(t)) + C_0

Idea: allow the mlat inflection point to drift linearly in time, testing
whether the high-mlat band shrinks (delta_mu>0) or expands (delta_mu<0)
across years.

Step 1 — fit on the C(t,|mlat|) heatmap (least_squares, trf).
Step 2 — self-consistent eval on sampled cache (one row group per year
parquet): rerun unwrap_v2 with C_pred(mlat,t) and event-balance cap,
compute Sci_rec - Sci_obs residual stats.
Step 3 — four-panel diagnostic plot.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.parquet as pq
from scipy.optimize import least_squares

ROOT = Path("/Users/skyair/Developer/ihep/blink")
NPZ_HEATMAP = ROOT / "n_below_study/v5_npz/C_2D_heatmap.npz"
NPZ_AACGM = ROOT / "n_below_study/aacgm_grid_2020.npz"
CACHE_TEMPLATE = "/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_{year}.parquet"
PLOT_OUT = ROOT / "plots/fit_mlat_dep_amp_time_drift_2D_v2.png"

L_CYCLE_S = 16e-6
T0 = np.datetime64("2017-06-22")
NEEDED_COLS = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s",
               "L_cycles", "Dt", "Lat", "Lon"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_grid(params, mlat_centers, t_years):
    """Vectorised on (n_mlat,) x (n_t,) grid -> (n_mlat, n_t)."""
    a, alpha, mu_m0, delta_mu, k_m, amp0, beta, mu_t, k_t, C_0 = params
    mu_m_t = mu_m0 + delta_mu * t_years                       # (n_t,)
    # sigma_m broadcast over mlat (axis 0) and time (axis 1)
    arg = (mlat_centers[:, None] - mu_m_t[None, :]) / k_m
    sigma_m = 1.0 / (1.0 + np.exp(-arg))                       # (n_mlat, n_t)
    sigma_t = 1.0 / (1.0 + np.exp(-(t_years - mu_t) / k_t))    # (n_t,)
    A = a * (1.0 + alpha * sigma_m)                            # (n_mlat, n_t)
    amp_eff = amp0 * (1.0 + beta * sigma_m)                    # (n_mlat, n_t)
    F = 1.0 - amp_eff * sigma_t[None, :]                       # (n_mlat, n_t)
    return A * F + C_0


def model_per_row(params, mlat_row, t_row):
    """Per-row evaluation on flat arrays (same length)."""
    a, alpha, mu_m0, delta_mu, k_m, amp0, beta, mu_t, k_t, C_0 = params
    mu_m_t = mu_m0 + delta_mu * t_row
    sigma_m = 1.0 / (1.0 + np.exp(-(mlat_row - mu_m_t) / k_m))
    sigma_t = 1.0 / (1.0 + np.exp(-(t_row - mu_t) / k_t))
    A = a * (1.0 + alpha * sigma_m)
    amp_eff = amp0 * (1.0 + beta * sigma_m)
    F = 1.0 - amp_eff * sigma_t
    return A * F + C_0


# ---------------------------------------------------------------------------
# Unwrap (self-consistent, v2 + event-balance cap)
# ---------------------------------------------------------------------------

def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    LL = lc * L_CYCLE_S
    lf = 1.0 - dt / lc
    pred = pho - (wide + (sci + C) * LL) / lf
    n = np.maximum(np.round((pred - large) / 1024.).astype(int), 0)
    mx = pho - wide
    out = large + n * 1024.0
    ov = out > mx
    if ov.any():
        nm = np.maximum(np.floor((mx - large) / 1024.).astype(int), 0)
        out = large + np.where(ov, nm, n) * 1024.0
    return out


def event_balance_cap(pho, large, wide, sci, lc, dt, C):
    LL = lc * L_CYCLE_S
    lf = 1.0 - dt / lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C)
    mle = pho - ((sci + 0.0) * LL + wide) / lf
    n1 = np.round((lv1 - large) / 1024.).astype(int)
    nmax = np.maximum(np.floor((mle - large) / 1024.).astype(int), 0)
    return large + np.where(n1 > nmax, nmax, n1) * 1024.0


# ---------------------------------------------------------------------------
# Sci_rec from unwrapped Large
# ---------------------------------------------------------------------------

def sci_rec_from_large(pho, large_unwrap, wide, lc, dt, C):
    LL = lc * L_CYCLE_S
    lf = 1.0 - dt / lc
    return (pho - large_unwrap) * lf / LL - wide / LL - C


# ---------------------------------------------------------------------------
# Cache sampling helper
# ---------------------------------------------------------------------------

def _aacgm_lookup(lat, lon, grid):
    lat_g, lon_g, mlat_g = grid["lat_grid"], grid["lon_grid"], grid["mlat"]
    # lat_g: (-45..45,1deg, 91 pts), lon_g: (0..360,2deg,181 pts) typical.
    lat_idx = np.clip(np.round((lat - lat_g[0]) / (lat_g[1] - lat_g[0])).astype(int), 0, len(lat_g) - 1)
    lon_idx = np.clip(np.round((lon - lon_g[0]) / (lon_g[1] - lon_g[0])).astype(int), 0, len(lon_g) - 1)
    return mlat_g[lat_idx, lon_idx]


def sample_cache():
    grid = np.load(NPZ_AACGM)
    chunks = []
    for year in range(2017, 2027):
        path = CACHE_TEMPLATE.format(year=year)
        pf = pq.ParquetFile(path)
        tbl = pf.read_row_group(0, columns=NEEDED_COLS)
        df = tbl.to_pandas()
        # Per-row mlat
        mlat = np.abs(_aacgm_lookup(df["Lat"].values, df["Lon"].values, grid))
        # Time in years from T0
        dates = np.array(df["date"].values, dtype="datetime64[D]")
        t_yrs = (dates - T0).astype("timedelta64[D]").astype(float) / 365.25
        chunks.append({
            "PHO": df["PHO"].values.astype(np.float64),
            "Wide": df["Wide"].values.astype(np.float64),
            "Large": df["Large"].values.astype(np.float64),
            "Sci": df["Sci_1s"].values.astype(np.float64),
            "lc": df["L_cycles"].values.astype(np.float64),
            "dt": df["Dt"].values.astype(np.float64),
            "mlat": mlat.astype(np.float64),
            "t_yrs": t_yrs.astype(np.float64),
        })
        print(f"  year {year}: {df.shape[0]} rows")
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    print(f"total sampled rows: {len(out['PHO'])}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Loading heatmap ===")
    z = np.load(NPZ_HEATMAP)
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

    # ---- Fit on heatmap ----
    def residual(params):
        C_pred = model_grid(params, mlat_centers, t_years)
        r = (C_pred - C_data_clean) * mask
        return r.ravel()

    # params: a, alpha, mu_m0, delta_mu, k_m, amp0, beta, mu_t, k_t, C_0
    p0 = [200.0, 1.7, 44.0, 0.0, 6.0, 0.15, 1.7, 5.0, 1.0, -80.0]
    lo = [10.0, 0.1, 30.0, -3.0, 0.5, 0.0, -2.0, 0.5, 0.2, -200.0]
    hi = [500.0, 20.0, 60.0, 3.0, 15.0, 1.5, 5.0, 9.0, 8.0, 200.0]
    print("\n=== Fitting model ===")
    res = least_squares(residual, p0, bounds=(lo, hi), method="trf", max_nfev=5000)
    print(f"  cost={res.cost:.1f}  nfev={res.nfev}  success={res.success}")
    a, alpha, mu_m0, delta_mu, k_m, amp0, beta, mu_t, k_t, C_0 = res.x
    print(f"  a       = {a:8.3f}")
    print(f"  alpha   = {alpha:8.3f}")
    print(f"  mu_m0   = {mu_m0:8.3f}   (mlat inflection at t=0)")
    print(f"  delta_mu= {delta_mu:+8.4f}  deg/yr")
    print(f"  k_m     = {k_m:8.3f}")
    print(f"  amp0    = {amp0:8.4f}")
    print(f"  beta    = {beta:8.3f}")
    print(f"  mu_t    = {mu_t:8.3f}")
    print(f"  k_t     = {k_t:8.3f}")
    print(f"  C_0     = {C_0:+8.3f}")

    C_pred_grid = model_grid(res.x, mlat_centers, t_years)
    resid = (C_data - C_pred_grid) * mask
    valid_resid = resid[mask]
    C_residual_std = float(np.std(valid_resid))
    mean_C = float(np.mean(C_data[mask]))
    print(f"  C_residual_std = {C_residual_std:.3f} cnt/s  ({C_residual_std/mean_C*100:.2f}%)")
    print(f"  C_residual_max = {np.max(np.abs(valid_resid)):.2f} cnt/s")

    # ---- Self-consistent cache eval ----
    print("\n=== Sampling cache (one row group per year) ===")
    s = sample_cache()
    PHO, Wide, Large = s["PHO"], s["Wide"], s["Large"]
    Sci, lc, dt = s["Sci"], s["lc"], s["dt"]
    mlat_row, t_row = s["mlat"], s["t_yrs"]

    # Filter: lc>0, dt<lc, finite, sci>50
    good = (lc > 0) & (dt < lc) & np.isfinite(mlat_row) & np.isfinite(t_row) & (Sci > 50)
    print(f"  rows after basic filter (lc>0, dt<lc, sci>50): {good.sum()}")

    def _eval_C(arr_mlat, arr_t):
        return model_per_row(res.x, arr_mlat, arr_t)

    # Predicted C with full model
    C_row = np.zeros_like(PHO)
    C_row[good] = _eval_C(mlat_row[good], t_row[good])

    # Baseline (C=150)
    lv_base = unwrap_v2(PHO[good], Large[good], Wide[good], Sci[good],
                        lc[good], dt[good], 150.0)

    # New self-consistent unwrap with event-balance cap
    lv_new = event_balance_cap(PHO[good], Large[good], Wide[good], Sci[good],
                               lc[good], dt[good], C_row[good])

    # Sci reconstruction
    Sci_rec = sci_rec_from_large(PHO[good], lv_new, Wide[good],
                                 lc[good], dt[good], C_row[good])
    sci_obs = Sci[good]
    resid_sci = Sci_rec - sci_obs

    # Final filter: finite, |resid| < 1000
    fin = np.isfinite(resid_sci) & (np.abs(resid_sci) < 1000)
    print(f"  rows after |resid|<1000 filter: {fin.sum()} / {good.sum()}")
    resid_sci_fin = resid_sci[fin]
    Sci_rec_residual_std = float(np.std(resid_sci_fin))
    Sci_rec_residual_max = float(np.max(np.abs(resid_sci_fin)))
    print(f"  Sci_rec_residual_std = {Sci_rec_residual_std:.3f} cnt/s  (WINNER METRIC)")
    print(f"  Sci_rec_residual_max = {Sci_rec_residual_max:.3f} cnt/s")

    # unwrap_change vs C=150 baseline
    unwrap_change_pct = 100.0 * float(np.mean(lv_new != lv_base))
    print(f"  unwrap_change_pct (lv_new != lv_base@C=150) = {unwrap_change_pct:.3f}%")

    # ---- 4-panel plot ----
    print("\n=== Plotting ===")
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    fig.suptitle(
        f"mlat_dep_amp_time_drift (10 params) — self-consistent v2\n"
        f"a={a:.0f}, alpha={alpha:.2f}, mu_m0={mu_m0:.1f}, d_mu={delta_mu:+.3f}/yr, "
        f"k_m={k_m:.2f}, amp0={amp0:.3f}, beta={beta:.2f}, "
        f"mu_t={mu_t:.2f}, k_t={k_t:.2f}, C0={C_0:.0f}\n"
        f"C_resid_std = {C_residual_std:.2f} cnt/s  |  "
        f"Sci_rec_resid_std = {Sci_rec_residual_std:.2f} cnt/s  |  "
        f"unwrap_change = {unwrap_change_pct:.2f}%",
        fontsize=11, fontweight="bold",
    )

    C_data_m = C_data.copy().astype(float); C_data_m[~mask] = np.nan
    C_pred_m = C_pred_grid.copy().astype(float); C_pred_m[~mask] = np.nan
    resid_m = resid.copy().astype(float); resid_m[~mask] = np.nan

    x_edges = np.concatenate([
        [mdates.date2num(month_dt[0]) - 15],
        mdates.date2num(month_dt[:-1] + (month_dt[1:] - month_dt[:-1]) / 2),
        [mdates.date2num(month_dt[-1]) + 15],
    ])

    def plot_pcm(ax, data, vmin, vmax, cmap, label, title):
        pcm = ax.pcolormesh(x_edges, edges, data, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading="flat")
        ax.set_ylabel("|mlat| (deg)", fontsize=11)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_pcm(axes[0], C_data_m, 0, 400, "viridis",
             "C_data (cnt/s)", "1. DATA — mean C")
    plot_pcm(axes[1], C_pred_m, 0, 400, "viridis",
             "C_model (cnt/s)", "2. MODEL — mlat_dep_amp_time_drift")
    plot_pcm(axes[2], resid_m, -30, 30, "RdBu_r",
             "data - model (cnt/s)", "3. RESIDUAL — data - model (+/- 30 cnt/s)")

    ax3 = axes[3]
    ax3.hist(resid_sci_fin, bins=200, range=(-200, 200),
             color="steelblue", edgecolor="black", linewidth=0.3)
    ax3.set_yscale("log")
    ax3.set_xlim(-200, 200)
    ax3.set_xlabel("Sci_rec - Sci_obs (cnt/s)", fontsize=11)
    ax3.set_ylabel("count (log)", fontsize=11)
    ax3.set_title(
        f"4. Sci_rec RESIDUAL on cache rows (n={len(resid_sci_fin)}, "
        f"std={Sci_rec_residual_std:.2f}, max={Sci_rec_residual_max:.0f} cnt/s)",
        fontsize=11,
    )
    ax3.axvline(0, color="k", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {PLOT_OUT}")

    import json
    summary = {
        "model_name": "mlat_dep_amp_time_drift",
        "n_params": 10,
        "formula": "mu_m(t)=mu_m0+delta_mu*t; C=a*(1+alpha*sigm((m-mu_m(t))/k_m))*"
                   "(1-amp0*(1+beta*sigm((m-mu_m(t))/k_m))*sigm((t-mu_t)/k_t))+C0",
        "params": {
            "a": float(a), "alpha": float(alpha), "mu_m0": float(mu_m0),
            "delta_mu": float(delta_mu), "k_m": float(k_m), "amp0": float(amp0),
            "beta": float(beta), "mu_t": float(mu_t), "k_t": float(k_t),
            "C_0": float(C_0),
        },
        "C_residual_std": C_residual_std,
        "Sci_rec_residual_std": Sci_rec_residual_std,
        "Sci_rec_residual_max": Sci_rec_residual_max,
        "unwrap_change_pct": unwrap_change_pct,
        "plot_path": str(PLOT_OUT),
    }
    print("\n=== JSON ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
