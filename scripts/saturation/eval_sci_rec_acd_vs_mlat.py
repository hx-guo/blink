#!/usr/bin/env python3
"""Row-level Sci_rec head-to-head: ACD-based vs mlat-based vs C=150 baseline.

For each sampled cache row:
  - Compute C_pred from ACD-based model and mlat-based model
  - Self-consistent unwrap_v2 + event-balance cap with each C_pred
  - Sci_rec = (pho-large_unwrap)*lf/LL - wide/LL - C_pred
  - Residual = Sci_rec - Sci_obs

Compare std of residual distributions.
"""
from __future__ import annotations
import glob, os, sys, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
ACD_PATH = "/Users/skyair/Developer/ihep/astro_sift/astro_sift/satellites/hxmt/acd.txt"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    pho=np.asarray(pho,float); large=np.asarray(large,float); wide=np.asarray(wide,float)
    sci=np.asarray(sci,float); LL=np.asarray(lc,float)*L
    lf=1.0-np.asarray(dt,float)/np.asarray(lc,float)
    pred=pho-(wide+(sci+C)*LL)/lf
    n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0)
        out=large+np.where(ov,nm,n)*1024.
    return out


def unwrap_with_cap(pho, large, wide, sci, lc, dt, C_pred):
    LL = lc * L
    lf = 1.0 - dt/lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C_pred)
    mle = pho - (sci*LL + wide) / lf
    n1 = np.round((lv1 - large)/1024).astype(int)
    nmax = np.maximum(np.floor((mle - large)/1024.).astype(int), 0)
    lv_final = large + np.where(n1 > nmax, nmax, n1)*1024.
    return lv_final


def load_acd_lookup():
    a = np.loadtxt(ACD_PATH)
    lat_grid = a[180:360][:, 0]
    lon_grid = a[0:180][0, :]
    acd = a[360:540]
    lon_grid_360 = (lon_grid + 360) % 360
    sort_idx = np.argsort(lon_grid_360)
    return RegularGridInterpolator(
        (lat_grid, lon_grid_360[sort_idx]), acd[:, sort_idx],
        bounds_error=False, fill_value=np.nan)


def load_aacgm_lookup():
    g = np.load("n_below_study/aacgm_grid_2020.npz")
    return RegularGridInterpolator(
        (g["lat_grid"], g["lon_grid"]), g["mlat"],
        bounds_error=False, fill_value=np.nan)


# Models
def C_mlat_8p(mlat, t, p):
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C_0 = p
    sm = sigm((mlat - mu_m) / k_m)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g * (1.0 - amp0 * g * st) + C_0


def C_mlat_11p_dual(mlat, t, p):
    a, alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C_0 = p
    s1 = sigm((mlat - mu1) / k1)
    s2 = sigm((mlat - mu2) / k2)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha1*s1 + alpha2*s2
    return a * g * (1.0 - amp0 * g * st) + C_0


def C_acd_8p(log_acd, t, p):
    a, alpha, mu_A, k_A, amp0, mu_t, k_t, C_0 = p
    sm = sigm((log_acd - mu_A) / k_A)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g * (1.0 - amp0 * g * st) + C_0


# Fit-result params (will override from CLI args)
P_MLAT_8 = [202.60, 1.695, 44.46, 6.33, 0.152, 5.25, 1.00, -79.26]
P_MLAT_11 = [205.683, 1.837, 44.918, 7.328, -0.073, 14.534, 1.274, 0.155, 5.252, 0.996, -79.329]
P_ACD_8  = None   # populated from /tmp/acd_fit_params.json after fit


def main():
    p = Path("/tmp/acd_fit_params.json")
    if not p.exists():
        sys.exit(f"Missing {p}. Run fit_C_ACD_t_8param first.")
    P_ACD = json.loads(p.read_text())
    print(f"Loaded ACD-fit params: {P_ACD}")

    acd_lookup = load_acd_lookup()
    aacgm_lookup = load_aacgm_lookup()
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))
    t0 = np.datetime64("2017-06-22")

    samples = {"mlat8": [], "mlat11": [], "acd8": [], "base150": []}
    n_rows_total = 0

    for f in files:
        pf = pq.ParquetFile(f); rg = pf.num_row_groups // 2
        df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
        lat = df["Lat"].values; lon = df["Lon"].values
        mlat = np.abs(aacgm_lookup(np.column_stack([lat, lon])))
        mlat = np.where(np.isnan(mlat), 0.0, mlat)
        acd_v = acd_lookup(np.column_stack([lat, lon]))
        # Convert date to t_years
        dates = df["date"].values
        date_mid = np.array([np.datetime64(d) for d in dates])
        t_years = ((date_mid - t0).astype("timedelta64[D]").astype(float)) / 365.25

        pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
        wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
        lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
        LL = lc*L; lf = 1.0 - dtv/lc

        for name, C_pred in [
            ("mlat8",   C_mlat_8p(mlat, t_years, P_MLAT_8)),
            ("mlat11",  C_mlat_11p_dual(mlat, t_years, P_MLAT_11)),
            ("acd8",    C_acd_8p(np.log10(np.maximum(acd_v, 1e-3)), t_years, P_ACD)),
            ("base150", np.full(len(pho), 150.0)),
        ]:
            lv = unwrap_with_cap(pho, lg, wd, sci, lc, dtv, C_pred)
            sci_rec = (pho - lv)*lf/LL - wd/LL - C_pred
            resid = sci_rec - sci
            ok = np.isfinite(resid) & (sci > 50) & np.isfinite(C_pred) & (np.abs(resid) < 1000)
            samples[name].append(resid[ok])
        n_rows_total += len(pho)
        print(f"  {os.path.basename(f)}: {len(pho):,} rows")

    print(f"\nTotal rows sampled: {n_rows_total:,}")
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = {"base150": "red", "mlat8": "C0", "mlat11": "C2", "acd8": "C3"}
    labels = {"base150": "C=150 baseline",
              "mlat8":   "mlat 8p α=β",
              "mlat11":  "mlat 11p dual-σ",
              "acd8":    "ACD 8p α=β"}
    bins = np.linspace(-200, 200, 201)
    print("\n=== Sci_rec residual std (cnt/s) ===")
    for name in ["base150", "mlat8", "mlat11", "acd8"]:
        r = np.concatenate(samples[name])
        std = np.std(r); mean = np.mean(r); med = np.median(r)
        n = len(r)
        ax.hist(r, bins=bins, histtype='step', lw=2,
                color=colors[name], density=True,
                label=f"{labels[name]}  std={std:.2f} mean={mean:+.2f}  n={n/1e6:.1f}M")
        print(f"  {labels[name]:25s}  std={std:.3f}  mean={mean:+.3f}  median={med:+.3f}  n={n:,}")
    ax.set_xlabel("Sci_rec − Sci_obs (cnt/s)", fontsize=12)
    ax.set_ylabel("density (log)", fontsize=12)
    ax.set_yscale('log')
    ax.set_title("Row-level Sci_rec residual: ACD vs mlat models",
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/eval_sci_rec_acd_vs_mlat.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
