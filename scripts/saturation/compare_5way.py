#!/usr/bin/env python3
"""5-way row-level Sci_rec: 8p vs v5t vs 25p vs 42p vs 59p."""
from __future__ import annotations
import glob, os, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
BOX_ID = {"a": 0, "b": 1, "c": 2, "A": 0, "B": 1, "C": 2}


def sigm(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


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
    LL=lc*L; lf=1.0-dt/lc
    lv1 = unwrap_v2(pho, large, wide, sci, lc, dt, C_pred)
    mle = pho - (sci*LL + wide) / lf
    n1 = np.round((lv1 - large)/1024).astype(int)
    nmax = np.maximum(np.floor((mle - large)/1024.).astype(int), 0)
    return large + np.where(n1 > nmax, nmax, n1)*1024.


P8 = (202.60, 1.695, 44.455, 6.331, 0.152, 5.252, 0.996, -79.257)


def C_8p(mlat, t):
    a, alpha, mu_m, k_m, amp0, mu_t, k_t, C0 = P8
    sm = sigm((np.abs(mlat) - mu_m) / k_m); st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return a * g * (1.0 - amp0 * g * st) + C0


def C_v5t(mlat, t, box, det, calib):
    s0_det = calib["s0_det"]; beta = float(calib["beta"]); w = float(calib["w"])
    k_coeffs = calib["k_coeffs"]; C_0 = float(calib["C0"])
    bi = np.array([BOX_ID[b] for b in box]); di = np.asarray(det, dtype=int)
    s0 = s0_det[bi*6 + di]
    g_t = 1.0 - beta * t
    k_t = (k_coeffs[0]
           + k_coeffs[1]*np.cos(w*t) + k_coeffs[2]*np.sin(w*t)
           + k_coeffs[3]*np.cos(2*w*t) + k_coeffs[4]*np.sin(2*w*t))
    mex = np.maximum(np.abs(mlat) - 20.0, 0.0)
    return s0 * g_t * (1.0 + k_t * mex**2) + C_0


def C_25p(mlat, t, box, det, p):
    a_det = np.array(p["a_det"])
    bi = np.array([BOX_ID[b] for b in box]); di = np.asarray(det, dtype=int)
    a = a_det[bi*6 + di]
    sm = sigm((np.abs(mlat) - p["mu_m"]) / p["k_m"])
    st = sigm((t - p["mu_t"]) / p["k_t"])
    g = 1.0 + p["alpha"] * sm
    return a * g * (1.0 - p["amp0"] * g * st) + p["C0"]


def C_42p(mlat, t, box, det, p):
    a_det = np.array(p["a_det"]); C0_det = np.array(p["C0_det"])
    bi = np.array([BOX_ID[b] for b in box]); di = np.asarray(det, dtype=int)
    idx = bi*6 + di
    a = a_det[idx]; C0 = C0_det[idx]
    sm = sigm((np.abs(mlat) - p["mu_m"]) / p["k_m"])
    st = sigm((t - p["mu_t"]) / p["k_t"])
    g = 1.0 + p["alpha"] * sm
    return a * g * (1.0 - p["amp0"] * g * st) + C0


def C_59p(mlat, t, box, det, p):
    a_det = np.array(p["a_det"]); C0_det = np.array(p["C0_det"])
    amp0_det = np.array(p["amp0_det"])
    bi = np.array([BOX_ID[b] for b in box]); di = np.asarray(det, dtype=int)
    idx = bi*6 + di
    a = a_det[idx]; C0 = C0_det[idx]; amp0 = amp0_det[idx]
    sm = sigm((np.abs(mlat) - p["mu_m"]) / p["k_m"])
    st = sigm((t - p["mu_t"]) / p["k_t"])
    g = 1.0 + p["alpha"] * sm
    return a * g * (1.0 - amp0 * g * st) + C0


def main():
    with np.load("n_below_study/v5_npz/v5t_calib.npz") as z:
        calib = {k: z[k] for k in ("s0_det", "beta", "w", "k_coeffs", "C0")}
    p25 = json.loads(Path("/tmp/per_det_25param.json").read_text())
    p42 = json.loads(Path("/tmp/per_det_42p.json").read_text())
    p59 = json.loads(Path("/tmp/per_det_59p.json").read_text())

    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    aacgm = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))
    t0 = np.datetime64("2017-06-22")

    samples = {k: [] for k in ["base150", "8p", "v5t", "25p", "42p", "59p"]}
    for f in files:
        pf = pq.ParquetFile(f); rg = pf.num_row_groups // 2
        df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
        lat = df["Lat"].values; lon = df["Lon"].values
        mlat = np.abs(aacgm(np.column_stack([lat, lon])))
        mlat = np.where(np.isnan(mlat), 0.0, mlat)
        dates = df["date"].values
        date_mid = np.array([np.datetime64(d) for d in dates])
        t_yr = ((date_mid - t0).astype("timedelta64[D]").astype(float)) / 365.25
        pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
        wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
        lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
        LL=lc*L; lf=1.0-dtv/lc
        box_v = df["box"].values; det_v = df["det"].values
        preds = {
            "base150": np.full(len(pho), 150.0),
            "8p":      C_8p(mlat, t_yr),
            "v5t":     C_v5t(mlat, t_yr, box_v, det_v, calib),
            "25p":     C_25p(mlat, t_yr, box_v, det_v, p25),
            "42p":     C_42p(mlat, t_yr, box_v, det_v, p42),
            "59p":     C_59p(mlat, t_yr, box_v, det_v, p59),
        }
        for name, Cp in preds.items():
            lv = unwrap_with_cap(pho, lg, wd, sci, lc, dtv, Cp)
            r = (pho - lv)*lf/LL - wd/LL - Cp - sci
            ok = np.isfinite(r) & (sci > 50) & np.isfinite(Cp) & (np.abs(r) < 1000)
            samples[name].append(r[ok])
        print(f"  {os.path.basename(f)}: {len(pho):,} rows")

    fig, ax = plt.subplots(figsize=(15, 9))
    colors = {"base150":"red","8p":"C0","v5t":"C2","25p":"C3","42p":"C4","59p":"C1"}
    labels = {
        "base150": "C=150 baseline           (0p)",
        "8p":      "8p α=β (phenomenological) ",
        "v5t":     "v5t (per-det + outgassing+solar)",
        "25p":     "25p (8p + per-det a)     ",
        "42p":     "42p (+ per-det C₀)       ",
        "59p":     "59p (+ per-det amp₀)     ",
    }
    bins = np.linspace(-200, 200, 201)
    print("\n=== Sci_rec residual stats ===")
    print(f"  {'model':<45s} {'std':>7s} {'mean':>8s} {'median':>8s} {'P95-P5':>8s}")
    for name in ["base150","8p","v5t","25p","42p","59p"]:
        r = np.concatenate(samples[name])
        std = np.std(r); mean = np.mean(r); med = np.median(r)
        p5, p95 = np.percentile(r, [5, 95]); n = len(r)
        print(f"  {labels[name]:<45s} {std:7.2f} {mean:+8.2f} {med:+8.2f} {p95-p5:8.1f}")
        ax.hist(r, bins=bins, histtype='step', lw=2, color=colors[name], density=True,
                label=f"{labels[name]}  std={std:.2f} mean={mean:+.2f} median={med:+.2f}")

    ax.set_xlabel("Sci_rec − Sci_obs (cnt/s)", fontsize=12)
    ax.set_ylabel("density (log)", fontsize=12)
    ax.set_yscale('log')
    ax.set_title("5-way row-level Sci_rec: phenomenological per-det progression vs v5t narrative",
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/compare_5way.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
