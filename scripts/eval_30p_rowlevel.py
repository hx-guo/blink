#!/usr/bin/env python3
"""Row-level Sci_rec: 25p vs 27p vs 30p."""
from __future__ import annotations
import glob, os, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
BOX_ID = {"a":0,"b":1,"c":2,"A":0,"B":1,"C":2}


def sigm(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    LL=lc*L; lf=1.0-dt/lc
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


def C_25p(mlat, t, idx, p):
    a = np.array(p["a_det"])[idx]
    sm = sigm((mlat - p["mu_m"]) / p["k_m"])
    st = sigm((t - p["mu_t"]) / p["k_t"])
    g = 1.0 + p["alpha"] * sm
    return a * g * (1.0 - p["amp0"] * g * st) + p["C0"]


def C_27p(mlat, t, idx, p):
    a = np.array(p["a_det"])[idx]
    st = sigm((t - p["mu_t"]) / p["k_t"])
    mu_m_t = p["mu_m0"] + p["dmu"] * st
    k_m_t = p["k_m0"] + p["dk"] * st
    sm = sigm((mlat - mu_m_t) / k_m_t)
    g = 1.0 + p["alpha"] * sm
    return a * g * (1.0 - p["amp0"] * g * st) + p["C0"]


def C_30p(mlat, t, idx, p):
    a = np.array(p["a_det"])[idx]
    st = sigm((t - p["mu_t1"]) / p["k_t1"])
    sd = sigm((t - p["mu_t2"]) / p["k_t2"])
    mu_m_t = p["mu_m0"] + p["dmu"] * sd
    k_m_t = p["k_m0"] + p["dk"] * sd
    sm = sigm((mlat - mu_m_t) / k_m_t)
    g_b = 1.0 + p["alpha"] * sm
    g_d = 1.0 + p["beta"] * sm
    return a * g_b * (1.0 - p["amp0"] * g_d * st) + p["C0"]


def main():
    p25 = json.loads(Path("/tmp/per_det_25param.json").read_text())
    p27 = json.loads(Path("/tmp/per_det_27p.json").read_text())
    p30 = json.loads(Path("/tmp/per_det_30p.json").read_text())

    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    aacgm = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))
    t0 = np.datetime64("2017-06-22")
    samples = {"25p":[], "27p":[], "30p":[]}
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
        bi = np.array([BOX_ID[b] for b in df["box"].values])
        di = df["det"].astype(int).values
        idx = bi*6 + di
        for name, fn in [("25p", lambda: C_25p(mlat, t_yr, idx, p25)),
                         ("27p", lambda: C_27p(mlat, t_yr, idx, p27)),
                         ("30p", lambda: C_30p(mlat, t_yr, idx, p30))]:
            Cp = fn()
            lv = unwrap_with_cap(pho, lg, wd, sci, lc, dtv, Cp)
            r = (pho - lv)*lf/LL - wd/LL - Cp - sci
            ok = np.isfinite(r) & (sci > 50) & np.isfinite(Cp) & (np.abs(r) < 1000)
            samples[name].append(r[ok])
        print(f"  {os.path.basename(f)}: {len(pho):,} rows")

    print("\n=== Sci_rec stats ===")
    print(f"  {'model':<10s} {'std':>7s} {'mean':>8s} {'median':>8s} {'P95-P5':>8s}")
    for name in ["25p","27p","30p"]:
        r = np.concatenate(samples[name])
        std = np.std(r); mean = np.mean(r); med = np.median(r)
        p5, p95 = np.percentile(r, [5, 95])
        print(f"  {name:<10s} {std:7.2f} {mean:+8.2f} {med:+8.2f} {p95-p5:8.1f}")


if __name__ == "__main__":
    main()
