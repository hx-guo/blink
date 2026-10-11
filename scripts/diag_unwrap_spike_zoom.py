#!/usr/bin/env python3
"""Zoom into the unwrap spike around 2020-05-25 09:25 UTC.

Print per-detector PHO, raw Large, Wide, Sci, L_cyc, Dt, computed mle, expected_Large,
n_wrap, capped n, Large_unwrapped, Sci_rec for the worst spike seconds.

This tells us whether unwrap_v2 is making a math error, or whether the input data
itself is partial/inconsistent (e.g. PHO truncated, Wide elevated).
"""
from __future__ import annotations
import os
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","met_sec","PHO","Wide","Large","Sci_1s",
          "L_cycles","Dt","Lat","Lon"]
CACHE_FILE = "/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_2020.parquet"


def main():
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)

    target_date="2020-05-25"
    # Window: ~09:25 UTC. MET = 09:25:00 = met_sec 265024500 (approx)
    # We'll search a 5-min window: 265023700..265025700
    print(f"Streaming {target_date} for window 09:20-09:30...", flush=True)
    pf = pq.ParquetFile(CACHE_FILE)
    chunks=[]
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[(b["date"]==target_date) &
              (b["met_sec"]>=265023600) & (b["met_sec"]<=265025400)]
        if len(b)>0: chunks.append(b)
    import pandas as pd
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    print(f"  loaded {len(df):,} rows", flush=True)

    # Compute v5t C
    am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
    am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-20.0)**2
    d_arr=np.array([np.datetime64(d) for d in df["date"].values])
    ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
    g_t=1.0-beta*ty
    k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
    box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
    detid=box_idx*6+df["det"].values
    C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0

    pho=df["PHO"].astype(float).values
    lg=df["Large"].astype(float).values
    wd=df["Wide"].astype(float).values
    sci=df["Sci_1s"].astype(float).values
    lc=df["L_cycles"].astype(float).values
    dtv=df["Dt"].astype(float).values
    LL=lc*L; lf=1.0-dtv/lc

    # Pass 1: unwrap with cap
    pred=pho-(wd+(sci+C_v5)*LL)/lf
    n1=np.maximum(np.round((pred-lg)/1024.).astype(int),0)
    mx=pho-wd
    out1=lg+n1*1024.
    ov1=out1>mx
    nm1=np.maximum(np.floor((mx-lg)/1024.).astype(int),0)
    n1_after_cap = np.where(ov1, nm1, n1)
    out1 = lg + n1_after_cap*1024.

    # Pass 2: event-balance cap
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
    n3=np.round((out1-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
    n_final = np.where(n3>nmax, nmax, n3)
    lv5=lg+n_final*1024.

    base=(pho-lv5)*lf/LL-wd/LL
    sci_rec=base-C_v5
    resid=sci_rec-sci

    df["C_v5"]=C_v5; df["pred_Large"]=pred; df["n1"]=n1; df["n1_cap"]=n1_after_cap
    df["mle"]=mle; df["n_final"]=n_final
    df["Large_unwrap"]=lv5; df["sci_rec"]=sci_rec; df["resid"]=resid
    df["mx"]=mx; df["nmax"]=nmax

    # Find worst 8 rows by |resid|
    worst = df.iloc[np.argsort(-np.abs(resid))[:12]].copy()
    print(f"\n=== Top 12 highest-|residual| rows in window ===")
    cols=["met_sec","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt",
          "C_v5","pred_Large","n1","n1_cap","mle","nmax","n_final",
          "Large_unwrap","sci_rec","resid"]
    print(worst[cols].to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    # Also pick a "normal" row from window for comparison
    normal_mask = (df["Sci_1s"]>100) & (df["Sci_1s"]<1500) & (np.abs(resid)<30)
    normal = df[normal_mask].sample(5, random_state=0)
    print(f"\n=== 5 normal rows (Sci 100-1500, |resid|<30) for comparison ===")
    print(normal[cols].to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    # Sanity: PHO < Large_unwrap?  Physical impossibility?
    print(f"\n=== Rows where Large_unwrap > PHO (physical impossibility) ===")
    bad = df[df["Large_unwrap"] > df["PHO"]]
    print(f"  count: {len(bad):,} / {len(df):,} ({len(bad)/len(df)*100:.3f}%)")
    if len(bad)>0:
        sample=bad.iloc[:8]
        print(sample[cols].to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    # Sanity: Large_unwrap > PHO - Wide?  Cap should prevent this
    print(f"\n=== Rows where Large_unwrap > PHO - Wide (cap violation) ===")
    bad2 = df[df["Large_unwrap"] > df["PHO"] - df["Wide"]]
    print(f"  count: {len(bad2):,} / {len(df):,} ({len(bad2)/len(df)*100:.3f}%)")


if __name__=="__main__":
    main()
