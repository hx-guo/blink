#!/usr/bin/env python3
"""Deep-dive: 2025-08 cluster forensics.

(1) Did Box A go offline (HV protection) during 2025-08?
(2) Which exact days are the cluster concentrated on?
(3) What do PHO / Large / Wide / Dt look like for the cluster rows
    vs main-band rows on the SAME days?
(4) Cross-correlate: where is Box A's HV on cluster days?

Reads the FULL 2025 cache (no pre-filter), since the cluster is only ~5k rows
of the year so we want everything.
"""
from __future__ import annotations
import os
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","met_sec","PHO","Wide","Large","Sci_1s","L_cycles","Dt","HV","Lat","Lon","ACD_sum","PM_0"]
CACHE_FILE = "/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_2025.parquet"


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


def main():
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)

    # Stream by row groups, collect only 2025-08 rows (avoid OOM)
    print("Streaming 2025 cache for 2025-08 rows...", flush=True)
    pf=pq.ParquetFile(CACHE_FILE)
    chunks=[]
    for rg in range(pf.num_row_groups):
        b=pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b=b[b["date"].str.startswith("2025-08")]
        if len(b)>0: chunks.append(b)
        if rg%4==0: print(f"  rg {rg}/{pf.num_row_groups}, kept {sum(len(c) for c in chunks):,}", flush=True)
    aug=__import__("pandas").concat(chunks, ignore_index=True) if chunks else None
    if aug is None or len(aug)==0:
        print("No 2025-08 rows found"); return
    print(f"\n  2025-08 rows: {len(aug):,}", flush=True)
    del chunks

    # Compute v5t C, base, sci_rec, residual
    pho=aug["PHO"].astype(float).values; lg=aug["Large"].astype(float).values
    wd=aug["Wide"].astype(float).values; sci=aug["Sci_1s"].astype(float).values
    lc=aug["L_cycles"].astype(float).values; dtv=aug["Dt"].astype(float).values
    hv=aug["HV"].astype(float).values
    am=np.abs(interp(np.column_stack([aug["Lat"].values,aug["Lon"].values])))
    am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-20.0)**2
    d_arr=np.array([np.datetime64(d) for d in aug["date"].values])
    ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
    g_t=1.0-beta*ty
    k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
    box_idx=np.select([aug["box"].values==b for b in "ABC"],[0,1,2],default=0)
    detid=box_idx*6+aug["det"].values
    C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0
    LL=lc*L; lf=1.0-dtv/lc
    lv3=unwrap_v2(pho,lg,wd,sci,lc,dtv,C_v5)
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
    n3=np.round((lv3-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
    lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
    base=(pho-lv5)*lf/LL-wd/LL
    sci_rec=base-C_v5
    resid=sci_rec-sci

    aug["sci_rec"]=sci_rec; aug["resid"]=resid

    cluster=(sci>=1100)&(sci<=1500)&(sci_rec>=500)&(sci_rec<=800)&np.isfinite(sci_rec)
    print(f"\n2025-08 cluster rows: {cluster.sum():,}")

    # === Q1: Box presence in 2025-08 ===
    print(f"\n=== Q1: Box presence each day in 2025-08 ===")
    print(f"      total per-box per-day count (across det+sec)")
    dates=sorted(aug["date"].unique())
    print(f"  date         BoxA      BoxB      BoxC    | BoxA_HV_med  BoxB_HV_med  BoxC_HV_med | cluster_rows")
    for d in dates:
        m=(aug["date"]==d)
        na=((aug["box"]=="A")&m).sum(); nb=((aug["box"]=="B")&m).sum(); nc=((aug["box"]=="C")&m).sum()
        hv_a=hv[(aug["box"]=="A")&m]; hv_b=hv[(aug["box"]=="B")&m]; hv_c=hv[(aug["box"]=="C")&m]
        med_a=np.median(hv_a) if len(hv_a)>0 else float('nan')
        med_b=np.median(hv_b) if len(hv_b)>0 else float('nan')
        med_c=np.median(hv_c) if len(hv_c)>0 else float('nan')
        ncl=(cluster&m).sum()
        print(f"  {d}  {na:>7,}  {nb:>7,}  {nc:>7,} | {med_a:>10.1f}  {med_b:>10.1f}  {med_c:>10.1f} | {ncl:>6,}")

    # === Q2: cluster days only — what are PHO/Wide/Large like? ===
    cl_dates=Counter(aug.loc[cluster,"date"])
    top_cluster_days=[d for d,_ in cl_dates.most_common(5)]
    print(f"\n=== Q2: Top 5 cluster days, PHO/Wide/Large/Sci/Dt distributions ===")
    print(f"  date         box  N_cluster  PHO_med    Wide_med  Large_med  Sci_med  L_cyc_med  Dt_med  Lat_med  Lon_med")
    for d in top_cluster_days:
        for box in "ABC":
            mc=cluster&(aug["date"]==d)&(aug["box"]==box)
            n=mc.sum()
            if n==0: continue
            print(f"  {d}  {box}  {n:>7,}  "
                  f"{np.median(pho[mc]):>8.0f}  {np.median(wd[mc]):>8.0f}  {np.median(lg[mc]):>8.0f}  "
                  f"{np.median(sci[mc]):>7.0f}  {np.median(lc[mc]):>8.0f}  {np.median(dtv[mc]):>7.0f}  "
                  f"{np.median(aug.loc[mc,'Lat']):>6.1f}  {np.median(aug.loc[mc,'Lon']):>6.1f}")

    # === Q3: same days, NON-cluster rows ===
    print(f"\n=== Q3: NON-cluster rows on same days (control comparison) ===")
    print(f"  date         box  N_other  PHO_med    Wide_med  Large_med  Sci_med  Sci_rec_med  resid_med")
    for d in top_cluster_days:
        for box in "ABC":
            mn=(~cluster)&(aug["date"]==d)&(aug["box"]==box)&np.isfinite(sci_rec)
            n=mn.sum()
            if n==0: continue
            print(f"  {d}  {box}  {n:>7,}  "
                  f"{np.median(pho[mn]):>8.0f}  {np.median(wd[mn]):>8.0f}  {np.median(lg[mn]):>8.0f}  "
                  f"{np.median(sci[mn]):>7.0f}  {np.median(sci_rec[mn]):>8.0f}  {np.median(resid[mn]):>7.0f}")

    # === Q4: Box A HV distribution over 2025-08 ===
    boxa_mask=(aug["box"]=="A")
    print(f"\n=== Q4: Box A HV distribution across 2025-08 ===")
    hv_a_aug=hv[boxa_mask]
    print(f"  Box A 2025-08 N_rows={len(hv_a_aug):,}")
    print(f"  HV   min={hv_a_aug.min():.1f}, p1={np.percentile(hv_a_aug,1):.1f}, med={np.median(hv_a_aug):.1f}, p99={np.percentile(hv_a_aug,99):.1f}, max={hv_a_aug.max():.1f}")
    print(f"  HV>−1100 fraction: {(hv_a_aug>-1100).mean()*100:.1f}%")
    print(f"  HV<−900 fraction:  {(hv_a_aug<-900).mean()*100:.1f}%")
    print(f"  HV in (−1100,−900) fraction:  {((hv_a_aug>-1100)&(hv_a_aug<-900)).mean()*100:.1f}%")

    # === Plot: per-day box counts + HV evolution ===
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    date_arr=np.array(sorted(aug["date"].unique()))
    day_count_A=np.array([((aug["box"]=="A")&(aug["date"]==d)).sum() for d in date_arr])
    day_count_B=np.array([((aug["box"]=="B")&(aug["date"]==d)).sum() for d in date_arr])
    day_count_C=np.array([((aug["box"]=="C")&(aug["date"]==d)).sum() for d in date_arr])
    day_cluster=np.array([(cluster&(aug["date"]==d)).sum() for d in date_arr])
    ax=axes[0]
    ax.plot(date_arr, day_count_A, "ro-", label="Box A row count")
    ax.plot(date_arr, day_count_B, "bs-", label="Box B")
    ax.plot(date_arr, day_count_C, "g^-", label="Box C")
    ax.set_ylabel("rows per day"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("2025-08 daily row count per box (Box A drop = HV protection mode)")
    ax.tick_params(axis='x', rotation=45)

    ax=axes[1]
    hv_a_perday=[]; hv_b_perday=[]; hv_c_perday=[]
    for d in date_arr:
        for box, ls in [("A",hv_a_perday),("B",hv_b_perday),("C",hv_c_perday)]:
            m=(aug["box"]==box)&(aug["date"]==d)
            ls.append(np.median(hv[m]) if m.any() else float('nan'))
    ax.plot(date_arr, hv_a_perday, "ro-", label="Box A HV median")
    ax.plot(date_arr, hv_b_perday, "bs-", label="Box B")
    ax.plot(date_arr, hv_c_perday, "g^-", label="Box C")
    ax.axhline(-1100, color='k', ls='--', alpha=0.5, label='Stage 1 bound')
    ax.axhline(-900, color='k', ls='--', alpha=0.5)
    ax.set_ylabel("median HV (V)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("2025-08 daily median HV per box")
    ax.tick_params(axis='x', rotation=45)

    ax=axes[2]
    ax.bar(date_arr, day_cluster, color='purple', alpha=0.7)
    ax.set_ylabel("cluster rows per day"); ax.set_xlabel("date")
    ax.grid(alpha=0.3)
    ax.set_title("Cluster row count per day")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    out="plots/diag_cluster_202508.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
