#!/usr/bin/env python3
"""Diagnose the violet-circled cluster in compare_v5_vs_v6_v5.png.

Tight cluster definition: Sci_obs 1100-1500, Sci_rec 500-800 (residual -1000 to -300).
Look at where (Lat/Lon, year-month, |mlat|) and what (ACD_sum, Wide/PHO ratio) it is.
"""
from __future__ import annotations
import glob, os
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon","ACD_sum","PM_0","PM_1","PM_2"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"


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
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # Collect tight-cluster rows + control sample
    cluster_rows={k:[] for k in ["date","lat","lon","mlat","sci","sci_rec","resid","acd","pm0","widepho","det"]}
    main_rows={k:[] for k in ["lat","lon","mlat","acd"]}  # control: main band
    for f in files:
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,4).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
            am=np.where(np.isnan(am),0.0,am)
            mt=np.maximum(0.0,am-20.0)**2
            d_arr=np.array([np.datetime64(d) for d in df["date"].values])
            ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
            g_t=1.0-beta*ty
            k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
            box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
            detid=box_idx*6+df["det"].values
            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
            wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            acd=df["ACD_sum"].astype(float).values

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

            ok=np.isfinite(sci_rec)&(sci>0)&(base>0)
            widepho=wd/np.maximum(pho,1)
            # Tight cluster: Sci_obs 1100-1500, Sci_rec 500-800
            cl=ok&(sci>=1100)&(sci<=1500)&(sci_rec>=500)&(sci_rec<=800)
            # Main control: Sci_obs 1100-1500, Sci_rec 1000-1500 (on y=x band)
            mn=ok&(sci>=1100)&(sci<=1500)&(sci_rec>=1000)&(sci_rec<=1500)

            if cl.any():
                cluster_rows["date"].extend(df["date"].values[cl])
                cluster_rows["lat"].extend(df["Lat"].values[cl])
                cluster_rows["lon"].extend(df["Lon"].values[cl])
                cluster_rows["mlat"].extend(am[cl])
                cluster_rows["sci"].extend(sci[cl])
                cluster_rows["sci_rec"].extend(sci_rec[cl])
                cluster_rows["resid"].extend(resid[cl])
                cluster_rows["acd"].extend(acd[cl])
                cluster_rows["pm0"].extend(df["PM_0"].values[cl])
                cluster_rows["widepho"].extend(widepho[cl])
                cluster_rows["det"].extend(detid[cl])
            if mn.any():
                main_rows["lat"].extend(df["Lat"].values[mn])
                main_rows["lon"].extend(df["Lon"].values[mn])
                main_rows["mlat"].extend(am[mn])
                main_rows["acd"].extend(acd[mn])
        print(f"  {os.path.basename(f)}: scanned",flush=True)

    cl=cluster_rows; mn=main_rows
    n_cl=len(cl["date"]); n_mn=len(mn["lat"])
    print(f"\n=== TIGHT cluster (Sci_obs 1100-1500, Sci_rec 500-800) ===")
    print(f"  cluster: {n_cl:,} rows")
    print(f"  main control (same Sci_obs, Sci_rec on y=x): {n_mn:,} rows")
    print(f"  cluster/main ratio: {n_cl/max(n_mn,1)*100:.2f}%")

    if n_cl == 0:
        print("No cluster rows. Bounds may need adjusting.")
        return

    sci_arr=np.array(cl["sci"]); sci_rec_arr=np.array(cl["sci_rec"])
    print(f"  Sci_obs distribution:   min={sci_arr.min():.0f}, med={np.median(sci_arr):.0f}, max={sci_arr.max():.0f}")
    print(f"  Sci_rec distribution:   min={sci_rec_arr.min():.0f}, med={np.median(sci_rec_arr):.0f}, max={sci_rec_arr.max():.0f}")
    print(f"  median residual: {np.median(np.array(cl['resid'])):.0f}")

    # ─── date distribution ───
    ym=Counter(d[:7] for d in cl["date"])
    print(f"\n  top 12 year-months in cluster:")
    for k,v in sorted(ym.items(), key=lambda x:-x[1])[:12]:
        print(f"    {k}: {v:>7,} ({v/n_cl*100:.2f}%)")

    # ─── lat / mlat / lon ───
    cl_mlat=np.array(cl["mlat"]); mn_mlat=np.array(mn["mlat"])
    cl_lat=np.array(cl["lat"]); mn_lat=np.array(mn["lat"])
    cl_lon=np.array(cl["lon"]); mn_lon=np.array(mn["lon"])
    print(f"\n  |mlat|: cluster med={np.median(cl_mlat):.1f}, main med={np.median(mn_mlat):.1f}")
    print(f"  Lat:    cluster med={np.median(cl_lat):.1f}, main med={np.median(mn_lat):.1f}")
    print(f"  Lon:    cluster med={np.median(cl_lon):.1f}, main med={np.median(mn_lon):.1f}")

    # ─── ACD/PM/Wide-PHO ratio ───
    cl_acd=np.array(cl["acd"]); mn_acd=np.array(mn["acd"])
    print(f"  ACD_sum: cluster med={np.median(cl_acd):.0f}, main med={np.median(mn_acd):.0f}")
    print(f"  PM_0 distribution in cluster:")
    pm0_cl=np.array(cl["pm0"])
    print(f"    frac PM_0>0: {(pm0_cl>0).mean()*100:.1f}%")
    print(f"    frac PM_0>10: {(pm0_cl>10).mean()*100:.1f}%")
    wp=np.array(cl["widepho"])
    print(f"  Wide/PHO ratio: med={np.median(wp):.3f}, p90={np.percentile(wp,90):.3f}, p99={np.percentile(wp,99):.3f}")
    print(f"    frac Wide/PHO>0.3: {(wp>0.3).mean()*100:.1f}%")

    # ─── per-det distribution ───
    cl_det=np.array(cl["det"])
    det_count=Counter(cl_det)
    print(f"\n  per-det distribution in cluster (expect ~5.5% per det):")
    for d in range(18):
        c=det_count.get(d,0)
        print(f"    {'ABC'[d//6]}{d%6}: {c:>7,} ({c/n_cl*100:.2f}%)")

    # ─── plot lat/lon spatial map ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0,0].hist(cl_mlat, bins=60, density=True, alpha=0.6, color='r', label='cluster')
    axes[0,0].hist(mn_mlat, bins=60, density=True, alpha=0.6, color='b', label='main control')
    axes[0,0].set_xlabel('|mlat| (deg)'); axes[0,0].set_ylabel('density')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3); axes[0,0].set_title('|mlat| distribution')

    axes[0,1].hist(cl_acd, bins=80, density=True, alpha=0.6, color='r', label='cluster')
    axes[0,1].hist(mn_acd, bins=80, density=True, alpha=0.6, color='b', label='main control')
    axes[0,1].set_xlabel('ACD_sum (cnt/s)'); axes[0,1].set_ylabel('density')
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3); axes[0,1].set_title('ACD_sum distribution')
    axes[0,1].set_yscale('log')

    axes[1,0].scatter(cl_lon[::20], cl_lat[::20], s=2, c='r', alpha=0.4, label='cluster')
    axes[1,0].set_xlabel('Lon'); axes[1,0].set_ylabel('Lat')
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
    axes[1,0].set_title(f'Cluster spatial: Lon vs Lat (subsample)')
    axes[1,0].set_xlim(0, 360); axes[1,0].set_ylim(-50, 50)

    axes[1,1].hist(wp, bins=80, range=(0, 1), density=True, color='r', alpha=0.6)
    axes[1,1].axvline(0.3, color='k', ls='--', label='clean cutoff 0.3')
    axes[1,1].set_xlabel('Wide / PHO'); axes[1,1].set_ylabel('density')
    axes[1,1].legend(); axes[1,1].grid(alpha=0.3); axes[1,1].set_title('Cluster: Wide/PHO ratio')

    plt.tight_layout()
    out="plots/diag_cluster_v6.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
