#!/usr/bin/env python3
"""Test the ACD-only model: C(row,det) = a_det + b_det * ACD_sum.

NO mlat, NO g(t), NO k(t), NO C0 — fully in-situ, 36 params total
(18 intercepts + 18 slopes), against v5t's 23 params.

Pipeline mirrors v5t verify exactly:
  1. Compute C ground truth per row from low-wrap region:
     for rows where Large is far from a 1024 boundary, no unwrap ambiguity,
     so C_truth = base_raw - sci with base_raw using raw Large.
  2. Fit a_det, b_det per det from clean rows: C_truth ~ a + b * ACD_sum.
  3. Run full pipeline (unwrap with C_acd, event-balance cap, Sci_rec).
  4. Report identical metrics as v5t: blob/main, below-y=x, residual median, MAD.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon","ACD_sum","PM_0"]
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
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # Pass 1: collect (C_truth, ACD_sum, detid) per row using initial C=150 unwrap
    print("=== PASS 1: collect training data ===")
    fit_acd=[]; fit_C=[]; fit_det=[]
    for f in files:
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,4).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
            wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            acd=df["ACD_sum"].astype(float).values; pm0=df["PM_0"].astype(float).values
            box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
            detid=box_idx*6+df["det"].values
            LL=lc*L; lf=1.0-dtv/lc

            # Initial unwrap with neutral C=150
            lv=unwrap_v2(pho,lg,wd,sci,lc,dtv,150.0)
            base=(pho-lv)*lf/LL-wd/LL
            C_truth=base-sci  # per-row C from conservation

            clean=(wd/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(C_truth)
            clean&=(np.abs(C_truth)<500)&(pm0<5)  # exclude SAA and gross outliers
            fit_acd.append(acd[clean]); fit_C.append(C_truth[clean])
            fit_det.append(detid[clean])
        print(f"  {os.path.basename(f)}: scanned")

    acd_all=np.concatenate(fit_acd); C_all=np.concatenate(fit_C); det_all=np.concatenate(fit_det)
    print(f"\nFit data: {len(acd_all):,} clean rows")

    # Fit a_det, b_det per det
    print("\n=== fit C = a_det + b_det * ACD per det ===")
    a_det=np.zeros(18); b_det=np.zeros(18)
    for d in range(18):
        m=(det_all==d)
        if m.sum()<1000: continue
        # OLS
        x=acd_all[m]; y=C_all[m]
        coef=np.polyfit(x,y,1); b_det[d],a_det[d]=coef
    print("   det  a_det    b_det")
    for d in range(18):
        box='ABC'[d//6]; di=d%6
        print(f"   {box}{di}  {a_det[d]:+6.2f}  {b_det[d]:+.4e}")
    print(f"\n   a_det median {np.median(a_det):+.1f} ± {np.std(a_det):.1f}")
    print(f"   b_det median {np.median(b_det):+.4e} ± {np.std(b_det):.4e}")

    # PASS 2: apply v6_acd model to all data, run full pipeline
    print("\n=== PASS 2: apply C_acd, run unwrap + event-balance cap ===")
    all_resid=[]; all_sci=[]; all_below=[]
    for f in files:
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,4).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
            wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            acd=df["ACD_sum"].astype(float).values
            box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
            detid=box_idx*6+df["det"].values
            LL=lc*L; lf=1.0-dtv/lc

            # C from v6_acd model
            C_acd=a_det[detid]+b_det[detid]*acd

            # Unwrap with C_acd
            lv3=unwrap_v2(pho,lg,wd,sci,lc,dtv,C_acd)
            # Event-balance cap (v5t-style)
            mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
            n3=np.round((lv3-lg)/1024).astype(int)
            nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
            lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
            base=(pho-lv5)*lf/LL-wd/LL
            resid=base-sci-C_acd

            ok=np.isfinite(base)&(sci>0)&(base>0)
            all_resid.append(resid[ok]); all_sci.append(sci[ok])
            # below-y=x: base<sci (i.e. recovered sci < observed sci, physically impossible)
            all_below.append((base<sci)[ok].sum())
        print(f"  {os.path.basename(f)}: applied",flush=True)

    resid=np.concatenate(all_resid); sci_obs=np.concatenate(all_sci)
    n_below=sum(all_below); N=len(resid)

    print(f"\n=== v6_acd METRICS (compare to v5t) ===")
    print(f"  N = {N:,}")
    print(f"  median residual: {np.median(resid):+.2f}  (v5t: 0.0)")
    print(f"  MAD residual: {np.median(np.abs(resid-np.median(resid))):.2f}  (v5t: 19)")
    print(f"  |resid|<30: {np.mean(np.abs(resid)<30)*100:.1f}%")
    blob=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-300)&(resid<=-50)).sum()
    main=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-50)&(resid<=100)).sum()
    print(f"  blob/main: {blob/max(main,1)*100:.2f}%  (v5t: 10.84%)")
    print(f"  below-y=x: {n_below/N*100:.4f}%  (v5t: 0.047%)")

    # Sci_rec = base - C; for plot
    Sci_rec=resid+sci_obs  # because resid=base-sci-C → base=resid+sci+C → Sci_rec=base-C=resid+sci

    # Plot: Sci_rec vs Sci_obs scatter (mimic v5t conservation plot)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    ix=np.random.RandomState(0).choice(len(sci_obs),min(200000,len(sci_obs)),replace=False)

    ax=axes[0]
    ax.scatter(sci_obs[ix],Sci_rec[ix],s=1,alpha=0.3,c='b',rasterized=True)
    ax.plot([1,10000],[1,10000],'r--',lw=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1,10000); ax.set_ylim(1,10000)
    ax.set_xlabel('Sci_obs'); ax.set_ylabel('Sci_rec')
    ax.set_title(f"v6_acd: C = a_det+b_det·ACD\nblob/main={blob/max(main,1)*100:.1f}%, below={n_below/N*100:.4f}%")
    ax.grid(alpha=0.3)

    ax=axes[1]
    LO,HI=30,10000; YL,YH=-400,800
    m=(sci_obs>=LO)&(sci_obs<=HI)&(resid>=YL)&(resid<=YH)
    ix2=np.random.RandomState(1).choice(np.where(m)[0],min(200000,m.sum()),replace=False)
    ax.scatter(sci_obs[ix2],resid[ix2],s=1,alpha=0.3,c='b',rasterized=True)
    ax.axhline(0,color='k',lw=1.5); ax.set_xscale('log')
    ax.set_xlim(LO,HI); ax.set_ylim(YL,YH)
    ax.set_xlabel('Sci_obs'); ax.set_ylabel('residual')
    ax.set_title(f"residual: median={np.median(resid):+.1f}, MAD={np.median(np.abs(resid-np.median(resid))):.1f}")
    ax.grid(alpha=0.3,which='both')

    plt.tight_layout()
    out="plots/v6_acd_only_verify.png"
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
