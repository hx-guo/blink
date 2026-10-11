#!/usr/bin/env python3
"""Side-by-side comparison: v5t (s0·g·[1+k·mlat²]+C0, 23 params) vs
v6_acd (a_det+b_det·ACD_sum, 36 params, no PM filter).

Both pipelines run on the SAME sampled cache rows; same unwrap + event-balance
cap; same metrics (blob/main, below-y=x, MAD, median residual).
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon","ACD_sum"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
B_THRESHOLD = 20.0


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


def apply_pipeline(pho,lg,wd,sci,lc,dtv,C):
    """v5t-style unwrap + event-balance cap. Returns (resid, sci_obs_arr, n_below_added)."""
    LL=lc*L; lf=1.0-dtv/lc
    lv3=unwrap_v2(pho,lg,wd,sci,lc,dtv,C)
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
    n3=np.round((lv3-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
    lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
    base=(pho-lv5)*lf/LL-wd/LL
    resid=base-sci-C
    ok=np.isfinite(base)&(sci>0)&(base>0)
    n_below=(base<sci)[ok].sum()
    return resid[ok], sci[ok], n_below


def metrics(resid, sci_obs, n_below_total, N_total):
    med=np.median(resid)
    mad=np.median(np.abs(resid-med))
    blob=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-300)&(resid<=-50)).sum()
    main=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-50)&(resid<=100)).sum()
    return dict(median=med, MAD=mad,
                blob_main=blob/max(main,1)*100,
                below=n_below_total/N_total*100,
                N=N_total)


def main():
    # Load v5t calib
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    print(f"v5t: beta={beta:.4f}, C0={C0:+.2f}")

    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # PASS 1: collect (C_truth, ACD_sum, detid) for v6_acd fit, no PM filter
    print("\n=== PASS 1: collect v6_acd training data (no PM filter) ===")
    fit_acd=[]; fit_C=[]; fit_det=[]
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
            lv=unwrap_v2(pho,lg,wd,sci,lc,dtv,150.0)
            base=(pho-lv)*lf/LL-wd/LL
            C_truth=base-sci
            clean=(wd/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(C_truth)&(np.abs(C_truth)<500)
            fit_acd.append(acd[clean]); fit_C.append(C_truth[clean]); fit_det.append(detid[clean])
        print(f"  {os.path.basename(f)}: scanned")

    acd_all=np.concatenate(fit_acd); C_all=np.concatenate(fit_C); det_all=np.concatenate(fit_det)
    print(f"\nFit rows: {len(acd_all):,}")

    print("\n=== fit v6_acd: C = a_det + b_det*ACD per det ===")
    a_det=np.zeros(18); b_det=np.zeros(18)
    for d in range(18):
        m=(det_all==d)
        if m.sum()<1000: continue
        x=acd_all[m]; y=C_all[m]
        coef=np.polyfit(x,y,1); b_det[d],a_det[d]=coef
    print("   det   a       b")
    for d in range(18):
        print(f"   {'ABC'[d//6]}{d%6}  {a_det[d]:+6.2f}  {b_det[d]:+.4e}")

    # PASS 2: apply both models on the SAME data, collect everything
    print("\n=== PASS 2: apply v5t and v6_acd on same data ===")
    r5_all=[]; sci5_all=[]; n_below5=0
    r6_all=[]; sci6_all=[]; n_below6=0
    for f in files:
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,4).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
            am=np.where(np.isnan(am),0.0,am)
            mt=np.maximum(0.0,am-B_THRESHOLD)**2
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
            C_acd=a_det[detid]+b_det[detid]*acd

            r5,s5,nb5=apply_pipeline(pho,lg,wd,sci,lc,dtv,C_v5)
            r6,s6,nb6=apply_pipeline(pho,lg,wd,sci,lc,dtv,C_acd)
            r5_all.append(r5); sci5_all.append(s5); n_below5+=nb5
            r6_all.append(r6); sci6_all.append(s6); n_below6+=nb6
        print(f"  {os.path.basename(f)}: applied",flush=True)

    r5=np.concatenate(r5_all); sci5=np.concatenate(sci5_all)
    r6=np.concatenate(r6_all); sci6=np.concatenate(sci6_all)
    m5=metrics(r5,sci5,n_below5,len(r5))
    m6=metrics(r6,sci6,n_below6,len(r6))

    print(f"\n=== METRICS ===")
    print(f"             v5t     v6_acd")
    print(f"  N         {m5['N']:>8,}  {m6['N']:>8,}")
    print(f"  median    {m5['median']:+7.2f}  {m6['median']:+7.2f}")
    print(f"  MAD       {m5['MAD']:>7.2f}  {m6['MAD']:>7.2f}")
    print(f"  blob/main {m5['blob_main']:>6.2f}%  {m6['blob_main']:>6.2f}%")
    print(f"  below-y=x {m5['below']:>6.4f}%  {m6['below']:>6.4f}%")

    # ─────── Side-by-side plot ───────
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    ix5=np.random.RandomState(0).choice(len(sci5),min(250000,len(sci5)),replace=False)
    ix6=np.random.RandomState(0).choice(len(sci6),min(250000,len(sci6)),replace=False)

    LO,HI=30,10000; YL,YH=-400,800

    # Top row: Sci_rec vs Sci_obs (log-log)
    sci_rec_5=r5+sci5; sci_rec_6=r6+sci6
    for ax, sci_obs, sci_rec, ix, name, m in [
        (axes[0,0], sci5, sci_rec_5, ix5, "v5t (23 params: s0·g·[1+k·mlat²]+C0)", m5),
        (axes[0,1], sci6, sci_rec_6, ix6, "v6_acd (36 params: a_det+b_det·ACD)", m6),
    ]:
        ax.scatter(sci_obs[ix],sci_rec[ix],s=1,alpha=0.25,c='steelblue',rasterized=True)
        ax.plot([1,10000],[1,10000],'r--',lw=1.2,label='y=x')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1,10000); ax.set_ylim(1,10000)
        ax.set_xlabel(r"$\rm Sci_{obs}$ (cnt/s)"); ax.set_ylabel(r"$\rm Sci_{rec}$ (cnt/s)")
        ax.set_title(f"{name}\nblob/main = {m['blob_main']:.2f}%  |  below y=x = {m['below']:.3f}%",
                     fontsize=11)
        ax.grid(alpha=0.3,which='both'); ax.legend()

    # Bottom row: residual vs sci_obs
    for ax, sci_obs, resid, ix, name, m in [
        (axes[1,0], sci5, r5, ix5, "v5t", m5),
        (axes[1,1], sci6, r6, ix6, "v6_acd", m6),
    ]:
        mask=(sci_obs>=LO)&(sci_obs<=HI)&(resid>=YL)&(resid<=YH)
        ix_use=ix[np.isin(ix,np.where(mask)[0])]
        ax.scatter(sci_obs[ix_use],resid[ix_use],s=1,alpha=0.25,c='darkorange',rasterized=True)
        ax.axhline(0,color='k',lw=1.5)
        # blob band
        ax.axhspan(-300,-50,alpha=0.08,color='r',label='blob zone')
        ax.set_xscale('log'); ax.set_xlim(LO,HI); ax.set_ylim(YL,YH)
        ax.set_xlabel(r"$\rm Sci_{obs}$ (cnt/s)"); ax.set_ylabel("residual (cnt/s)")
        ax.set_title(f"{name}: median = {m['median']:+.1f}, MAD = {m['MAD']:.1f}",fontsize=11)
        ax.grid(alpha=0.3,which='both'); ax.legend(loc='upper left')

    fig.suptitle(f"v5t vs v6_acd PHO-conservation reconstruction (full mission, ~{m5['N']/1e6:.1f} M rows)",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out="plots/compare_v5_vs_v6.png"
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
