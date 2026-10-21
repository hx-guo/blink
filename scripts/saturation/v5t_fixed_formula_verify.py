#!/usr/bin/env python3
"""Calibrate the time-dependent v5t formula ONCE, then verify with zero refit.

Formula:
    C(det, |mlat|, t) = s0_det * g(t) * [1 + k(t) * max(0,|mlat|-20)^2]
      g(t) = 1 - beta*(t - t0)            (PMT outgassing, shared across dets)
      k(t) = k0 + k1*(t - t0)             (solar modulation of CR)

Calibration:
  - s0_det, g(t): from per-day s_det stored in npz (separable, verified).
  - k(t): sample cache, estimate k per year from high-|mlat| resid, fit linear.

Verification (the honest test): recompute residual-AFTER on sampled data using
the FIXED analytic formula (no per-day lookup), color by |mlat| and year. If the
three clouds flatten with globally-fit constants, the closed-form calibration
holds and isn't circular.

Output: plots/v5t_fixed_verify.png  + printed parameters
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0
MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]


def unwrap_large_v2(pho, large, wide, sci, l_cycles, dt, C):
    pho=np.asarray(pho,float); large=np.asarray(large,float); wide=np.asarray(wide,float)
    sci=np.asarray(sci,float); L=np.asarray(l_cycles,float)*L_CYCLES_TO_SEC
    lf=1.0-np.asarray(dt,float)/np.asarray(l_cycles,float)
    predicted=pho-(wide+(sci+C)*L)/lf
    n=np.maximum(np.round((predicted-large)/1024.0).astype(int),0)
    maxa=pho-wide; lc=large+n*1024.0; over=lc>maxa
    if over.any():
        nmax=np.maximum(np.floor((maxa-large)/1024.0).astype(int),0)
        lc=large+np.where(over,nmax,n)*1024.0
    return lc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/Volumes/Graphite/work/blink/blink_clean_relaxed")
    ap.add_argument("--full-npz", default="n_below_study/v5_npz/v5_agg_full.npz")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    ap.add_argument("--rowgroups-per-file", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--output", default="plots/v5t_fixed_verify.png")
    args=ap.parse_args()

    z=np.load(args.full_npz)
    dates=z["dates"]; s_det_daily=z["s_det_daily"]
    dt=np.array([np.datetime64(d) for d in dates])
    t0=dt[0]
    t_years=(dt-t0).astype("timedelta64[D]").astype(float)/365.25

    # --- Calibrate g(t) + s0_det from npz ---
    s_flat=s_det_daily.reshape(len(dates),18)
    # monthly-median shared curve normalized to first-180-day baseline
    base180=np.nanmedian(s_flat[:180],axis=0)        # (18,)
    norm=s_flat/base180[None,:]
    # g(t) per day = median across dets of normalized value
    g_day=np.nanmedian(norm,axis=1)                  # (n_days,)
    good=np.isfinite(g_day)
    # linear g(t)=1-beta*t, anchored through (0,1) and the late-time median so the
    # mid-cycle dip (e.g. 2022) can't bias the slope steep the way plain lstsq did.
    tg = t_years[good]; gg = g_day[good]
    l_m = tg > tg.max() - 1.0
    g_l = np.nanmedian(gg[l_m]); t_l = np.nanmedian(tg[l_m])
    beta = float((1.0 - g_l) / t_l)
    def g_of(ty): return 1.0 - beta * ty
    s0_det=np.full(18,np.nan)
    for j in range(18):
        col=s_flat[:,j]/g_of(t_years)
        s0_det[j]=np.nanmedian(col)
    print(f"[calib] g(t)=1-{beta:.4f}*t (linear) -> {g_of(t_years[-1]):.3f} at end")
    print(f"[calib] s0_det (t0 sensitivity):")
    for bi,box in enumerate("ABC"):
        print("   "+box+": "+" ".join(f"{s0_det[bi*6+d]:6.1f}" for d in range(6)))

    date_to_i={d:i for i,d in enumerate(dates)}
    grid=np.load(args.aacgm_grid)
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=[f for f in sorted(glob.glob(os.path.join(args.cache_dir,"clean_relaxed_20*.parquet"))) if "sample" not in f]

    # --- Sample cache, estimate k per year + recompute residual two ways ---
    rec={k:[] for k in ["sci","resid_fix","resid_perday","absmlat","year","ty"]}
    k_year_num={}; k_year_den={}
    for f in files:
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,args.rowgroups_per_file).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
            am=np.where(np.isnan(am),0.0,am)
            mt=np.maximum(0.0,am-B_THRESHOLD)**2
            # per-day s_det (npz) and per-day t
            s_perday=np.full(len(df),np.nan); ty_row=np.zeros(len(df))
            for date,idx in df.groupby("date").groups.items():
                di=date_to_i.get(date)
                if di is None: continue
                sd=s_det_daily[di]; sub=df.loc[idx]
                ai=np.asarray(idx)
                ty_row[ai]=t_years[di]
                for bi,box in enumerate("ABC"):
                    for det in range(6):
                        m=(sub["box"].values==box)&(sub["det"].values==det)
                        if np.isfinite(sd[bi,det]): s_perday[ai[m]]=sd[bi,det]
            # det index + s0 per row
            box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
            detid=box_idx*6+df["det"].values
            s0_row=s0_det[detid]
            s_fix=s0_row*g_of(ty_row)

            pho=df["PHO"].astype(float).values; large=df["Large"].astype(float).values
            wide=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            L=lc*L_CYCLES_TO_SEC; lf=1.0-dtv/lc

            # resid_v2 (C=150) for k estimation
            lv2=unwrap_large_v2(pho,large,wide,sci,lc,dtv,150.0)
            base_v2=(pho-lv2)*lf/L-wide/L
            resid_v2=base_v2-sci
            clean=(wide/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(resid_v2)&(np.abs(resid_v2)<2000)
            # accumulate k(year): (resid_v2/s_fix - 1)/mlat^2 over high mlat
            hi=clean&(am>=35)&(s_fix>1e-3)
            yr=df["date"].str[:4].astype(int).values
            kk=(resid_v2/s_fix-1.0)/np.maximum(mt,1.0)
            for y in np.unique(yr[hi]):
                msk=hi&(yr==y)
                k_year_num.setdefault(y,[]).append(np.nansum(kk[msk]))
                k_year_den.setdefault(y,[]).append(msk.sum())

            rec["sci"].append(sci); rec["absmlat"].append(am); rec["year"].append(yr)
            rec["ty"].append(ty_row); rec["s_fix_tmp_pho"]=None  # placeholder
            # store raw to recompute residuals after k(t) known
            rec.setdefault("_pho",[]).append(pho); rec.setdefault("_large",[]).append(large)
            rec.setdefault("_wide",[]).append(wide); rec.setdefault("_lc",[]).append(lc)
            rec.setdefault("_dt",[]).append(dtv); rec.setdefault("_mt",[]).append(mt)
            rec.setdefault("_sfix",[]).append(s_fix); rec.setdefault("_sperday",[]).append(s_perday)
        print(f"  {os.path.basename(f)}: scanned")

    # k per year and linear k(t)
    years=np.array(sorted(k_year_num))
    k_year=np.array([np.sum(k_year_num[y])/max(np.sum(k_year_den[y]),1) for y in years])
    ty_year=np.array([ (np.datetime64(f"{y}-07-01")-t0).astype("timedelta64[D]").astype(float)/365.25 for y in years])
    # k(t): solar cycle, SINGLE sinusoid (P=11yr, 3 param). 2nd harmonic removed
    # (it overfit the mid-cycle and made the 2026 extrapolation curl up).
    P_SOLAR=11.0; w=2*np.pi/P_SOLAR
    Amat=np.column_stack([np.ones_like(ty_year),np.cos(w*ty_year),np.sin(w*ty_year)])
    c3,*_=np.linalg.lstsq(Amat,k_year,rcond=None)
    kcoef=np.array([c3[0],c3[1],c3[2],0.0,0.0])  # keep 5 slots; 2nd-harmonic coeffs zeroed
    def k_of(ty):
        return kcoef[0]+kcoef[1]*np.cos(w*ty)+kcoef[2]*np.sin(w*ty)
    print(f"\n[calib] k(year):")
    for y,kv in zip(years,k_year):
        tyy=(np.datetime64(f'{y}-07-01')-t0).astype('timedelta64[D]').astype(float)/365.25
        print(f"   {y}: k={kv:.5f}  (harmonic fit {k_of(tyy):.5f})")
    print(f"[calib] k(t)=fundamental+2nd harmonic, P=11yr, coeffs={kcoef}")

    # --- recompute residuals with FIXED formula (zero refit) ---
    pho=np.concatenate(rec["_pho"]); large=np.concatenate(rec["_large"])
    wide=np.concatenate(rec["_wide"]); lc=np.concatenate(rec["_lc"])
    dtv=np.concatenate(rec["_dt"]); mt=np.concatenate(rec["_mt"])
    s_fix=np.concatenate(rec["_sfix"]); s_perday=np.concatenate(rec["_sperday"])
    sci=np.concatenate(rec["sci"]); am=np.concatenate(rec["absmlat"])
    yr=np.concatenate(rec["year"]); ty=np.concatenate(rec["ty"])
    L=lc*L_CYCLES_TO_SEC; lf=1.0-dtv/lc

    def redo(Cpr):
        lv3=unwrap_large_v2(pho,large,wide,sci,lc,dtv,Cpr)
        maxle=pho-((sci+MIN_C_SLACK)*L+wide)/lf
        n3=np.round((lv3-large)/1024).astype(int)
        nmax=np.maximum(np.floor((maxle-large)/1024.0).astype(int),0)
        lv5=large+np.where(n3>nmax,nmax,n3)*1024.0
        base=(pho-lv5)*lf/L-wide/L
        return base-sci-Cpr

    C_fix=s_fix*(1.0+k_of(ty)*mt)
    C_perday=s_perday*(1.0+0.00188*mt)
    resid_fix=redo(C_fix)
    resid_perday=redo(C_perday)

    # --- freeze calibration to disk (global C0 absorbs residual offset) ---
    C0=float(np.nanmedian(resid_fix[np.isfinite(resid_fix)&(sci>0)]))
    calib_path="n_below_study/v5_npz/v5t_calib.npz"
    np.savez(calib_path, s0_det=s0_det, beta=beta, w=w, k_coeffs=kcoef,
             t0=str(dates[0]), C0=C0)
    print(f"\n[calib] saved {calib_path}: C0={C0:+.1f} (global offset absorbed)")

    ok=np.isfinite(resid_fix)&np.isfinite(resid_perday)&(sci>0)&np.isfinite(s_fix)&np.isfinite(s_perday)
    idx=np.where(ok)[0]
    if len(idx)>args.max_points:
        idx=np.random.RandomState(0).choice(idx,args.max_points,replace=False)
    sci=sci[idx]; am=am[idx]; yr=yr[idx]; resid_fix=resid_fix[idx]; resid_perday=resid_perday[idx]
    print(f"\nPlotting {len(sci):,} points")

    LO,HI=30.0,10000.0; YL,YH=-400,800
    def inb(r): return (sci>=LO)&(sci<=HI)&(r>=YL)&(r<=YH)
    fig,axes=plt.subplots(2,2,figsize=(20,14))
    shuf=np.random.RandomState(1).permutation(len(sci))
    panels=[
        (axes[0,0],resid_perday,am,"plasma","per-day refit  (colored |mlat|)","|mlat|",0,60),
        (axes[0,1],resid_fix,am,"plasma","FIXED formula s0·g(t)·[1+k(t)m²]  (colored |mlat|)","|mlat|",0,60),
        (axes[1,0],resid_perday,yr,"turbo","per-day refit  (colored year)","year",None,None),
        (axes[1,1],resid_fix,yr,"turbo","FIXED formula  (colored year)","year",None,None),
    ]
    for ax,r,c,cm,title,clab,vmn,vmx in panels:
        m=inb(r); o=shuf[np.isin(shuf,np.where(m)[0])]
        sca=ax.scatter(sci[o],r[o],c=c[o],cmap=cm,s=2,alpha=0.5,vmin=vmn,vmax=vmx,
                       rasterized=True,edgecolor="none")
        ax.axhline(0,color="k",lw=1.5); ax.set_xscale("log")
        ax.set_xlim(LO,HI); ax.set_ylim(YL,YH)
        ax.set_xlabel("Sci_1s observed (cnt/s)"); ax.set_ylabel("residual AFTER (cnt/s)")
        med=np.median(r[m]); ax.set_title(f"{title}\nmedian={med:+.1f}",fontsize=11)
        fig.colorbar(sca,ax=ax).set_label(clab); ax.grid(True,alpha=0.3,which="both")
    fig.suptitle(f"Fixed-formula (1 calibration) vs per-day refit  |  g(t)=1-{beta:.4f}t (linear), k(t)=single sinusoid",
                 fontsize=13,fontweight="bold",y=1.0)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(args.output,dpi=110,bbox_inches="tight"); plt.close(fig)
    print(f"Saved {args.output}")

    # numeric: high-mlat residual median per year, fixed vs perday
    print("\n=== high-|mlat|(>=35) residual median per year: perday vs fixed ===")
    for y in np.unique(yr):
        m=(yr==y)&(am>=35)
        if m.sum()<50: continue
        print(f"  {y}: perday={np.median(resid_perday[m]):+6.1f}  fixed={np.median(resid_fix[m]):+6.1f}")


if __name__=="__main__":
    main()
