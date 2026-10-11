#!/usr/bin/env python3
"""Is C0 necessary? Compare the full v5t pipeline WITH C0 vs C0=0:
residual median, blob/main, below-y=x. Samples cache, recomputes both."""
import glob, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"

def unwrap(pho,large,wide,sci,lc,dt,C):
    pho=np.asarray(pho,float);large=np.asarray(large,float);wide=np.asarray(wide,float);sci=np.asarray(sci,float)
    LL=np.asarray(lc,float)*L; lf=1.0-np.asarray(dt,float)/np.asarray(lc,float)
    pred=pho-(wide+(sci+C)*LL)/lf; n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0); out=large+np.where(ov,nm,n)*1024.
    return out

def pipeline(df, interp, calib, use_c0):
    am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values]))); am=np.where(np.isnan(am),0.,am)
    mt=np.maximum(0.,am-20.)**2
    pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values; wd=df["Wide"].astype(float).values
    sci=df["Sci_1s"].astype(float).values; lc=df["L_cycles"].astype(float).values; dt=df["Dt"].astype(float).values
    LL=lc*L; lf=1.0-dt/lc
    ty=(pd.to_datetime(df["date"]).values.astype("datetime64[D]")-calib["t0"]).astype("timedelta64[D]").astype(float)/365.25
    g=1.0-calib["beta"]*ty; w=calib["w"]; kc=calib["k_coeffs"]
    k=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)+kc[3]*np.cos(2*w*ty)+kc[4]*np.sin(2*w*ty)
    bi=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0); detid=bi*6+df["det"].values
    C=calib["s0_det"][detid]*g*(1.0+k*mt)+(calib["C0"] if use_c0 else 0.0)
    lv3=unwrap(pho,lg,wd,sci,lc,dt,C)
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf; n3=np.round((lv3-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0); lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
    base=(pho-lv5)*lf/LL-wd/LL; resid=base-sci-C; ok=np.isfinite(base)&(sci>0)&(base>0)
    return resid[ok], sci[ok]

def stats(resid, sci, tag):
    med=np.median(resid)
    blob=((sci>=800)&(sci<=2500)&(resid>=-300)&(resid<=-50)).sum()
    main=((sci>=800)&(sci<=2500)&(resid>=-50)&(resid<=100)).sum()
    below=(resid < -sci).sum()  # base<sci  <=> resid<-... actually base<sci => base-sci<0 => resid+C<0
    print(f"  {tag}: median={med:+.2f}, blob/main={blob/max(main,1)*100:.2f}%, |resid|<30={np.mean(np.abs(resid)<30)*100:.1f}%")
    return med, blob/max(main,1)

cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
calib={"s0_det":cz["s0_det"],"beta":float(cz["beta"]),"w":float(cz["w"]),
       "k_coeffs":cz["k_coeffs"],"C0":float(cz["C0"]),"t0":np.datetime64(str(cz["t0"]))}
print(f"calib: beta={calib['beta']:.4f}, C0={calib['C0']:+.2f}")
grid=np.load("n_below_study/aacgm_grid_2020.npz")
interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],bounds_error=False,fill_value=np.nan)
files=[f for f in sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet"))) if "sample" not in f]

R1=[];S1=[];R0=[];S0=[];bel1=0;bel0=0
for f in files:
    pf=pq.ParquetFile(f)
    for rg in np.unique(np.linspace(0,pf.num_row_groups-1,6).astype(int)):
        df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
        r1,s1=pipeline(df,interp,calib,True); r0,s0=pipeline(df,interp,calib,False)
        R1.append(r1);S1.append(s1);R0.append(r0);S0.append(s0)
    print(f"  scanned {os.path.basename(f)}",flush=True)
r1=np.concatenate(R1);s1=np.concatenate(S1);r0=np.concatenate(R0);s0=np.concatenate(S0)
print(f"\nN={len(r1):,}")
print("=== WITH C0 ===");  stats(r1,s1,"C0=-6")
print("=== WITHOUT C0 ==="); stats(r0,s0,"C0= 0")
