#!/usr/bin/env python3
"""Is k(t)'s functional form adequate (esp. 2026)?

Extract half-year LS k (the 'truth'), then fit several forms and compare
RMS + the 2026 residual:
  - harmonic P=11yr (current, 5 param)
  - single sinusoid P=11yr (3 param)
  - sinusoid with FREE period (4 param) -> tests whether P=11 is wrong
  - OULU neutron-monitor driven k=a+b*N (2 param) if /tmp/oulu_nm.txt present

Output: plots/diag_kt_form.png
"""
import glob, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

L = 16e-6; NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"


def unwrap(pho, large, wide, sci, lc, dt, C):
    pho=np.asarray(pho,float); large=np.asarray(large,float); wide=np.asarray(wide,float); sci=np.asarray(sci,float)
    LL=np.asarray(lc,float)*L; lf=1.0-np.asarray(dt,float)/np.asarray(lc,float)
    pred=pho-(wide+(sci+C)*LL)/lf
    n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0); out=large+np.where(ov,nm,n)*1024.
    return out


def main():
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],bounds_error=False,fill_value=np.nan)
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz"); C0=float(cz["C0"]); t0=np.datetime64(str(cz["t0"]))

    files=[f for f in sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet"))) if "sample" not in f]
    R=[]; AM=[]; TY=[]
    for f in files:
        pf=pq.ParquetFile(f)
        for rg in np.unique(np.linspace(0,pf.num_row_groups-1,8).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values]))); am=np.where(np.isnan(am),0.,am)
            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values; wd=df["Wide"].astype(float).values
            sc=df["Sci_1s"].astype(float).values; lc=df["L_cycles"].astype(float).values; dt=df["Dt"].astype(float).values
            lf=1.0-dt/lc; lv=unwrap(pho,lg,wd,sc,lc,dt,150.0)
            res=(pho-lv)*lf/(lc*L)-wd/(lc*L)-sc
            ok=(wd/np.maximum(pho,1)<0.3)&(sc>100)&np.isfinite(res)&(np.abs(res)<2000)
            ty=(pd.to_datetime(df["date"]).values.astype("datetime64[D]")-t0).astype("timedelta64[D]").astype(float)/365.25
            R.append(res[ok]); AM.append(am[ok]); TY.append(ty[ok])
        print(f"  scanned {os.path.basename(f)}",flush=True)
    res=np.concatenate(R); amlat=np.concatenate(AM); ty=np.concatenate(TY)

    # half-year LS k (truth)
    tb=np.linspace(0,9,19); kt=[]; ky=[]
    for i in range(len(tb)-1):
        mt=(ty>=tb[i])&(ty<tb[i+1]); eq=res[mt&(amlat<5)]; hm=mt&(amlat>=25)
        if len(eq)<100 or hm.sum()<300: continue
        eqm=np.median(eq); w2=np.maximum(0.,amlat[hm]-20)**2; yv=(res[hm]-C0)/(eqm-C0)-1.0; g=w2>25
        if g.sum()<200: continue
        kt.append(0.5*(tb[i]+tb[i+1])); ky.append(np.sum(w2[g]*yv[g])/np.sum(w2[g]**2))
    kt=np.array(kt); ky=np.array(ky); yr=2017.48+kt
    print(f"\nhalf-year k: {len(ky)} bins")

    fits={}
    # 1. harmonic P=11 (5 param)
    w=2*np.pi/11
    A5=np.column_stack([np.ones_like(kt),np.cos(w*kt),np.sin(w*kt),np.cos(2*w*kt),np.sin(2*w*kt)])
    c5=np.linalg.lstsq(A5,ky,rcond=None)[0]; fits["harmonic P=11 (5p)"]=A5@c5
    # 2. single sinusoid P=11 (3 param)
    A3=np.column_stack([np.ones_like(kt),np.cos(w*kt),np.sin(w*kt)])
    c3=np.linalg.lstsq(A3,ky,rcond=None)[0]; fits["sinusoid P=11 (3p)"]=A3@c3
    # 3. sinusoid free period
    def sinP(t,k0,a,b,P): return k0+a*np.cos(2*np.pi*t/P)+b*np.sin(2*np.pi*t/P)
    try:
        p,_=curve_fit(sinP,kt,ky,p0=[0.0019,0.0003,0.0003,11],bounds=([0.001,-0.01,-0.01,6],[0.003,0.01,0.01,16]),maxfev=20000)
        fits[f"sinusoid P free ={p[3]:.1f}yr (4p)"]=sinP(kt,*p); Pfree=p[3]
    except Exception as e:
        print("Pfree fail",e); Pfree=np.nan
    # 4. OULU driven
    oulu_path="/tmp/oulu_nm.txt"
    if os.path.exists(oulu_path):
        od=[]; ov=[]
        for line in open(oulu_path):
            s=line.strip()
            if not s or s.startswith("#") or ";" not in s: continue
            d,v=s.split(";")
            try: ov.append(float(v)); od.append(np.datetime64(d.strip().replace(" ","T")))
            except ValueError: continue
        od=np.array(od); ov=np.array(ov)
        oty=(od.astype("datetime64[D]")-t0).astype("timedelta64[D]").astype(float)/365.25
        ohalf=np.array([np.mean(ov[(oty>=tb[i])&(oty<tb[i+1])]) if ((oty>=tb[i])&(oty<tb[i+1])).sum()>0 else np.nan
                        for i in range(len(tb)-1)])
        # align to kt bins
        bin_idx=[int(np.searchsorted(tb,t)-1) for t in kt]
        oal=np.array([ohalf[b] for b in bin_idx])
        good=np.isfinite(oal)
        if good.sum()>3:
            b1,b0=np.polyfit(oal[good],ky[good],1)
            kO=np.full_like(ky,np.nan); kO[good]=b0+b1*oal[good]
            fits["OULU-driven (2p)"]=kO

    # plot
    fig,ax=plt.subplots(figsize=(12,7))
    ax.plot(yr,ky,"ko-",lw=2,ms=6,label="half-year LS k (data)",zorder=5)
    styles=["--","-.",":","-"]
    print("\n=== fit comparison ===")
    for (name,kf),st in zip(fits.items(),styles):
        m=np.isfinite(kf)
        rms=np.sqrt(np.nanmean((ky-kf)**2))
        r2026=ky[-1]-kf[-1] if np.isfinite(kf[-1]) else np.nan
        ax.plot(yr[m],kf[m],st,lw=2,label=f"{name}: RMS={rms*1e5:.1f}e-5, 2026Δ={r2026*1e5:+.1f}e-5")
        print(f"  {name}: RMS={rms:.6f}, 2026 residual={r2026:+.6f}")
    ax.set_xlabel("year"); ax.set_ylabel("k(t)")
    ax.set_title("k(t) functional form comparison (data = half-year LS slope)",fontsize=13,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/diag_kt_form.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    print("Saved plots/diag_kt_form.png")


if __name__=="__main__":
    main()
