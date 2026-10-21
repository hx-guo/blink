#!/usr/bin/env python3
"""g(t) form: is exponential justified by the data, or is it indistinguishable
from linear over the 8.9-yr window? Extract equatorial residual (|mlat|<5, the
mlat-term vanishes), fit linear / exp / quadratic, compare RMS + curvature."""
import glob, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

L = 16e-6; NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"

def unwrap(pho,large,wide,sci,lc,dt,C):
    pho=np.asarray(pho,float);large=np.asarray(large,float);wide=np.asarray(wide,float);sci=np.asarray(sci,float)
    LL=np.asarray(lc,float)*L; lf=1.0-np.asarray(dt,float)/np.asarray(lc,float)
    pred=pho-(wide+(sci+C)*LL)/lf; n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0); out=large+np.where(ov,nm,n)*1024.
    return out

grid=np.load("n_below_study/aacgm_grid_2020.npz")
interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],bounds_error=False,fill_value=np.nan)
t0=np.datetime64("2017-06-22")
files=[f for f in sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet"))) if "sample" not in f]
RES=[];TY=[]
for f in files:
    pf=pq.ParquetFile(f)
    for rg in np.unique(np.linspace(0,pf.num_row_groups-1,6).astype(int)):
        df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
        am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values]))); am=np.where(np.isnan(am),0.,am)
        eqm=am<5
        if eqm.sum()==0: continue
        d=df[eqm]
        pho=d["PHO"].astype(float).values;lg=d["Large"].astype(float).values;wd=d["Wide"].astype(float).values
        sc=d["Sci_1s"].astype(float).values;lc=d["L_cycles"].astype(float).values;dt=d["Dt"].astype(float).values
        lf=1.0-dt/lc; lv=unwrap(pho,lg,wd,sc,lc,dt,150.0); res=(pho-lv)*lf/(lc*L)-wd/(lc*L)-sc
        ok=(wd/np.maximum(pho,1)<0.3)&(sc>100)&np.isfinite(res)&(np.abs(res)<2000)
        ty=(pd.to_datetime(d["date"]).values.astype("datetime64[D]")-t0).astype("timedelta64[D]").astype(float)/365.25
        RES.append(res[ok]);TY.append(ty[ok])
    print(f"  scanned {os.path.basename(f)}",flush=True)
res=np.concatenate(RES);ty=np.concatenate(TY)

# monthly median g(t)
tb=np.linspace(0,9,55); tc=0.5*(tb[:-1]+tb[1:]); gm=[]
for i in range(len(tc)):
    m=(ty>=tb[i])&(ty<tb[i+1])
    gm.append(np.median(res[m]) if m.sum()>200 else np.nan)
gm=np.array(gm); good=np.isfinite(gm); x=tc[good]; y=gm[good]/np.nanmedian(gm[good][:4])

# fits
lin=lambda t,b: 1-b*t
expf=lambda t,A,tau: A+(1-A)*np.exp(-t/tau)
quad=lambda t,b,c: 1+b*t+c*t**2
pl,_=curve_fit(lin,x,y,p0=[0.03])
try: pe,_=curve_fit(expf,x,y,p0=[0.5,8],bounds=([0,1],[0.95,60]),maxfev=20000)
except: pe=[np.nan,np.nan]
pq2,_=curve_fit(quad,x,y,p0=[-0.03,0])
rms_l=np.std(y-lin(x,*pl)); rms_e=np.std(y-expf(x,*pe)); rms_q=np.std(y-quad(x,*pq2))
# curvature significance: refit quad, get c and its error
from numpy.polynomial import polynomial as P
co,cov=np.polyfit(x,y,2,cov=True); c2=co[0]; c2err=np.sqrt(cov[0,0])

print(f"\n=== g(t) form comparison (N={len(y)} monthly points) ===")
print(f"linear  g=1-{pl[0]:.4f}t            RMS={rms_l:.4f}")
print(f"exp     A={pe[0]:.3f} tau={pe[1]:.1f}yr  RMS={rms_e:.4f}")
print(f"quad    c(t^2)={c2:.5f} +- {c2err:.5f}  ({abs(c2/c2err):.1f} sigma)  RMS={rms_q:.4f}")
print(f"-> curvature {'SIGNIFICANT' if abs(c2/c2err)>3 else 'NOT significant (data ~linear)'}")

fig,(a1,a2)=plt.subplots(2,1,figsize=(12,8),height_ratios=[2,1])
a1.plot(2017.48+x,y,"ko",ms=4,label="equatorial g(t) data (monthly)")
xx=np.linspace(0,9,100)
a1.plot(2017.48+xx,lin(xx,*pl),"b-",lw=2,label=f"linear (RMS={rms_l:.4f})")
a1.plot(2017.48+xx,expf(xx,*pe),"r--",lw=2,label=f"exp A={pe[0]:.2f} tau={pe[1]:.0f}yr (RMS={rms_e:.4f})")
a1.set_ylabel("g(t) = residual/residual(t0)"); a1.legend(fontsize=10); a1.grid(alpha=0.3)
a1.set_title(f"g(t) form: exp vs linear  (curvature {abs(c2/c2err):.1f} sigma)",fontsize=12)
a2.plot(2017.48+x,y-lin(x,*pl),"bo-",ms=3,label="linear residual")
a2.plot(2017.48+x,y-expf(x,*pe),"r^-",ms=3,label="exp residual")
a2.axhline(0,color="k",lw=0.8); a2.set_xlabel("year"); a2.set_ylabel("fit residual")
a2.legend(fontsize=9); a2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("plots/diag_gt_form.png",dpi=120,bbox_inches="tight"); plt.close(fig)
print("Saved plots/diag_gt_form.png")
