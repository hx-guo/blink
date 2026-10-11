#!/usr/bin/env python3
"""Spot check: does <ACD_sum>_year correlate with the v5t per-year k value?

If r > 0.9, ACD is a viable in-situ replacement for the OULU/sinusoid k(t).
If not, keep the single-sinusoid form.

Output:
  - print yearly table of (k_year, acd_year_median, n_rows)
  - print Pearson r
  - save plots/diag_acd_vs_kyear.png (scatter + side-by-side time series)
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

L = 16e-6
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon","ACD_sum"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
B_THRESHOLD = 20.0


def unwrap(pho, large, wide, sci, lc, dtv, C):
    pho=np.asarray(pho,float); large=np.asarray(large,float); wide=np.asarray(wide,float)
    sci=np.asarray(sci,float); LL=np.asarray(lc,float)*L
    lf=1.0-np.asarray(dtv,float)/np.asarray(lc,float)
    pred=pho-(wide+(sci+C)*LL)/lf
    n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0)
        out=large+np.where(ov,nm,n)*1024.
    return out


def main():
    # Load v5t calib to get s0_det, beta, t0
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    print(f"Loaded v5t calib: beta={beta:.4f}, C0={float(cz['C0']):+.2f}")

    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # Accumulate per-year: k (resid/s_fix-1)/mlat^2 high-mlat; ACD median (low-ACD non-SAA)
    k_num={}; k_den={}; acd_low={}
    for f in files:
        year=int(os.path.basename(f).split("_")[2].split(".")[0])
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,6).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
            am=np.where(np.isnan(am),0.0,am)
            mt=np.maximum(0.0,am-B_THRESHOLD)**2

            # per-row t_years
            d_arr=np.array([np.datetime64(d) for d in df["date"].values])
            ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
            g_t=1.0-beta*ty
            box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
            detid=box_idx*6+df["det"].values
            s_fix=s0_det[detid]*g_t

            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
            wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            acd=df["ACD_sum"].astype(float).values
            LL=lc*L; lf=1.0-dtv/lc

            lv2=unwrap(pho,lg,wd,sci,lc,dtv,150.0)
            base=(pho-lv2)*lf/LL-wd/LL; resid=base-sci
            clean=(wd/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(resid)&(np.abs(resid)<2000)
            hi=clean&(am>=35)&(s_fix>1e-3)

            kk=(resid/s_fix-1.0)/np.maximum(mt,1.0)
            k_num[year]=k_num.get(year,0)+np.nansum(kk[hi])
            k_den[year]=k_den.get(year,0)+hi.sum()

            # ACD: only non-SAA rows; we want the "background CR" signal
            # ACD_sum < 25000 excludes the SAA bursts.
            nonsaa=clean&(acd<25000)&(acd>1000)
            if nonsaa.any():
                acd_low.setdefault(year,[]).extend(acd[nonsaa].tolist())
        print(f"  {os.path.basename(f)}: scanned")

    years=np.array(sorted(k_num))
    k_year=np.array([k_num[y]/max(k_den[y],1) for y in years])
    acd_year=np.array([np.median(acd_low[y]) if y in acd_low else np.nan for y in years])

    print("\n=== per-year table ===")
    print(f"  year   k_year      acd_year_med   n_acd")
    for y,kv,av in zip(years,k_year,acd_year):
        n=len(acd_low.get(y,[]))
        print(f"  {y}   {kv:.5f}    {av:8.0f}     {n}")

    mask=np.isfinite(k_year)&np.isfinite(acd_year)
    r,p=pearsonr(k_year[mask],acd_year[mask])
    print(f"\nPearson r(k_year, acd_year) = {r:+.4f}  (p={p:.3g})")

    # Linear fit: k = a + b * acd
    A=np.column_stack([np.ones(mask.sum()),acd_year[mask]])
    coef,*_=np.linalg.lstsq(A,k_year[mask],rcond=None)
    a,b=coef
    print(f"Linear fit: k = {a:+.5f} + {b:+.4e} * acd_year_med")
    res=k_year[mask]-(a+b*acd_year[mask])
    print(f"RMS residual: {np.sqrt(np.mean(res**2)):.5f}  (compared to k_year std={np.std(k_year[mask]):.5f})")

    # Compare to v5t single-sinusoid k(t)
    w=float(cz["w"]); kc=cz["k_coeffs"]
    ty_year=np.array([(np.datetime64(f"{y}-07-01")-t0).astype("timedelta64[D]").astype(float)/365.25 for y in years])
    k_sinu=kc[0]+kc[1]*np.cos(w*ty_year)+kc[2]*np.sin(w*ty_year)
    res_sinu=k_year[mask]-k_sinu[mask]
    print(f"v5t sinusoid RMS residual: {np.sqrt(np.mean(res_sinu**2)):.5f}")

    # Plot
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    ax[0].scatter(acd_year[mask],k_year[mask],s=80,c=years[mask],cmap="turbo")
    xs=np.linspace(acd_year[mask].min(),acd_year[mask].max(),100)
    ax[0].plot(xs,a+b*xs,"k--",label=f"k = {a:+.4f}{b:+.3e}*acd\nr={r:.3f}")
    ax[0].set_xlabel("median ACD_sum, non-SAA (per year)")
    ax[0].set_ylabel("k_year (v5t high-mlat residual estimator)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"k vs ACD per year — r={r:+.3f}")

    ax[1].plot(years[mask],k_year[mask],"ko-",label="k_year (data)")
    ax[1].plot(years,k_sinu,"r-",alpha=0.6,label="v5t sinusoid")
    ax[1].plot(years[mask],a+b*acd_year[mask],"b-",alpha=0.6,label=f"linear in ACD")
    ax[1].set_xlabel("Year"); ax[1].set_ylabel("k")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("k(t): data vs sinusoid vs ACD-driven")
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/diag_acd_vs_kyear.png",dpi=120,bbox_inches="tight")
    print(f"Saved plots/diag_acd_vs_kyear.png")


if __name__=="__main__":
    main()
