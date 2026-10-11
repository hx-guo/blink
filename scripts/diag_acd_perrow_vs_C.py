#!/usr/bin/env python3
"""Per-row test: does C(det,mlat,t) from v5t correlate with ACD_sum per row?

If r > 0.95, replace v5t's s0_det·g(t)·[1+k(t)mlat²]+C0 (23 params)
with C = a_det + b_det · ACD_sum (36 params, fully data-driven).

Test:
  1. Sample cache, compute v5t-predicted C per row.
  2. Compute observed residual: resid_obs = base_v5(C=0) - sci
     (this is "what C should be at this row" given conservation).
  3. Scatter resid_obs vs ACD_sum, per det.
  4. Linear fit per det: C_pred_acd = a_det + b_det · ACD_sum.
  5. Compare residual MAD: v5t formula vs ACD-driven, per det.
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
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon","ACD_sum","PM_0","PM_1","PM_2"]
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
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    print(f"v5t: beta={beta:.4f}, C0={C0:+.2f}, w={w:.4f}")

    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # Collect per-row arrays
    all_C_v5=[]; all_resid_obs=[]; all_acd=[]; all_pm0=[]; all_detid=[]; all_year=[]; all_mt=[]
    for f in files:
        year=int(os.path.basename(f).split("_")[2].split(".")[0])
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        # Sample fewer row groups for speed
        for rg in np.unique(np.linspace(0,n_rg-1,3).astype(int)):
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
            C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0

            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
            wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            acd=df["ACD_sum"].astype(float).values
            pm0=df["PM_0"].astype(float).values
            LL=lc*L; lf=1.0-dtv/lc

            # base assuming C=0 → resid_obs is "what C should be"
            lv2=unwrap(pho,lg,wd,sci,lc,dtv,C_v5)  # use v5t C for unwrap
            base=(pho-lv2)*lf/LL-wd/LL
            resid_obs=base-sci  # observed C-like quantity per row
            clean=(wd/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(resid_obs)&(np.abs(resid_obs)<500)
            # Exclude SAA (high PM)
            clean=clean&(pm0<5)

            all_C_v5.append(C_v5[clean]); all_resid_obs.append(resid_obs[clean])
            all_acd.append(acd[clean]); all_pm0.append(pm0[clean])
            all_detid.append(detid[clean]); all_year.append(np.full(clean.sum(),year))
            all_mt.append(mt[clean])
        print(f"  {os.path.basename(f)}: scanned")

    C_v5=np.concatenate(all_C_v5); resid_obs=np.concatenate(all_resid_obs)
    acd=np.concatenate(all_acd); pm0=np.concatenate(all_pm0)
    detid=np.concatenate(all_detid); year=np.concatenate(all_year); mt=np.concatenate(all_mt)
    print(f"\nTotal clean rows: {len(C_v5):,}")

    # Per-det linear fit: resid_obs = a_det + b_det * ACD_sum
    print("\n=== per-det linear fit: C_obs = a + b*ACD_sum ===")
    print("  det   a      b         r       n        v5t_RMS   acd_RMS")
    a_det=np.zeros(18); b_det=np.zeros(18); r_det=np.zeros(18)
    rms_v5_det=np.zeros(18); rms_acd_det=np.zeros(18)
    for d in range(18):
        m=(detid==d)
        if m.sum()<1000: continue
        y=resid_obs[m]; x=acd[m]
        coef=np.polyfit(x,y,1); b_det[d],a_det[d]=coef
        r_det[d],_=pearsonr(x,y)
        # v5t residual: actual - v5t predicted
        v5_pred=C_v5[m]
        rms_v5_det[d]=np.sqrt(np.mean((y-v5_pred)**2))
        # ACD residual: actual - acd predicted
        acd_pred=a_det[d]+b_det[d]*x
        rms_acd_det[d]=np.sqrt(np.mean((y-acd_pred)**2))
        box='ABC'[d//6]; di=d%6
        print(f"  {box}{di}  {a_det[d]:+6.2f} {b_det[d]:+.4e}  {r_det[d]:.3f}  {m.sum():,}  {rms_v5_det[d]:5.1f}    {rms_acd_det[d]:5.1f}")

    # Overall r
    r_all,_=pearsonr(acd,resid_obs)
    print(f"\nOverall r(ACD_sum, C_obs) = {r_all:+.4f}")

    rms_all_v5=np.sqrt(np.mean((resid_obs-C_v5)**2))
    print(f"v5t RMS:  {rms_all_v5:.2f} (per-row C residual)")
    # ACD per-det:
    pred_acd=a_det[detid]+b_det[detid]*acd
    rms_all_acd=np.sqrt(np.mean((resid_obs-pred_acd)**2))
    print(f"ACD RMS:  {rms_all_acd:.2f}  ({'BETTER' if rms_all_acd<rms_all_v5 else 'WORSE'})")

    # Plot: sample of resid_obs vs ACD_sum colored by det
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    rs=np.random.RandomState(0); ix=rs.choice(len(C_v5),min(80000,len(C_v5)),replace=False)
    ax=axes[0,0]
    sc=ax.scatter(acd[ix],resid_obs[ix],c=detid[ix],cmap='tab20',s=2,alpha=0.4,rasterized=True)
    xs=np.linspace(0,acd.max(),100)
    # overlay per-det fits
    for d in range(18):
        if r_det[d]==0: continue
        ax.plot(xs,a_det[d]+b_det[d]*xs,'-',color=plt.cm.tab20(d/18),alpha=0.5,lw=0.7)
    ax.set_xlabel("ACD_sum (per row)"); ax.set_ylabel("C_obs = base(C=v5t)-sci (per row)")
    ax.set_title(f"per-row: r_overall={r_all:.3f}")
    ax.grid(alpha=0.3); fig.colorbar(sc,ax=ax).set_label("det 0-17")

    ax=axes[0,1]
    ax.scatter(C_v5[ix],resid_obs[ix],c=acd[ix],cmap='viridis',s=2,alpha=0.4,rasterized=True,vmin=0,vmax=30000)
    ax.plot([resid_obs.min(),resid_obs.max()],[resid_obs.min(),resid_obs.max()],'r--')
    ax.set_xlabel("C_v5t prediction"); ax.set_ylabel("C_obs")
    rms_v5_all=np.sqrt(np.mean((resid_obs-C_v5)**2))
    ax.set_title(f"v5t vs observed: RMS={rms_v5_all:.1f}")
    ax.grid(alpha=0.3)

    ax=axes[1,0]
    pred_acd_all=a_det[detid]+b_det[detid]*acd
    ax.scatter(pred_acd_all[ix],resid_obs[ix],c=acd[ix],cmap='viridis',s=2,alpha=0.4,rasterized=True,vmin=0,vmax=30000)
    ax.plot([resid_obs.min(),resid_obs.max()],[resid_obs.min(),resid_obs.max()],'r--')
    ax.set_xlabel("C_acd = a_det + b_det*ACD"); ax.set_ylabel("C_obs")
    rms_acd_all=np.sqrt(np.mean((resid_obs-pred_acd_all)**2))
    ax.set_title(f"ACD-driven vs observed: RMS={rms_acd_all:.1f}")
    ax.grid(alpha=0.3)

    # Compare per-det RMS
    ax=axes[1,1]
    ax.bar(np.arange(18)-0.2,rms_v5_det,0.4,label="v5t",color='r',alpha=0.7)
    ax.bar(np.arange(18)+0.2,rms_acd_det,0.4,label="ACD",color='b',alpha=0.7)
    ax.set_xticks(range(18)); ax.set_xticklabels([f"{'ABC'[d//6]}{d%6}" for d in range(18)],rotation=45,fontsize=8)
    ax.set_ylabel("RMS C residual"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("per-det RMS: v5t vs ACD-driven")

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out="plots/diag_acd_perrow_vs_C.png"
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
