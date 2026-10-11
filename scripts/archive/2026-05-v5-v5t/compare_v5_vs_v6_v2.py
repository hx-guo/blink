#!/usr/bin/env python3
"""v5t vs v6_acd PHO-conservation comparison — density-colored, full formula.

Same clean filter as plot_v5t_conservation.py: wide/PHO<0.3 & sci>100 & |resid|<2000
so the v5t panel matches the formal v5t conservation plot visually.

Output: plots/compare_v5_vs_v6_v2.png
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    LL=lc*L; lf=1.0-dtv/lc
    lv3=unwrap_v2(pho,lg,wd,sci,lc,dtv,C)
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
    n3=np.round((lv3-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
    lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
    base=(pho-lv5)*lf/LL-wd/LL
    return base


def main():
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # PASS 1: collect for v6 fit (no PM filter)
    print("=== PASS 1 ===")
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
    a_det=np.zeros(18); b_det=np.zeros(18)
    for d in range(18):
        m=(det_all==d)
        if m.sum()<1000: continue
        coef=np.polyfit(acd_all[m],C_all[m],1); b_det[d],a_det[d]=coef
    print(f"v6_acd a_det median {np.median(a_det):+.1f}, b_det median {np.median(b_det):+.4e}")

    # PASS 2: apply both, use same clean filter as plot_v5t_conservation
    print("\n=== PASS 2 ===")
    r5_all=[]; sci5_all=[]; r6_all=[]; sci6_all=[]
    n_total5=0; n_below5=0; n_total6=0; n_below6=0
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
            base5=apply_pipeline(pho,lg,wd,sci,lc,dtv,C_v5)
            base6=apply_pipeline(pho,lg,wd,sci,lc,dtv,C_acd)
            r5=base5-sci-C_v5
            r6=base6-sci-C_acd

            # Same clean filter as plot_v5t_conservation
            clean5=(wd/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(r5)&(np.abs(r5)<2000)&np.isfinite(base5)&(base5>0)
            clean6=(wd/np.maximum(pho,1)<0.3)&(sci>100)&np.isfinite(r6)&(np.abs(r6)<2000)&np.isfinite(base6)&(base6>0)
            n_total5+=clean5.sum(); n_below5+=(base5[clean5]<sci[clean5]).sum()
            n_total6+=clean6.sum(); n_below6+=(base6[clean6]<sci[clean6]).sum()
            r5_all.append(r5[clean5]); sci5_all.append(sci[clean5])
            r6_all.append(r6[clean6]); sci6_all.append(sci[clean6])
        print(f"  {os.path.basename(f)}: applied",flush=True)

    r5=np.concatenate(r5_all); sci5=np.concatenate(sci5_all)
    r6=np.concatenate(r6_all); sci6=np.concatenate(sci6_all)

    def stats(resid, sci_obs, n_below, N):
        med=np.median(resid)
        mad=np.median(np.abs(resid-med))
        blob=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-300)&(resid<=-50)).sum()
        main=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-50)&(resid<=100)).sum()
        return dict(median=med, MAD=mad,
                    blob_main=blob/max(main,1)*100,
                    below=n_below/N*100, N=N)
    m5=stats(r5,sci5,n_below5,n_total5)
    m6=stats(r6,sci6,n_below6,n_total6)

    print(f"\n             v5t       v6_acd")
    print(f"  N         {m5['N']:>9,}  {m6['N']:>9,}")
    print(f"  median    {m5['median']:+8.2f}  {m6['median']:+8.2f}")
    print(f"  MAD       {m5['MAD']:>8.2f}  {m6['MAD']:>8.2f}")
    print(f"  blob/main {m5['blob_main']:>7.2f}%  {m6['blob_main']:>7.2f}%")
    print(f"  below-y=x {m5['below']:>7.4f}%  {m6['below']:>7.4f}%")

    # ─── Density-colored plot, full formula in titles ───
    mpl.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.18, 1.0, 0.55], hspace=0.18, wspace=0.18)

    # Top row: formula text panels
    ax_t5 = fig.add_subplot(gs[0, 0]); ax_t5.axis('off')
    ax_t6 = fig.add_subplot(gs[0, 1]); ax_t6.axis('off')

    formula_v5 = (
        r"$\bf{v5t\ (23\ params)}$" + "\n\n"
        r"$\rm Sci_{rec} = \dfrac{1}{L_{cyc}\cdot 16\mu s}\left[(PHO-Large)\dfrac{L_{cyc}-Dt}{L_{cyc}} - Wide\right] - C$" + "\n\n"
        r"$C(\rm det, |mlat|, t) = s_{0,det}\, g(t)\,[1 + k(t)\max(0,|mlat|-20^{\circ})^2] + C_0$" + "\n\n"
        r"$g(t) = 1-\beta t$ (PMT outgassing),  $k(t) = c_0+a_1\cos\omega t+b_1\sin\omega t$,  $\omega=2\pi/11\rm\,yr$"
    )
    formula_v6 = (
        r"$\bf{v6\_acd\ (36\ params)}$" + "\n\n"
        r"$\rm Sci_{rec} = \dfrac{1}{L_{cyc}\cdot 16\mu s}\left[(PHO-Large)\dfrac{L_{cyc}-Dt}{L_{cyc}} - Wide\right] - C$" + "\n\n"
        r"$C(\rm det,\,row) = a_{det} + b_{det}\cdot ACD\_sum(row)$" + "\n\n"
        r"$\rm ACD\_sum = \sum_{i=0}^{17}Cnt\_VetoDet_i$  (engineering 1\,Hz, APID 0548 ASU board)"
    )
    ax_t5.text(0.02, 0.5, formula_v5, fontsize=12, va='center', ha='left',
               transform=ax_t5.transAxes)
    ax_t6.text(0.02, 0.5, formula_v6, fontsize=12, va='center', ha='left',
               transform=ax_t6.transAxes)

    # Mid row: Sci_rec vs Sci_obs (log-log) with hexbin density coloring
    ax_s5 = fig.add_subplot(gs[1, 0])
    ax_s6 = fig.add_subplot(gs[1, 1])
    sci_rec_5 = r5 + sci5
    sci_rec_6 = r6 + sci6
    for ax, sci_obs, sci_rec, m, label in [
        (ax_s5, sci5, sci_rec_5, m5, "v5t"),
        (ax_s6, sci6, sci_rec_6, m6, "v6_acd"),
    ]:
        valid = (sci_obs>=10) & (sci_rec>0)
        hb = ax.hexbin(sci_obs[valid], sci_rec[valid],
                       xscale='log', yscale='log',
                       gridsize=120, cmap='viridis', mincnt=1,
                       bins='log', extent=(np.log10(10),np.log10(10000),np.log10(1),np.log10(10000)))
        xs = np.logspace(0, 4, 50)
        ax.plot(xs, xs, 'r-', lw=2, label="y = x (perfect recovery)")
        ax.set_xlim(10, 10000); ax.set_ylim(1, 10000)
        ax.set_xlabel(r"$\rm Sci_{obs}$ observed (cnt/s)", fontsize=12)
        ax.set_ylabel(r"$\rm Sci_{rec}$ recovered (cnt/s)", fontsize=12)
        ax.set_title(f"{label}:  blob/main = {m['blob_main']:.2f}%   "
                     f"|   below y=x = {m['below']:.3f}%   "
                     f"|   median = {m['median']:+.1f},  MAD = {m['MAD']:.1f}",
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, which='both')
        ax.legend(loc='lower right')
        cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("log10 density")

    # Bottom row: residual vs Sci_obs
    ax_r5 = fig.add_subplot(gs[2, 0])
    ax_r6 = fig.add_subplot(gs[2, 1])
    for ax, sci_obs, resid, m, label in [
        (ax_r5, sci5, r5, m5, "v5t"),
        (ax_r6, sci6, r6, m6, "v6_acd"),
    ]:
        m_in = (sci_obs>=30)&(sci_obs<=10000)&(resid>=-400)&(resid<=400)
        hb = ax.hexbin(sci_obs[m_in], resid[m_in],
                       xscale='log', gridsize=120, cmap='viridis', mincnt=1, bins='log')
        ax.axhline(0, color='r', lw=1.5)
        ax.axhspan(-300, -50, alpha=0.10, color='red', label='blob zone')
        ax.set_xscale('log'); ax.set_xlim(30, 10000); ax.set_ylim(-400, 400)
        ax.set_xlabel(r"$\rm Sci_{obs}$ observed (cnt/s)", fontsize=12)
        ax.set_ylabel(r"residual $\rm Sci_{rec}-Sci_{obs}$ (cnt/s)", fontsize=12)
        ax.set_title(f"{label}: residual", fontsize=12)
        ax.grid(alpha=0.3, which='both'); ax.legend(loc='upper left')
        fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02).set_label("log10 density")

    fig.suptitle(f"v5t vs v6_acd PHO-conservation reconstruction  |  full mission 2017–2026  |  N≈{m5['N']/1e6:.1f}M clean rows",
                 fontsize=14, fontweight='bold', y=0.995)
    Path("plots").mkdir(exist_ok=True)
    out="plots/compare_v5_vs_v6_v2.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
