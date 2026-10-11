#!/usr/bin/env python3
"""v5t vs v6_acd: NO clean filter (no wide/PHO<0.3, no sci>100, no |resid|<2000).

Only pure sanity bounds (isfinite, sci>0, base>0). All bright-source seconds,
low-statistics seconds, and outliers contribute to the reported metrics —
unfiltered honest comparison.

For v6_acd fit, |C_truth|<500 is kept as a numerical sanity bound for polyfit
(polyfit otherwise dragged by hard outliers); but the FINAL apply pass is fully
unfiltered.

Output: plots/compare_v5_vs_v6_v3.png
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

    # PASS 1: collect C_truth for v6_acd fit
    # Only numerical sanity: |C_truth|<500 (polyfit robustness against hard outliers)
    print("=== PASS 1: fit v6_acd (only numerical sanity, NO physics filter) ===")
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
            # Numerical-only sanity (polyfit needs finite + bounded)
            ok=np.isfinite(C_truth)&(np.abs(C_truth)<500)
            fit_acd.append(acd[ok]); fit_C.append(C_truth[ok]); fit_det.append(detid[ok])
        print(f"  {os.path.basename(f)}: scanned")

    acd_all=np.concatenate(fit_acd); C_all=np.concatenate(fit_C); det_all=np.concatenate(fit_det)
    a_det=np.zeros(18); b_det=np.zeros(18)
    for d in range(18):
        m=(det_all==d)
        if m.sum()<1000: continue
        coef=np.polyfit(acd_all[m],C_all[m],1); b_det[d],a_det[d]=coef
    print(f"v6_acd a_det median {np.median(a_det):+.1f}, b_det median {np.median(b_det):+.4e}")

    # PASS 2: apply both models, ONLY sanity (isfinite, sci>0, base>0)
    print("\n=== PASS 2: apply v5t and v6_acd, NO clean filter ===")
    r5_all=[]; sci5_all=[]; dates5_all=[]; r6_all=[]; sci6_all=[]; dates6_all=[]
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

            # ONLY sanity bounds (no physics filter)
            ok5=np.isfinite(r5)&np.isfinite(base5)&(sci>0)&(base5>0)
            ok6=np.isfinite(r6)&np.isfinite(base6)&(sci>0)&(base6>0)
            n_total5+=ok5.sum(); n_below5+=(base5[ok5]<sci[ok5]).sum()
            n_total6+=ok6.sum(); n_below6+=(base6[ok6]<sci[ok6]).sum()
            dates_arr=df["date"].values
            r5_all.append(r5[ok5]); sci5_all.append(sci[ok5]); dates5_all.append(dates_arr[ok5])
            r6_all.append(r6[ok6]); sci6_all.append(sci[ok6]); dates6_all.append(dates_arr[ok6])
        print(f"  {os.path.basename(f)}: applied",flush=True)

    r5=np.concatenate(r5_all); sci5=np.concatenate(sci5_all); dates5=np.concatenate(dates5_all)
    r6=np.concatenate(r6_all); sci6=np.concatenate(sci6_all); dates6=np.concatenate(dates6_all)

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

    # ─── Cluster diagnostic (the violet circle in the v5t/v6 plots) ───
    # Cluster: Sci_obs in [800, 1800], Sci_rec in [400, 900]
    sci_rec5=r5+sci5; sci_rec6=r6+sci6
    cluster5=(sci5>=800)&(sci5<=1800)&(sci_rec5>=400)&(sci_rec5<=900)
    cluster6=(sci6>=800)&(sci6<=1800)&(sci_rec6>=400)&(sci_rec6<=900)
    print(f"\n=== CLUSTER (sci_obs 800-1800, sci_rec 400-900) ===")
    print(f"  v5t in cluster: {cluster5.sum():,} / {len(r5):,} = {cluster5.mean()*100:.3f}%")
    print(f"  v6 in cluster:  {cluster6.sum():,} / {len(r6):,} = {cluster6.mean()*100:.3f}%")

    if cluster5.sum() > 0:
        cl_dates=dates5[cluster5]
        from collections import Counter
        # By year-month for trend
        ym=Counter(d[:7] for d in cl_dates)
        print(f"\n  Top 15 year-months (v5t cluster):")
        for k,v in sorted(ym.items(), key=lambda x:-x[1])[:15]:
            print(f"    {k}: {v:,} rows  ({v/cluster5.sum()*100:.1f}%)")
        # By exact date for spike days
        dc=Counter(cl_dates)
        print(f"\n  Top 10 days (v5t cluster):")
        for k,v in sorted(dc.items(), key=lambda x:-x[1])[:10]:
            print(f"    {k}: {v:,} rows")

    # ─── Density-colored plot, full formula in titles ───
    mpl.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.18, 1.0, 0.55], hspace=0.18, wspace=0.18)

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

    # Helper: density-colored scatter using 2D histogram lookup
    def density_scatter(ax, x, y, *, bins=180, xlog=True, ylog=False,
                        cmap='viridis', s=2, max_pts=600000):
        """Subsample + per-point density from 2D log-hist lookup; scatter points."""
        if len(x) > max_pts:
            idx = np.random.RandomState(0).choice(len(x), max_pts, replace=False)
            x = x[idx]; y = y[idx]
        # density grid in log/linear space
        xx = np.log10(x) if xlog else x
        yy = np.log10(y) if ylog else y
        H, xe, ye = np.histogram2d(xx, yy, bins=bins)
        # Per-point density via bin lookup
        ix = np.clip(np.searchsorted(xe, xx)-1, 0, H.shape[0]-1)
        iy = np.clip(np.searchsorted(ye, yy)-1, 0, H.shape[1]-1)
        dens = H[ix, iy]
        # Sort so high-density points draw on top
        order = np.argsort(dens)
        sc = ax.scatter(x[order], y[order], c=dens[order], s=s,
                        cmap=cmap, edgecolors='none',
                        norm=mpl.colors.LogNorm(vmin=max(dens.min(),1), vmax=dens.max()),
                        rasterized=True)
        return sc

    ax_s5 = fig.add_subplot(gs[1, 0])
    ax_s6 = fig.add_subplot(gs[1, 1])
    sci_rec_5 = r5 + sci5
    sci_rec_6 = r6 + sci6
    for ax, sci_obs, sci_rec, m, label in [
        (ax_s5, sci5, sci_rec_5, m5, "v5t"),
        (ax_s6, sci6, sci_rec_6, m6, "v6_acd"),
    ]:
        valid = (sci_obs>=10) & (sci_rec>0)
        sc = density_scatter(ax, sci_obs[valid], sci_rec[valid], xlog=True, ylog=True)
        xs = np.logspace(0, 4, 50)
        ax.plot(xs, xs, 'r-', lw=2, label="y = x (perfect recovery)")
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(10, 10000); ax.set_ylim(1, 10000)
        ax.set_xlabel(r"$\rm Sci_{obs}$ observed (cnt/s)", fontsize=12)
        ax.set_ylabel(r"$\rm Sci_{rec}$ recovered (cnt/s)", fontsize=12)
        ax.set_title(f"{label}:  blob/main = {m['blob_main']:.2f}%   "
                     f"|   below y=x = {m['below']:.3f}%   "
                     f"|   median = {m['median']:+.1f},  MAD = {m['MAD']:.1f}",
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, which='both')
        ax.legend(loc='lower right')
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("local density (counts per bin)")

    ax_r5 = fig.add_subplot(gs[2, 0])
    ax_r6 = fig.add_subplot(gs[2, 1])
    for ax, sci_obs, resid, m, label in [
        (ax_r5, sci5, r5, m5, "v5t"),
        (ax_r6, sci6, r6, m6, "v6_acd"),
    ]:
        m_in = (sci_obs>=10)&(sci_obs<=10000)&(resid>=-400)&(resid<=400)
        sc = density_scatter(ax, sci_obs[m_in], resid[m_in], xlog=True, ylog=False)
        ax.axhline(0, color='r', lw=1.5)
        ax.axhspan(-300, -50, alpha=0.10, color='red', label='blob zone')
        ax.set_xscale('log'); ax.set_xlim(10, 10000); ax.set_ylim(-400, 400)
        ax.set_xlabel(r"$\rm Sci_{obs}$ observed (cnt/s)", fontsize=12)
        ax.set_ylabel(r"residual $\rm Sci_{rec}-Sci_{obs}$ (cnt/s)", fontsize=12)
        ax.set_title(f"{label}: residual", fontsize=12)
        ax.grid(alpha=0.3, which='both'); ax.legend(loc='upper left')
        fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02).set_label("local density (counts per bin)")

    fig.suptitle(f"v5t vs v6_acd PHO-conservation reconstruction  |  FULL DATA, NO clean filter  |  N≈{m5['N']/1e6:.1f}M rows",
                 fontsize=14, fontweight='bold', y=0.995)
    Path("plots").mkdir(exist_ok=True)
    out="plots/compare_v5_vs_v6_v5.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
