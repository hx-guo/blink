#!/usr/bin/env python3
"""5-model comparison with each model OPTIMIZED:
  1. Median-correction offset (residual median → 0)
  2. Higher polynomial degree for B and C
  3. Re-fit D with better initial guess + bounds

Output: classic metrics + LaTeX-annotated 5-panel plot.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.stats import skew

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


def classic_stats(resid, n_below, N):
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    std = float(np.std(resid))
    rms = float(np.sqrt(np.mean(resid**2)))
    p05 = float(np.percentile(resid, 5))
    p95 = float(np.percentile(resid, 95))
    sk  = float(skew(resid))
    return dict(median=med, MAD=mad, std=std, RMS=rms,
                P05=p05, P95=p95, skewness=sk,
                below=n_below/max(N,1)*100, N=N)


def main():
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det_v5=cz["s0_det"]; beta_v5=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w_v5=float(cz["w"]); kc_v5=cz["k_coeffs"]; C0_v5=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # ─── PASS 1: collect training data ───
    print("=== PASS 1: collect ===", flush=True)
    acd_all=[]; C_all=[]; det_all=[]
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
            ok=np.isfinite(C_truth)&(np.abs(C_truth)<500)
            acd_all.append(acd[ok]); C_all.append(C_truth[ok]); det_all.append(detid[ok])
        print(f"  {os.path.basename(f)}", flush=True)
    acd=np.concatenate(acd_all); Cv=np.concatenate(C_all); detv=np.concatenate(det_all)
    print(f"  N={len(acd):,}", flush=True)

    # sub-sample for nonlinear fits
    if len(acd) > 2_000_000:
        idx = np.random.RandomState(42).choice(len(acd), 2_000_000, replace=False)
        acd_fit = acd[idx]; Cv_fit = Cv[idx]; detv_fit = detv[idx]
    else:
        acd_fit = acd; Cv_fit = Cv; detv_fit = detv

    # ─── Fit each model (now with higher degree where applicable) ───
    print("\n=== fitting ===", flush=True)

    # A: linear, 36 params
    A_a=np.zeros(18); A_b=np.zeros(18)
    for d in range(18):
        m=(detv==d)
        if m.sum()<1000: continue
        coef=np.polyfit(acd[m],Cv[m],1); A_b[d],A_a[d]=coef
    print(f"A linear 36 params")

    # B: cubic per-det, 72 params (boost from quadratic)
    B_a=np.zeros(18); B_b=np.zeros(18); B_c=np.zeros(18); B_d=np.zeros(18)
    for d in range(18):
        m=(detv==d)
        if m.sum()<1000: continue
        coef=np.polyfit(acd[m],Cv[m],3)  # [d_cube, c_sq, b_lin, a_const]
        B_d[d],B_c[d],B_b[d],B_a[d]=coef
    print(f"B cubic per-det 72 params")

    # C: shared 4th-order shape + per-det amplitude, 22 params
    def model_C(params, acd_arr, det_arr):
        s = params[:18]
        α = params[18]; β = params[19]; γ = params[20]; off = params[21]
        x = acd_arr / 1e4
        return off + s[det_arr] * (x + α*x**2 + β*x**3 + γ*x**4)
    def resid_C(params):
        return Cv_fit - model_C(params, acd_fit, detv_fit)
    p0_C = np.concatenate([A_b*1e4, [0.0, 0.0, 0.0, 100.0]])
    res = least_squares(resid_C, p0_C, method='trf', max_nfev=100)
    s_C = res.x[:18]; α_C=res.x[18]; β_C=res.x[19]; γ_C=res.x[20]; off_C=res.x[21]
    print(f"C 4th-order shared shape 22 params: α={α_C:.4f}, β={β_C:.5f}, γ={γ_C:.6f}, off={off_C:.1f}")

    # D: v5t-form with better init + bounds
    def model_D(params, acd_arr, det_arr):
        s = params[:18]; gamma = params[18]; thr = params[19]
        return s[det_arr] * (1.0 + gamma * np.maximum(0.0, acd_arr-thr)**2)
    def resid_D(params):
        return Cv_fit - model_D(params, acd_fit, detv_fit)
    # init: typical s_det ~ 100, gamma small positive, thr ~ 20000 (similar to v5t mlat threshold scale)
    p0_D = np.concatenate([np.full(18, 100.0), [1e-9, 20000.0]])
    # bounds: s positive, gamma positive, thr in reasonable range
    lb = np.concatenate([np.full(18, 1.0), [1e-12, 5000.0]])
    ub = np.concatenate([np.full(18, 500.0), [1e-6, 80000.0]])
    res = least_squares(resid_D, p0_D, method='trf', bounds=(lb, ub), max_nfev=200)
    s_D = res.x[:18]; γ_D=res.x[18]; thr_D=res.x[19]
    print(f"D v5t-form bounded 20 params: γ={γ_D:.4e}, thr={thr_D:.0f}")

    # ─── PASS 2: apply, collect residuals ───
    print("\n=== PASS 2: apply ===", flush=True)
    R = {k: [] for k in ['v5t','A','B','C','D']}
    S = {k: [] for k in ['v5t','A','B','C','D']}
    nb = {k: 0 for k in ['v5t','A','B','C','D']}
    nt = {k: 0 for k in ['v5t','A','B','C','D']}
    for f in files:
        pf=pq.ParquetFile(f); n_rg=pf.num_row_groups
        for rg in np.unique(np.linspace(0,n_rg-1,4).astype(int)):
            df=pf.read_row_group(int(rg),columns=NEEDED).to_pandas()
            am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
            am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-B_THRESHOLD)**2
            d_arr=np.array([np.datetime64(d) for d in df["date"].values])
            ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
            g_t=1.0-beta_v5*ty
            k_t=kc_v5[0]+kc_v5[1]*np.cos(w_v5*ty)+kc_v5[2]*np.sin(w_v5*ty)
            box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
            detid=box_idx*6+df["det"].values
            pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
            wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
            lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
            acdv=df["ACD_sum"].astype(float).values

            C_v5=s0_det_v5[detid]*g_t*(1.0+k_t*mt)+C0_v5
            C_A=A_a[detid]+A_b[detid]*acdv
            C_B=B_a[detid]+B_b[detid]*acdv+B_c[detid]*acdv**2+B_d[detid]*acdv**3
            x=acdv/1e4
            C_C=off_C + s_C[detid]*(x + α_C*x**2 + β_C*x**3 + γ_C*x**4)
            C_D=s_D[detid]*(1.0 + γ_D*np.maximum(0.0, acdv-thr_D)**2)

            for tag, C in [('v5t',C_v5),('A',C_A),('B',C_B),('C',C_C),('D',C_D)]:
                base = apply_pipeline(pho,lg,wd,sci,lc,dtv,C)
                resid = base - sci - C
                ok=np.isfinite(resid)&np.isfinite(base)&(sci>0)&(base>0)
                R[tag].append(resid[ok]); S[tag].append(sci[ok])
                nt[tag]+=ok.sum(); nb[tag]+=(base[ok]<sci[ok]).sum()
        print(f"  {os.path.basename(f)}", flush=True)

    # ─── Stage A: raw metrics ───
    print("\n=== RAW (no offset correction) ===", flush=True)
    Mr = {}
    for tag in ['v5t','A','B','C','D']:
        r=np.concatenate(R[tag]); Mr[tag]=classic_stats(r, nb[tag], nt[tag])
    hdr=f"{'model':>5}  {'median':>7}  {'MAD':>5}  {'std':>6}  {'RMS':>6}  {'P05':>7}  {'P95':>7}  {'skew':>5}  {'below%':>7}"
    print(hdr)
    for tag in ['v5t','A','B','C','D']:
        m=Mr[tag]
        print(f"  {tag:>3}  {m['median']:+6.2f}  {m['MAD']:>5.1f}  {m['std']:>6.1f}  {m['RMS']:>6.1f}  {m['P05']:+6.1f}  {m['P95']:+6.1f}  {m['skewness']:+5.2f}  {m['below']:>6.4f}%")

    # ─── Stage B: median-corrected (offset added so median(resid)=0) ───
    print("\n=== OFFSET-CORRECTED (median → 0) ===", flush=True)
    M = {}
    offsets = {}
    for tag in ['v5t','A','B','C','D']:
        r=np.concatenate(R[tag])
        offset = float(np.median(r))
        offsets[tag] = offset
        r_corr = r - offset
        M[tag]=classic_stats(r_corr, nb[tag], nt[tag])
    print(hdr)
    for tag in ['v5t','A','B','C','D']:
        m=M[tag]
        print(f"  {tag:>3}  {m['median']:+6.2f}  {m['MAD']:>5.1f}  {m['std']:>6.1f}  {m['RMS']:>6.1f}  {m['P05']:+6.1f}  {m['P95']:+6.1f}  {m['skewness']:+5.2f}  {m['below']:>6.4f}%  (offset={offsets[tag]:+.2f})")

    # Save fits + offsets
    np.savez("n_below_study/v5_npz/v6_optimal_fits.npz",
             A_a=A_a, A_b=A_b,
             B_a=B_a, B_b=B_b, B_c=B_c, B_d=B_d,
             C_s=s_C, C_α=α_C, C_β=β_C, C_γ=γ_C, C_off=off_C,
             D_s=s_D, D_γ=γ_D, D_thr=thr_D,
             offsets_v5t=offsets['v5t'], offsets_A=offsets['A'],
             offsets_B=offsets['B'], offsets_C=offsets['C'], offsets_D=offsets['D'])

    # ─── Plot 5 models with offset-corrected metrics ───
    mpl.rcParams.update({"font.family":"DejaVu Sans","text.usetex":False})
    fig = plt.figure(figsize=(22, 28))
    gs = fig.add_gridspec(5, 2, width_ratios=[1.0, 0.8],
                          height_ratios=[1]*5, hspace=0.32, wspace=0.20)

    def density_scatter(ax, x, y, *, bins=180, xlog=True, ylog=False,
                        cmap='viridis', s=2.5, max_pts=400000):
        if len(x) > max_pts:
            idx = np.random.RandomState(0).choice(len(x), max_pts, replace=False)
            x = x[idx]; y = y[idx]
        xx = np.log10(x) if xlog else x
        yy = np.log10(y) if ylog else y
        H, xe, ye = np.histogram2d(xx, yy, bins=bins)
        ix = np.clip(np.searchsorted(xe, xx)-1, 0, H.shape[0]-1)
        iy = np.clip(np.searchsorted(ye, yy)-1, 0, H.shape[1]-1)
        dens = H[ix, iy]
        order = np.argsort(dens)
        return ax.scatter(x[order], y[order], c=dens[order], s=s,
                          cmap=cmap, edgecolors='none',
                          norm=mpl.colors.LogNorm(vmin=max(dens.min(),1), vmax=dens.max()),
                          rasterized=True)

    formula_text = {
        'v5t': (
            r"$\bf{v5t\ (23\ parameters)}$" + "\n\n"
            r"$\rm Sci_{rec} = \dfrac{1}{L_{cyc}\cdot 16\mu s}\left[(PHO-Large)\dfrac{L_{cyc}-Dt}{L_{cyc}} - Wide\right] - C$" + "\n\n"
            r"$C(\rm det, |mlat|, t) = s_{0,det}\, g(t)\,\left[1 + k(t)\,\max(0,|mlat|-20^{\circ})^2\right] + C_0$" + "\n\n"
            r"$g(t) = 1 - \beta\,t$  (PMT outgassing, linear)" + "\n"
            r"$k(t) = c_0 + a_1\cos\omega t + b_1\sin\omega t,\ \omega = 2\pi/11\,\rm yr$" + "\n\n"
            r"$s_{0,\rm det}$ : 18 per-detector sensitivities at $t_0$" + "\n"
            r"$C_0$ : common-mode offset (already calibrated)" + "\n\n"
            r"$23 = 18\,s_{0,\rm det} + \beta + 3\ k\text{-}coeffs + C_0 + \omega$"
        ),
        'A': (
            r"$\bf{A\ -\ linear\ ACD\ (37\ parameters\ with\ offset)}$" + "\n\n"
            r"$C_{\rm det}(t) = a_{\rm det} + b_{\rm det}\cdot ACD(t) + c_0$" + "\n\n"
            r"$ACD(t) = \sum_{i=0}^{17} \rm Cnt\_VetoDet_i(t)$" + "\n"
            r"  (engineering 1 Hz, APID 0548 ASU board)" + "\n\n"
            r"$a_{\rm det}$ : 18 per-det intercepts" + "\n"
            r"$b_{\rm det}$ : 18 per-det slopes" + "\n"
            r"$c_0$ : global median-correction offset" + "\n\n"
            r"$37 = 18\,a_{\rm det} + 18\,b_{\rm det} + c_0$"
        ),
        'B': (
            r"$\bf{B\ -\ cubic\ per\text{-}det\ (73\ parameters)}$" + "\n\n"
            r"$C_{\rm det}(t) = a_{\rm det} + b_{\rm det}\,ACD + c_{\rm det}\,ACD^2 + d_{\rm det}\,ACD^3 + c_0$" + "\n\n"
            r"$a_{\rm det}, b_{\rm det}, c_{\rm det}, d_{\rm det}$ : independent" + "\n"
            r"  cubic coefficients per detector (full freedom)" + "\n\n"
            r"$c_0$ : global median-correction" + "\n\n"
            r"Highest per-det degree → maximum flexibility" + "\n"
            r"to capture local C(ACD) curvature" + "\n\n"
            r"$73 = 4 \times 18 + c_0$"
        ),
        'C': (
            r"$\bf{C\ -\ shared\ 4th\text{-}order\ shape\ (23\ parameters)}$" + "\n\n"
            r"$C_{\rm det}(t) = s_{\rm det}\left[x + \alpha\,x^2 + \beta\,x^3 + \gamma\,x^4\right] + c_0$" + "\n"
            r"  with  $x = ACD(t)/10^4$" + "\n\n"
            r"$s_{\rm det}$ : 18 per-det amplitudes" + "\n"
            r"$\alpha, \beta, \gamma$ : shared shape coefficients" + "\n"
            r"$c_0$ : common-mode offset" + "\n\n"
            r"Same structure as v5t: per-det amplitude × global" + "\n"
            r"non-linear shape. ACD replaces $|mlat|$ as axis." + "\n\n"
            r"$23 = 18\,s_{\rm det} + \alpha + \beta + \gamma + c_0$" + "\n"
            r"$\bf{(exactly\ v5t\ parameter\ count!)}$"
        ),
        'D': (
            r"$\bf{D\ -\ v5t\text{-}form\ with\ ACD\ (21\ parameters)}$" + "\n\n"
            r"$C_{\rm det}(t) = s_{\rm det}\left[1 + \gamma\,\max(0,\,ACD - thr)^2\right] + c_0$" + "\n\n"
            r"$s_{\rm det}$ : 18 per-det baselines" + "\n"
            r"$\gamma$ : quadratic coefficient" + "\n"
            r"$\rm thr$ : ACD threshold above which C rises" + "\n"
            r"$c_0$ : median-correction offset" + "\n\n"
            r"Direct analog of v5t's $[1 + k\,\max(0,|mlat|-20^{\circ})^2]$" + "\n"
            r"with mlat axis replaced by ACD axis." + "\n\n"
            r"Bounded fit: $s\!>\!0,\ \gamma\!>\!0,\ 5{\rm k}\!<\!thr\!<\!80{\rm k}$" + "\n\n"
            r"$21 = 18\,s_{\rm det} + \gamma + thr + c_0$"
        ),
    }

    LO, HI = 10, 10000
    for i, tag in enumerate(['v5t','A','B','C','D']):
        ax_s = fig.add_subplot(gs[i, 0])
        r=np.concatenate(R[tag])-offsets[tag]
        s=np.concatenate(S[tag])
        sci_rec = r + s
        valid = (s>=LO)&(sci_rec>0)
        sc = density_scatter(ax_s, s[valid], sci_rec[valid], xlog=True, ylog=True)
        xs = np.logspace(0, 4, 50)
        ax_s.plot(xs, xs, 'r-', lw=1.5, label="$y = x$ (perfect)")
        ax_s.set_xscale('log'); ax_s.set_yscale('log')
        ax_s.set_xlim(LO, 10000); ax_s.set_ylim(1, 10000)
        ax_s.set_xlabel(r"$\rm Sci_{obs}$ (cnt/s)", fontsize=12)
        ax_s.set_ylabel(r"$\rm Sci_{rec}$ (cnt/s)", fontsize=12)
        m = M[tag]
        metric_str = (
            f"median = {m['median']:+.2f}   MAD = {m['MAD']:.1f}   "
            f"std = {m['std']:.1f}   RMS = {m['RMS']:.1f}\n"
            f"$P_5$ = {m['P05']:+.1f}   $P_{{95}}$ = {m['P95']:+.1f}   "
            f"skew = {m['skewness']:+.2f}   below $y\!=\!x$ = {m['below']:.3f}%"
        )
        ax_s.set_title(f"{tag}    N = {m['N']/1e6:.1f}M    (offset applied: {offsets[tag]:+.2f})\n{metric_str}",
                       fontsize=11)
        ax_s.grid(alpha=0.3, which='both')
        ax_s.legend(loc='lower right', fontsize=10)
        cb = fig.colorbar(sc, ax=ax_s, shrink=0.75, pad=0.02)
        cb.set_label("local density", fontsize=10)

        ax_t = fig.add_subplot(gs[i, 1])
        ax_t.axis('off')
        ax_t.text(0.02, 0.98, formula_text[tag], fontsize=12, va='top', ha='left',
                  transform=ax_t.transAxes, family='serif',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f8f8', edgecolor='gray'))

    fig.suptitle(
        r"PHO-conservation reconstruction: 5 OPTIMIZED models" + "\n"
        r"(full mission 2017–2026, NO clean filter, ~24.8 M rows, median offset applied)",
        fontsize=14, fontweight='bold', y=0.995
    )
    Path("plots").mkdir(exist_ok=True)
    out = "plots/v6_models_optimal.png"
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
