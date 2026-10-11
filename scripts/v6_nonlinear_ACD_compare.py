#!/usr/bin/env python3
"""Compare 4 ACD-driven C-models against v5t baseline.

Models (all run through SAME unwrap + event-balance cap pipeline,
metrics computed on FULL DATA, NO clean filter):

  v5t             : 23 params (baseline)        C = s0·g·[1+k·mlat²] + C0
  A linear        : 36 params (current v6_acd)  C = a_det + b_det·ACD
  B quadratic     : 54 params                   C = a_det + b_det·ACD + c_det·ACD²
  C shared cubic  : 21 params                   C = s_det · poly3(ACD/1e4)
                                                where poly3 = x + α·x² + β·x³
  D v5t-axis-ACD  : 20 params                   C = s_det · [1 + γ·max(0,ACD-thr)²]

Report blob/main, MAD, median, below-y=x; pick the best.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit, least_squares

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


def stats(resid, sci_obs, n_below, N):
    med=np.median(resid)
    mad=np.median(np.abs(resid-med))
    blob=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-300)&(resid<=-50)).sum()
    main=((sci_obs>=800)&(sci_obs<=2500)&(resid>=-50)&(resid<=100)).sum()
    return dict(median=med, MAD=mad,
                blob_main=blob/max(main,1)*100,
                below=n_below/max(N,1)*100, N=N)


def main():
    # Load v5t calib for baseline
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det_v5=cz["s0_det"]; beta_v5=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w_v5=float(cz["w"]); kc_v5=cz["k_coeffs"]; C0_v5=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    # ─── PASS 1: collect (acd, C_truth, detid) for all model fits ───
    print("=== PASS 1: collecting training data ===", flush=True)
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
        print(f"  {os.path.basename(f)}: scanned", flush=True)
    acd=np.concatenate(acd_all); Cv=np.concatenate(C_all); detv=np.concatenate(det_all)
    print(f"  N={len(acd):,}", flush=True)

    # Sub-sample for nonlinear fits (full N=24M kills least_squares)
    if len(acd) > 2_000_000:
        idx = np.random.RandomState(42).choice(len(acd), 2_000_000, replace=False)
        acd_fit = acd[idx]; Cv_fit = Cv[idx]; detv_fit = detv[idx]
        print(f"  using {len(acd_fit):,} sub-sample rows for nonlinear fits", flush=True)
    else:
        acd_fit = acd; Cv_fit = Cv; detv_fit = detv

    # ─── Fit model A: linear per-det ───
    print("\n=== Model A: linear (36 params) ===", flush=True)
    A_a=np.zeros(18); A_b=np.zeros(18)
    for d in range(18):
        m=(detv==d)
        if m.sum()<1000: continue
        coef=np.polyfit(acd[m],Cv[m],1); A_b[d],A_a[d]=coef
    print(f"  a_det median {np.median(A_a):+.1f}, b_det median {np.median(A_b):+.4e}")

    # ─── Fit model B: quadratic per-det ───
    print("\n=== Model B: quadratic per-det (54 params) ===", flush=True)
    B_a=np.zeros(18); B_b=np.zeros(18); B_c=np.zeros(18)
    for d in range(18):
        m=(detv==d)
        if m.sum()<1000: continue
        coef=np.polyfit(acd[m],Cv[m],2)  # returns [c, b, a]
        B_c[d],B_b[d],B_a[d]=coef
    print(f"  a_det median {np.median(B_a):+.1f}, b median {np.median(B_b):+.4e}, c median {np.median(B_c):+.4e}")

    # ─── Fit model C: shared cubic shape × per-det amplitude ───
    # C = s_det * (x + α·x² + β·x³), x = ACD/1e4
    # 18 s_det + 2 shape (α,β) = 20 params, but each s_det also can have +offset → 21 if include offset
    # Use objective: minimize Σ (C_truth - s_det·shape(ACD))² jointly
    print("\n=== Model C: shared cubic shape × s_det (21 params) ===", flush=True)
    # Initial guess: s_det from A_b (slope), α=β=0 (degenerates to linear)
    # We optimize: minimize Σ (Cv - s_det·(x + α·x² + β·x³))²
    x = acd / 1e4
    s_C = np.zeros(18)
    # Initial s_det from linear fit slope, then iterate with α,β
    alpha = 0.0; beta = 0.0; offset_C = 0.0
    for d in range(18):
        s_C[d] = A_b[d] * 1e4  # linear slope rescaled
    # Joint nonlinear least squares
    def model_C(params, acd, det):
        s = params[:18]
        alpha = params[18]; beta = params[19]; off = params[20]
        x = acd / 1e4
        return off + s[det] * (x + alpha*x**2 + beta*x**3)
    def residual_C(params):
        pred = model_C(params, acd_fit, detv_fit)
        return Cv_fit - pred
    p0_C = np.concatenate([s_C, [0.0, 0.0, 100.0]])  # offset starts at 100
    res = least_squares(residual_C, p0_C, method='trf', max_nfev=100)
    C_params = res.x
    s_C = C_params[:18]; alpha = C_params[18]; beta = C_params[19]; off_C = C_params[20]
    print(f"  s_det median {np.median(s_C):+.1f}, α={alpha:+.4f}, β={beta:+.4f}, offset={off_C:+.1f}")
    print(f"  cost = {res.cost:.0f}")

    # ─── Fit model D: v5t-axis-ACD ───
    # C = s_det · [1 + γ·max(0, ACD-thr)²] + offset
    print("\n=== Model D: v5t form with ACD axis (20 params) ===", flush=True)
    # 18 s_det + γ + thr = 20 params (no separate offset; absorb into s_det baseline)
    # Plus 1 offset = 21 if needed. Try 20 first.
    def model_D(params, acd, det):
        s = params[:18]
        gamma = params[18]; thr = params[19]
        return s[det] * (1.0 + gamma * np.maximum(0.0, acd-thr)**2)
    def residual_D(params):
        pred = model_D(params, acd_fit, detv_fit)
        return Cv_fit - pred
    # initial: s_det = a_det (intercept), γ~1e-9, thr=20000
    p0_D = np.concatenate([A_a, [1e-9, 20000.0]])
    res = least_squares(residual_D, p0_D, method='trf', max_nfev=100)
    D_params = res.x
    s_D = D_params[:18]; gamma_D = D_params[18]; thr_D = D_params[19]
    print(f"  s_det median {np.median(s_D):+.1f}, γ={gamma_D:+.4e}, thr={thr_D:+.0f}")
    print(f"  cost = {res.cost:.0f}")

    # ─── PASS 2: apply all 5 models on same data ───
    print("\n=== PASS 2: applying all models ===", flush=True)
    R = {k: [] for k in ['v5t', 'A', 'B', 'C', 'D']}
    S = {k: [] for k in ['v5t', 'A', 'B', 'C', 'D']}
    nb = {k: 0 for k in ['v5t', 'A', 'B', 'C', 'D']}
    nt = {k: 0 for k in ['v5t', 'A', 'B', 'C', 'D']}
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

            # v5t
            C_v5=s0_det_v5[detid]*g_t*(1.0+k_t*mt)+C0_v5
            # A linear
            C_A=A_a[detid]+A_b[detid]*acdv
            # B quadratic
            C_B=B_a[detid]+B_b[detid]*acdv+B_c[detid]*acdv**2
            # C shared cubic
            x=acdv/1e4
            C_C=off_C + s_C[detid]*(x + alpha*x**2 + beta*x**3)
            # D v5t-style with ACD
            C_D=s_D[detid]*(1.0 + gamma_D*np.maximum(0.0, acdv-thr_D)**2)

            for tag, C in [('v5t',C_v5), ('A',C_A), ('B',C_B), ('C',C_C), ('D',C_D)]:
                base = apply_pipeline(pho,lg,wd,sci,lc,dtv,C)
                resid = base - sci - C
                ok=np.isfinite(resid)&np.isfinite(base)&(sci>0)&(base>0)
                R[tag].append(resid[ok]); S[tag].append(sci[ok])
                nt[tag]+=ok.sum(); nb[tag]+=(base[ok]<sci[ok]).sum()
        print(f"  {os.path.basename(f)}: applied", flush=True)

    # ─── Compute metrics ───
    M = {}
    for tag in ['v5t', 'A', 'B', 'C', 'D']:
        r = np.concatenate(R[tag]); s = np.concatenate(S[tag])
        M[tag] = stats(r, s, nb[tag], nt[tag])

    print(f"\n=== METRICS ===")
    print(f"  model         N             median   MAD    blob/main  below-y=x")
    descriptions = {
        'v5t': 'v5t (23 params, s0·g·[1+k·mlat²]+C0)',
        'A':   'A linear (36 params, a+b·ACD)',
        'B':   'B quadratic (54 params, a+b·ACD+c·ACD²)',
        'C':   'C shared cubic (21 params, s·poly3(ACD/1e4)+off)',
        'D':   'D v5t-form ACD axis (20 params, s·[1+γ(ACD-thr)²])',
    }
    for tag in ['v5t', 'A', 'B', 'C', 'D']:
        m = M[tag]
        print(f"  {tag:>5}  N={m['N']:>9,}  median={m['median']:+7.2f}  MAD={m['MAD']:>5.1f}  blob={m['blob_main']:>5.2f}%  below={m['below']:.4f}%   {descriptions[tag]}")

    # ─── Save fit parameters for downstream ───
    npz_path = "n_below_study/v5_npz/v6_nonlinear_fits.npz"
    np.savez(npz_path,
             A_a=A_a, A_b=A_b,
             B_a=B_a, B_b=B_b, B_c=B_c,
             C_s=s_C, C_alpha=alpha, C_beta=beta, C_off=off_C,
             D_s=s_D, D_gamma=gamma_D, D_thr=thr_D)
    print(f"\n  Saved fits → {npz_path}")

    # ─── Plot: Sci_rec vs Sci_obs for all 5 models ───
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    LO, HI = 10, 10000
    for ax, tag in zip(axes[:5], ['v5t','A','B','C','D']):
        r = np.concatenate(R[tag]); s = np.concatenate(S[tag])
        sci_rec = r + s
        valid = (s >= LO) & (sci_rec > 0)
        ss = s[valid]; sr = sci_rec[valid]
        if len(ss) > 500000:
            idx = np.random.RandomState(0).choice(len(ss), 500000, replace=False)
            ss = ss[idx]; sr = sr[idx]
        # density via 2d hist
        H, xe, ye = np.histogram2d(np.log10(ss), np.log10(sr), bins=120)
        ix = np.clip(np.searchsorted(xe, np.log10(ss))-1, 0, H.shape[0]-1)
        iy = np.clip(np.searchsorted(ye, np.log10(sr))-1, 0, H.shape[1]-1)
        dens = H[ix, iy]
        order = np.argsort(dens)
        ax.scatter(ss[order], sr[order], c=dens[order], s=1.5, cmap='viridis',
                   edgecolors='none', norm=mpl.colors.LogNorm(vmin=1, vmax=max(2,dens.max())),
                   rasterized=True)
        xs=np.logspace(0,4,50); ax.plot(xs,xs,'r-',lw=1.2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(LO, 10000); ax.set_ylim(1, 10000)
        ax.set_xlabel(r"$\rm Sci_{obs}$"); ax.set_ylabel(r"$\rm Sci_{rec}$")
        m = M[tag]
        ax.set_title(f"{tag}: blob/main={m['blob_main']:.2f}%, MAD={m['MAD']:.1f}, below={m['below']:.3f}%",
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, which='both')
    axes[5].axis('off')
    text = "Models compared (FULL DATA, NO clean filter):\n\n"
    for tag in ['v5t','A','B','C','D']:
        text += f"{tag}: {descriptions[tag]}\n"
        text += f"     blob/main={M[tag]['blob_main']:.2f}%, MAD={M[tag]['MAD']:.1f}, median={M[tag]['median']:+.1f}\n\n"
    axes[5].text(0.02, 0.95, text, fontsize=11, va='top', family='monospace')
    fig.suptitle("v5t baseline vs 4 ACD-driven C-model forms", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = "plots/v6_nonlinear_compare.png"
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
