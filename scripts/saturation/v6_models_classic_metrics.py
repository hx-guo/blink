#!/usr/bin/env python3
"""5-model comparison with CLASSIC statistics only (no blob heuristic).

Loads v5t calib + previously-saved v6 nonlinear fits (n_below_study/v5_npz/v6_nonlinear_fits.npz).
Runs PASS 2 only — applies each model + collects residuals + computes metrics.

Metrics (all classic):
  median, MAD, std, RMS, skewness, P05, P95, below-y=x fraction

Plot: 5 panels (one per model) with detailed LaTeX formula + parameter values + metrics.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
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


def classic_stats(resid, sci_obs, n_below, N):
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
    fits=np.load("n_below_study/v5_npz/v6_nonlinear_fits.npz")
    A_a=fits["A_a"]; A_b=fits["A_b"]
    B_a=fits["B_a"]; B_b=fits["B_b"]; B_c=fits["B_c"]
    s_C=fits["C_s"]; alpha=float(fits["C_alpha"]); beta=float(fits["C_beta"]); off_C=float(fits["C_off"])
    s_D=fits["D_s"]; gamma_D=float(fits["D_gamma"]); thr_D=float(fits["D_thr"])

    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))

    R = {k: [] for k in ['v5t', 'A', 'B', 'C', 'D']}
    S = {k: [] for k in ['v5t', 'A', 'B', 'C', 'D']}
    nb = {k: 0 for k in ['v5t', 'A', 'B', 'C', 'D']}
    nt = {k: 0 for k in ['v5t', 'A', 'B', 'C', 'D']}

    print("Applying 5 models...", flush=True)
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
            C_B=B_a[detid]+B_b[detid]*acdv+B_c[detid]*acdv**2
            x=acdv/1e4
            C_C=off_C + s_C[detid]*(x + alpha*x**2 + beta*x**3)
            C_D=s_D[detid]*(1.0 + gamma_D*np.maximum(0.0, acdv-thr_D)**2)

            for tag, C in [('v5t',C_v5), ('A',C_A), ('B',C_B), ('C',C_C), ('D',C_D)]:
                base = apply_pipeline(pho,lg,wd,sci,lc,dtv,C)
                resid = base - sci - C
                ok=np.isfinite(resid)&np.isfinite(base)&(sci>0)&(base>0)
                R[tag].append(resid[ok]); S[tag].append(sci[ok])
                nt[tag]+=ok.sum(); nb[tag]+=(base[ok]<sci[ok]).sum()
        print(f"  {os.path.basename(f)}: done", flush=True)

    M = {tag: classic_stats(np.concatenate(R[tag]), np.concatenate(S[tag]), nb[tag], nt[tag])
         for tag in ['v5t','A','B','C','D']}

    print(f"\n{'model':>5}  {'N':>11}  {'median':>7}  {'MAD':>5}  {'std':>6}  {'RMS':>6}  {'P05':>7}  {'P95':>7}  {'skew':>6}  {'below%':>7}")
    for tag in ['v5t','A','B','C','D']:
        m=M[tag]
        print(f"  {tag:>3}  {m['N']:>11,}  {m['median']:+6.2f}  {m['MAD']:>5.1f}  {m['std']:>6.1f}  {m['RMS']:>6.1f}  {m['P05']:+6.1f}  {m['P95']:+6.1f}  {m['skewness']:+5.2f}  {m['below']:>6.4f}%")

    # ─── Build plot ───
    mpl.rcParams.update({"font.family": "DejaVu Sans", "text.usetex": False})
    fig = plt.figure(figsize=(22, 28))
    gs = fig.add_gridspec(5, 2,
                          width_ratios=[1.0, 0.7],
                          height_ratios=[1, 1, 1, 1, 1],
                          hspace=0.32, wspace=0.20)

    # Helper density scatter
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

    # Formula text per model (mathtext-compatible)
    formula_text = {
        'v5t': (
            r"$\bf{v5t\ (23\ parameters)}$" + "\n\n"
            r"$\rm Sci_{rec} = \dfrac{1}{L_{cyc}\cdot 16\mu s}\left[(PHO-Large)\dfrac{L_{cyc}-Dt}{L_{cyc}} - Wide\right] - C$" + "\n\n"
            r"$C(\rm det, |mlat|, t) = s_{0,det}\, g(t)\,\left[1 + k(t)\,\max(0,|mlat|-20^{\circ})^2\right] + C_0$" + "\n\n"
            r"$g(t) = 1 - \beta\,t$  (PMT outgassing, linear)" + "\n"
            r"$k(t) = c_0 + a_1\cos\omega t + b_1\sin\omega t,\ \ \omega = 2\pi/11\,\rm yr$  (solar cycle)" + "\n\n"
            r"$s_{0,\rm det}$ : 18 per-detector sensitivities at $t_0$" + "\n"
            r"$C_0$ : common-mode offset" + "\n\n"
            r"Parameters: $18\,s_{0,\rm det} + \beta + 3\ \rm k\text{-}coeffs + C_0 + \omega = 23$"
        ),
        'A': (
            r"$\bf{A\ -\ linear\ ACD\ (36\ parameters)}$" + "\n\n"
            r"$C(\rm det,\,row) = a_{det} + b_{det}\,\cdot ACD\_sum(row)$" + "\n\n"
            r"$\rm ACD\_sum = \sum_{i=0}^{17} Cnt\_VetoDet_i$" + "\n"
            r"  (engineering 1 Hz, APID 0548 ASU board)" + "\n\n"
            r"$a_{\rm det}$ : 18 per-detector intercepts (cnt/s)" + "\n"
            r"$b_{\rm det}$ : 18 per-detector slopes (cnt/s per unit ACD)" + "\n\n"
            r"Parameters: $18\,a_{\rm det} + 18\,b_{\rm det} = 36$"
        ),
        'B': (
            r"$\bf{B\ -\ quadratic\ per\text{-}det\ (54\ parameters)}$" + "\n\n"
            r"$C(\rm det,\,row) = a_{det} + b_{det}\,ACD + c_{det}\,ACD^2$" + "\n\n"
            r"$a_{\rm det}, b_{\rm det}, c_{\rm det}$ : independent quadratic" + "\n"
            r"  coefficients per detector" + "\n\n"
            r"$c_{\rm det}<0$ captures saturation:" + "\n"
            r"$C$ rises with ACD then plateaus / declines at high ACD" + "\n"
            r"(visible in C-vs-ACD diagnostic, peak near ACD$\approx$50,000)" + "\n\n"
            r"Parameters: $3 \times 18 = 54$"
        ),
        'C': (
            r"$\bf{C\ -\ shared\ cubic\ shape}$" + "\n\n"
            r"$C(\rm det,\,row) = s_{det}\,\left[\,x + \alpha\,x^2 + \beta\,x^3\right] + offset$" + "\n"
            r"$x = ACD\_sum/10^4$" + "\n\n"
            r"$s_{\rm det}$ : 18 per-detector amplitudes" + "\n"
            r"$\alpha, \beta$ : shared cubic-shape coefficients (all dets)" + "\n"
            r"$\rm offset$ : common-mode" + "\n\n"
            r"Same structure as v5t: per-det amplitude $\times$" + "\n"
            r"global non-linear shape. ACD replaces $|mlat|$ as axis." + "\n\n"
            r"Parameters: $18\,s_{\rm det} + \alpha + \beta + \rm offset = 21$"
        ),
        'D': (
            r"$\bf{D\ -\ v5t\text{-}form\ with\ ACD\ axis\ (20\ parameters)}$" + "\n\n"
            r"$C(\rm det,\,row) = s_{det}\,\left[1 + \gamma\,\max(0,\,ACD\_sum - thr)^2\right]$" + "\n\n"
            r"$s_{\rm det}$ : 18 per-detector baseline" + "\n"
            r"$\gamma$ : global second-order coefficient" + "\n"
            r"$\rm thr$ : ACD threshold above which $C$ rises" + "\n\n"
            r"Direct analog of v5t's $[1 + k\,\max(0,|mlat|-20^{\circ})^2]$" + "\n"
            r"with mlat axis replaced by ACD axis." + "\n\n"
            r"Parameters: $18\,s_{\rm det} + \gamma + \rm thr = 20$" + "\n\n"
            r"$\bf{Note\!:}$ least-squares fit did not converge to a" + "\n"
            r"physical mode; included for completeness."
        ),
    }

    LO, HI = 10, 10000
    for i, tag in enumerate(['v5t', 'A', 'B', 'C', 'D']):
        # Left: scatter
        ax_s = fig.add_subplot(gs[i, 0])
        r = np.concatenate(R[tag]); s = np.concatenate(S[tag])
        sci_rec = r + s
        valid = (s >= LO) & (sci_rec > 0)
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
        ax_s.set_title(f"{tag}    N = {m['N']/1e6:.1f}M\n{metric_str}",
                       fontsize=11)
        ax_s.grid(alpha=0.3, which='both')
        ax_s.legend(loc='lower right', fontsize=10)
        cb = fig.colorbar(sc, ax=ax_s, shrink=0.75, pad=0.02)
        cb.set_label("local density", fontsize=10)

        # Right: LaTeX formula
        ax_t = fig.add_subplot(gs[i, 1])
        ax_t.axis('off')
        ax_t.text(0.02, 0.98, formula_text[tag], fontsize=12, va='top', ha='left',
                  transform=ax_t.transAxes, family='serif',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f8f8', edgecolor='gray'))

    fig.suptitle(
        r"5-model comparison of PHO-conservation reconstruction" + "\n"
        r"(full mission 2017–2026, NO clean filter, ~24.8 M rows, classic statistics)",
        fontsize=14, fontweight='bold', y=0.995
    )
    Path("plots").mkdir(exist_ok=True)
    out = "plots/v6_models_classic_metrics.png"
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
