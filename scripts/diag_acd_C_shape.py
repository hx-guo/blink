#!/usr/bin/env python3
"""Diagnose the C vs ACD_sum functional shape.

If C(ACD) is genuinely linear, the v6_acd model is the right form.
If it's concave / saturating / piecewise, the linear fit over-predicts C
at high ACD → causes the leftward tails in compare_v5_vs_v6 plot.

Compute per-row C_truth = base(C=150 unwrap) - sci, then bin by ACD_sum
and plot median C per bin per det.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

L = 16e-6
NEEDED = ["box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt","ACD_sum"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"


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


def main():
    files=sorted(glob.glob(os.path.join(CACHE,"clean_relaxed_20*.parquet")))
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
            ok=np.isfinite(C_truth)&(np.abs(C_truth)<500)&(acd>0)&(acd<150000)
            acd_all.append(acd[ok]); C_all.append(C_truth[ok]); det_all.append(detid[ok])
        print(f"  {os.path.basename(f)}: scanned", flush=True)
    acd=np.concatenate(acd_all); Cv=np.concatenate(C_all); detv=np.concatenate(det_all)
    print(f"\nN={len(acd):,}")
    print(f"ACD range: {acd.min():.0f} – {acd.max():.0f}, p99={np.percentile(acd,99):.0f}")

    # === Global C(ACD) — binned median ===
    bins = np.logspace(np.log10(500), np.log10(150000), 50)
    centers = np.sqrt(bins[:-1]*bins[1:])
    median_C=[]; p25_C=[]; p75_C=[]; counts=[]
    for i in range(len(bins)-1):
        m=(acd>=bins[i])&(acd<bins[i+1])
        if m.sum()<100:
            median_C.append(np.nan); p25_C.append(np.nan); p75_C.append(np.nan); counts.append(0)
        else:
            median_C.append(np.median(Cv[m]))
            p25_C.append(np.percentile(Cv[m],25))
            p75_C.append(np.percentile(Cv[m],75))
            counts.append(m.sum())
    median_C=np.array(median_C); p25_C=np.array(p25_C); p75_C=np.array(p75_C)

    print(f"\n=== global C vs ACD (binned median) ===")
    print(f"   ACD     n      med_C   p25   p75")
    for i in range(0,len(centers),5):
        if counts[i]>0:
            print(f"  {centers[i]:>7.0f} {counts[i]:>9,d} {median_C[i]:>+7.1f} {p25_C[i]:>+6.1f} {p75_C[i]:>+6.1f}")

    # Linear fit on ACD < 30000 (low-mid region only) to see how high-ACD departs
    low_mask=centers<30000
    valid=low_mask & np.isfinite(median_C)
    coef = np.polyfit(centers[valid], median_C[valid], 1)
    b_low, a_low = coef
    print(f"\nLow-ACD (ACD<30k) linear fit: C = {a_low:+.1f} + {b_low:+.4e} * ACD")
    # Compare with global linear fit
    valid_all=np.isfinite(median_C)
    coef_all = np.polyfit(centers[valid_all], median_C[valid_all], 1)
    b_all, a_all = coef_all
    print(f"Global linear fit:          C = {a_all:+.1f} + {b_all:+.4e} * ACD")
    print(f"Slope ratio b_all/b_low = {b_all/b_low:.3f}  (1.0 = perfectly linear)")

    # ─── Plot ───
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax=axes[0,0]
    # 2D density scatter of C vs ACD
    H, xe, ye = np.histogram2d(np.log10(acd), Cv, bins=[80, 100])
    ax.imshow(H.T, origin='lower', aspect='auto', cmap='viridis',
              extent=[xe[0],xe[-1],ye[0],ye[-1]], norm=plt.matplotlib.colors.LogNorm())
    ax.plot(np.log10(centers), median_C, 'r-', lw=2.5, label='binned median')
    ax.plot(np.log10(centers), p25_C, 'r--', lw=1, alpha=0.6, label='25/75 pct')
    ax.plot(np.log10(centers), p75_C, 'r--', lw=1, alpha=0.6)
    # Show low-ACD linear extrapolation
    x_line=np.linspace(centers[0], centers[-1], 100)
    ax.plot(np.log10(x_line), a_low + b_low * x_line, 'w--', lw=1.5,
            label=f'linear fit on ACD<30k\n(extrapolated)')
    ax.plot(np.log10(x_line), a_all + b_all * x_line, 'y-', lw=1.5,
            label=f'linear fit on all ACD')
    ax.set_xlabel('log10(ACD_sum)'); ax.set_ylabel('C_truth (per row)')
    ax.set_title('C vs ACD_sum: density + binned median'); ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # Per-det shape: pick 4 representative dets
    pick_dets=[0, 7, 12, 17]  # A0, B1, C0, C5
    ax=axes[0,1]
    for d in pick_dets:
        m=(detv==d)
        if m.sum()<5000: continue
        meds=[]; cs=[]
        for i in range(len(bins)-1):
            mm=m&(acd>=bins[i])&(acd<bins[i+1])
            if mm.sum()<50: meds.append(np.nan); cs.append(np.nan); continue
            meds.append(np.median(Cv[mm])); cs.append(centers[i])
        meds=np.array(meds); cs=np.array(cs)
        valid=np.isfinite(meds)
        ax.plot(cs[valid], meds[valid], 'o-', label=f"det {'ABC'[d//6]}{d%6}")
    ax.set_xscale('log')
    ax.set_xlabel('ACD_sum'); ax.set_ylabel('median C_truth')
    ax.set_title('C(ACD) per detector — same shape across dets?')
    ax.legend(); ax.grid(alpha=0.3)

    # Linear residuals — does C lie above or below the linear fit?
    ax=axes[1,0]
    Cv_pred_global=a_all + b_all * acd
    Cv_pred_low=a_low + b_low * acd
    sub=np.random.RandomState(0).choice(len(acd), min(200000, len(acd)), replace=False)
    ax.scatter(acd[sub], Cv[sub]-Cv_pred_global[sub], s=0.5, alpha=0.2, c='gray',
               label='C_truth - linear_global', rasterized=True)
    ax.axhline(0, color='r', lw=1)
    ax.set_xscale('log'); ax.set_xlabel('ACD_sum'); ax.set_ylabel('C - linear fit')
    ax.set_title('Residual from global linear fit (positive = under-predicted)')
    ax.set_ylim(-200, 200); ax.grid(alpha=0.3)

    # Try alternate forms and see which fits best
    ax=axes[1,1]
    ax.plot(np.log10(centers), median_C, 'k-', lw=2, label='data')
    # Linear (global)
    ax.plot(np.log10(centers), a_all + b_all*centers, 'r-', alpha=0.7,
            label=f'linear: a+b·ACD (RMS={np.sqrt(np.nanmean((median_C-(a_all+b_all*centers))**2)):.1f})')
    # Saturating: C = a + b·ACD / (1 + ACD/ACD_sat)
    from scipy.optimize import curve_fit
    def sat(x, a, b, x_sat):
        return a + b*x/(1+x/x_sat)
    try:
        popt, _ = curve_fit(sat, centers[valid_all], median_C[valid_all],
                            p0=[100, 0.005, 50000], maxfev=5000)
        sat_fit = sat(centers, *popt)
        rms_sat=np.sqrt(np.nanmean((median_C-sat_fit)**2))
        ax.plot(np.log10(centers), sat_fit, 'g-', alpha=0.7,
                label=f'saturating: a+b·ACD/(1+ACD/x_sat) (RMS={rms_sat:.1f})\n  a={popt[0]:.1f}, b={popt[1]:.4f}, x_sat={popt[2]:.0f}')
    except Exception as e:
        print(f'saturating fit failed: {e}')
    # Log
    coef_log=np.polyfit(np.log(centers[valid_all]), median_C[valid_all], 1)
    log_fit=coef_log[0]*np.log(centers) + coef_log[1]
    rms_log=np.sqrt(np.nanmean((median_C-log_fit)**2))
    ax.plot(np.log10(centers), log_fit, 'b-', alpha=0.7,
            label=f'log: a+b·ln(ACD) (RMS={rms_log:.1f})')
    # Sqrt
    coef_sq=np.polyfit(np.sqrt(centers[valid_all]), median_C[valid_all], 1)
    sq_fit=coef_sq[0]*np.sqrt(centers) + coef_sq[1]
    rms_sq=np.sqrt(np.nanmean((median_C-sq_fit)**2))
    ax.plot(np.log10(centers), sq_fit, 'm-', alpha=0.7,
            label=f'sqrt: a+b·√ACD (RMS={rms_sq:.1f})')
    ax.set_xlabel('log10(ACD_sum)'); ax.set_ylabel('median C_truth')
    ax.set_title('Functional form candidates'); ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out="plots/diag_acd_C_shape.png"
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
