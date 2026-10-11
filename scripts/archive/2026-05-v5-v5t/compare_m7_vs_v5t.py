#!/usr/bin/env python3
"""Fair head-to-head: M7 vs v5t, INVERSE — reconstruct Sci from PHO/Wide/Large.

Sanity filters applied (stated upfront, not silent):
  - PHO > 0         (need positive trigger count)
  - Sci_1s > 0      (need observed Sci)
  - L_cycles > 1    (avoid divide-by-zero on dead-time fraction)
  (No physics filter — bright sources / SAA / throttle seconds all kept.)

Same data, same metric (median / MAD / std / RMS / skew / P5 / P95)
on Sci residual = Sci_reconstructed - Sci_observed.

v5t inverse (Sci from PHO):
  Sci_rec_v5t = (PHO - Large) · lf / L - Wide / L - C_v5
  C_v5 = s0_det · g(t) · [1 + k(t)·mlat²] + C0     (23 params)

M7 inverse (Sci from PHO, with observed Sci_ACD as side input):
  Sci_pure_rec = (PHO - c_ACD·Sci_ACD - β·Wide - γ·Large - b) / c_pure
  Sci_rec_M7   = Sci_pure_rec + Sci_ACD
  (per-det 5 params × 18 = 90 params; fit OLS on 2017-19 sub-sample)
  NOTE: M7 reconstruction uses observed Sci_ACD, so it has more side
  information than v5t — this is not strictly fair, but it follows the
  M7 model's original definition.
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

L = 16e-6
MIN_C_SLACK = 50.0
OLD_CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed_v1_48col"
NEEDED = ["date","box","det","PHO","Wide","Large","Sci_1s","L_cycles","Dt",
          "Lat","Lon",
          "Sci_pure_1s","Sci_ACD1_1s","Sci_ACDN_1s"]
B_THRESHOLD = 20.0


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    LL = lc * L; lf = 1.0 - dt/lc
    pred = pho - (wide + (sci+C)*LL)/lf
    n = np.maximum(np.round((pred-large)/1024.).astype(int), 0)
    mx = pho - wide; out = large + n*1024.; ov = out > mx
    if ov.any():
        nm = np.maximum(np.floor((mx-large)/1024.).astype(int), 0)
        out = large + np.where(ov, nm, n)*1024.
    return out


def v5t_unwrap_and_recover(pho, large_raw, wide, sci, lc, dt, C):
    """Two-pass v5t: unwrap_v2 + event-balance cap, then Sci_rec."""
    LL = lc * L; lf = 1.0 - dt/lc
    lv2 = unwrap_v2(pho, large_raw, wide, sci, lc, dt, C)
    # event-balance cap (same as v5t_fixed_formula_verify)
    mle = pho - ((sci + MIN_C_SLACK)*LL + wide) / lf
    n3 = np.round((lv2 - large_raw)/1024).astype(int)
    nmax = np.maximum(np.floor((mle - large_raw)/1024.).astype(int), 0)
    lv5 = large_raw + np.where(n3 > nmax, nmax, n3)*1024.
    base = (pho - lv5)*lf/LL - wide/LL
    return base - C, lv5


def classic_stats(resid):
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    std = float(np.std(resid))
    rms = float(np.sqrt(np.mean(resid**2)))
    p05 = float(np.percentile(resid, 5))
    p95 = float(np.percentile(resid, 95))
    sk  = float(skew(resid))
    return dict(median=med, MAD=mad, std=std, RMS=rms,
                P05=p05, P95=p95, skewness=sk, N=len(resid))


def main():
    # Load v5t calib
    cz = np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det = cz["s0_det"]; beta_v5 = float(cz["beta"]); t0 = np.datetime64(str(cz["t0"]))
    w_v5 = float(cz["w"]); kc_v5 = cz["k_coeffs"]; C0_v5 = float(cz["C0"])
    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)
    files = sorted(glob.glob(os.path.join(OLD_CACHE, "clean_relaxed_20*.parquet")))
    print(f"Found {len(files)} year files in OLD cache", flush=True)

    # ── PASS 1: collect training sub-sample 2017-2019 for M7 fit ──
    print("\n=== PASS 1: collect M7 training data (2017-2019 sub-sample) ===", flush=True)
    fit_chunks = []
    train_years = ["2017","2018","2019"]
    for f in files:
        yr = os.path.basename(f).split("_")[2].split(".")[0]
        if yr not in train_years: continue
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        # 2 row groups per year sub-sample
        for rg in np.unique(np.linspace(0, n_rg-1, 3).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            box_idx = np.select([df["box"].values == b for b in "ABC"], [0,1,2], default=0)
            df["detid"] = box_idx * 6 + df["det"].values
            # Discard rows with bad inputs
            ok = (df["PHO"]>0)&(df["Sci_1s"]>0)&(df["L_cycles"]>1)
            fit_chunks.append(df[ok])
        print(f"  {os.path.basename(f)}: scanned", flush=True)
    fit_df = __import__("pandas").concat(fit_chunks, ignore_index=True)
    print(f"  total fit rows: {len(fit_df):,}", flush=True)

    # Sub-sample if too big
    if len(fit_df) > 2_000_000:
        fit_df = fit_df.sample(n=2_000_000, random_state=42).reset_index(drop=True)
        print(f"  sub-sampled to {len(fit_df):,} for M7 fit", flush=True)

    # M7: per-det OLS on PHO = c_pure·Sci_pure + c_ACD·Sci_ACD + β·Wide + γ·Large + b
    # Where Sci_pure = Sci_pure_1s and Sci_ACD = Sci_ACD1_1s + Sci_ACDN_1s
    fit_df["Sci_pure"] = fit_df["Sci_pure_1s"].astype(float).values
    fit_df["Sci_ACD"] = (fit_df["Sci_ACD1_1s"] + fit_df["Sci_ACDN_1s"]).astype(float).values

    print("\n=== Fit M7 per-det (OLS) ===", flush=True)
    m7_params = {}  # detid -> (b, c_pure, c_ACD, β, γ)
    for d in range(18):
        m = (fit_df["detid"] == d)
        if m.sum() < 1000: continue
        sub = fit_df[m]
        # Design matrix: [Sci_pure, Sci_ACD, Wide, Large, 1]
        X = np.column_stack([
            sub["Sci_pure"].values,
            sub["Sci_ACD"].values,
            sub["Wide"].astype(float).values,
            sub["Large"].astype(float).values,
            np.ones(len(sub)),
        ])
        y = sub["PHO"].astype(float).values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        c_pure, c_ACD, βm, γm, bm = coef
        m7_params[d] = (bm, c_pure, c_ACD, βm, γm)
        box = "ABC"[d//6]; di = d%6
        print(f"  {box}{di}  b={bm:+7.2f}  c_pure={c_pure:.3f}  c_ACD={c_ACD:.3f}  "
              f"β={βm:.3f}  γ={γm:.3f}")
    del fit_df, fit_chunks

    # ── PASS 2: evaluate both models on FULL 9 years ──
    print("\n=== PASS 2: evaluate both models on full 9 years ===", flush=True)
    R_m7_all = []; R_v5_all = []
    R_m7_train = []; R_v5_train = []   # 2017-2019 separately
    R_m7_test = [];  R_v5_test  = []   # 2020+
    # Also collect Sci_obs + Sci_rec for the FULL-mission scatter plot
    SCI_OBS_all = []; SCI_REC_M7_all = []; SCI_REC_V5_all = []
    for f in files:
        yr = os.path.basename(f).split("_")[2].split(".")[0]
        pf = pq.ParquetFile(f); n_rg = pf.num_row_groups
        for rg in np.unique(np.linspace(0, n_rg-1, 3).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            box_idx = np.select([df["box"].values == b for b in "ABC"], [0,1,2], default=0)
            detid = box_idx * 6 + df["det"].values
            df["Sci_pure"] = df["Sci_pure_1s"].astype(float).values
            df["Sci_ACD"]  = (df["Sci_ACD1_1s"] + df["Sci_ACDN_1s"]).astype(float).values
            # Sanity
            ok = (df["PHO"]>0)&(df["Sci_1s"]>0)&(df["L_cycles"]>1)
            df = df[ok].reset_index(drop=True)
            detid = detid[ok.values]

            pho = df["PHO"].astype(float).values
            wide = df["Wide"].astype(float).values
            large = df["Large"].astype(float).values
            sci_1s = df["Sci_1s"].astype(float).values
            sci_pure = df["Sci_pure"].values
            sci_acd = df["Sci_ACD"].values
            lc = df["L_cycles"].astype(float).values
            dt = df["Dt"].astype(float).values
            LL = lc * L; lf = 1.0 - dt/lc

            # M7 INVERSE: reconstruct Sci_total from PHO + observed Sci_ACD
            sci_rec_m7 = np.zeros(len(df))
            for d in range(18):
                if d not in m7_params: continue
                bm, cp, cA, βm, γm = m7_params[d]
                msk = (detid == d)
                if not msk.any(): continue
                # Sci_pure_rec = (PHO - cA·Sci_ACD - β·Wide - γ·Large - b) / cp
                sci_pure_rec = (pho[msk] - cA*sci_acd[msk] - βm*wide[msk]
                                - γm*large[msk] - bm) / cp
                sci_rec_m7[msk] = sci_pure_rec + sci_acd[msk]
            resid_m7 = sci_rec_m7 - sci_1s

            # v5t INVERSE: unwrap_v2 + event-balance cap on Large, then Sci_rec
            am = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
            am = np.where(np.isnan(am), 0.0, am); mt = np.maximum(0.0, am-B_THRESHOLD)**2
            d_arr = np.array([np.datetime64(d) for d in df["date"].values])
            ty = (d_arr - t0).astype("timedelta64[D]").astype(float)/365.25
            g_t = 1.0 - beta_v5*ty
            k_t = kc_v5[0] + kc_v5[1]*np.cos(w_v5*ty) + kc_v5[2]*np.sin(w_v5*ty)
            C_v5 = s0_det[detid]*g_t*(1.0 + k_t*mt) + C0_v5
            sci_rec_v5, _ = v5t_unwrap_and_recover(pho, large, wide, sci_1s, lc, dt, C_v5)
            resid_v5 = sci_rec_v5 - sci_1s

            R_m7_all.append(resid_m7); R_v5_all.append(resid_v5)
            if yr in train_years:
                R_m7_train.append(resid_m7); R_v5_train.append(resid_v5)
            else:
                R_m7_test.append(resid_m7); R_v5_test.append(resid_v5)
            # Collect for plot
            SCI_OBS_all.append(sci_1s.astype(np.float32))
            SCI_REC_M7_all.append(sci_rec_m7.astype(np.float32))
            SCI_REC_V5_all.append(sci_rec_v5.astype(np.float32))
        print(f"  {os.path.basename(f)}: done", flush=True)

    r_m7 = np.concatenate(R_m7_all);   r_v5 = np.concatenate(R_v5_all)
    r_m7_tr = np.concatenate(R_m7_train); r_v5_tr = np.concatenate(R_v5_train)
    r_m7_te = np.concatenate(R_m7_test) if R_m7_test else np.array([])
    r_v5_te = np.concatenate(R_v5_test) if R_v5_test else np.array([])

    print("\n" + "="*80)
    print("INVERSE residual on Sci (Sci_rec - Sci_observed)  [filters: PHO>0, Sci>0, L_cyc>1]")
    print("="*80)
    for tag, r in [("M7 all",   r_m7),
                   ("v5t all",  r_v5),
                   ("M7 2017-2019 (train)", r_m7_tr),
                   ("v5t 2017-2019",        r_v5_tr),
                   ("M7 2020-2026 (out-of-sample)", r_m7_te),
                   ("v5t 2020-2026",                r_v5_te)]:
        if len(r) == 0: continue
        m = classic_stats(r)
        print(f"  {tag:>32}  N={m['N']:>10,}  "
              f"median={m['median']:+7.2f}  MAD={m['MAD']:>6.1f}  "
              f"std={m['std']:>6.1f}  RMS={m['RMS']:>6.1f}  "
              f"P5={m['P05']:+7.1f}  P95={m['P95']:+7.1f}  skew={m['skewness']:+5.2f}")

    # ─── Plot: side by side PHO_pred vs PHO_obs density scatter ───
    print("\nPlotting...", flush=True)
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Matches the official plot_v5t_conservation.py settings:
    #   xlim/ylim (30, 1e4) — focused on main + low-Sci fan-out region
    #   sub-sample 800k points
    #   density via 2D log-histogram lookup
    # NOTE: x>0 & y>0 required for log axes (display only,
    # not used in the per-row metric computation above).
    def density_scatter(ax, x, y, title, vmax_n=800000):
        ok_disp = (x > 0) & (y > 0)
        n_dropped_x = int((x <= 0).sum()); n_dropped_y = int((y <= 0).sum())
        x = x[ok_disp]; y = y[ok_disp]
        if len(x) > vmax_n:
            idx = np.random.RandomState(0).choice(len(x), vmax_n, replace=False)
            x = x[idx]; y = y[idx]
        H, xe, ye = np.histogram2d(np.log10(np.maximum(x,1)), np.log10(np.maximum(y,1)), bins=200)
        ix = np.clip(np.searchsorted(xe, np.log10(np.maximum(x,1)))-1, 0, H.shape[0]-1)
        iy = np.clip(np.searchsorted(ye, np.log10(np.maximum(y,1)))-1, 0, H.shape[1]-1)
        dens = H[ix, iy]
        order = np.argsort(dens)
        sc = ax.scatter(x[order], y[order], c=dens[order], s=1.5, cmap='viridis',
                        edgecolors='none',
                        norm=mpl.colors.LogNorm(vmin=max(dens.min(),1), vmax=dens.max()),
                        rasterized=True)
        ax.plot([1, 1e6], [1, 1e6], 'r-', lw=1.2, label='y = x')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(30, 1e4); ax.set_ylim(30, 1e4)   # match official conservation plot
        ax.set_title(title + f"\n(x>0 & y>0 — drops {n_dropped_x} x≤0, {n_dropped_y} y≤0)",
                     fontsize=10)
        ax.grid(alpha=0.3, which='both')
        return sc

    # Use FULL-mission Sci data collected in PASS 2 (concatenated already in metrics)
    print("  concatenating full-mission Sci arrays for scatter plot...", flush=True)
    sci_obs_full = np.concatenate(SCI_OBS_all)
    sci_rec_m7_p = np.concatenate(SCI_REC_M7_all)
    sci_rec_v5_p = np.concatenate(SCI_REC_V5_all)
    print(f"  full-mission Sci arrays: {len(sci_obs_full):,} rows", flush=True)

    m_m7 = classic_stats(sci_rec_m7_p - sci_obs_full)
    m_v5 = classic_stats(sci_rec_v5_p - sci_obs_full)

    density_scatter(axes[0,0], sci_obs_full, sci_rec_m7_p,
                    f"M7 (90 params; uses observed Sci_ACD)\n"
                    f"median={m_m7['median']:+.1f}, MAD={m_m7['MAD']:.1f}, RMS={m_m7['RMS']:.1f}")
    density_scatter(axes[0,1], sci_obs_full, sci_rec_v5_p,
                    f"v5t (23 params; blind reconstruction)\n"
                    f"median={m_v5['median']:+.1f}, MAD={m_v5['MAD']:.1f}, RMS={m_v5['RMS']:.1f}")
    axes[0,0].set_xlabel(r'Sci$_{\rm obs}$ (cnt/s)'); axes[0,0].set_ylabel(r'Sci$_{\rm rec}$ (cnt/s)')
    axes[0,1].set_xlabel(r'Sci$_{\rm obs}$ (cnt/s)'); axes[0,1].set_ylabel(r'Sci$_{\rm rec}$ (cnt/s)')

    bins = np.linspace(-500, 500, 200)
    axes[1,0].hist(sci_rec_m7_p - sci_obs_full, bins=bins, alpha=0.6, color='r', label='M7 90 params')
    axes[1,0].hist(sci_rec_v5_p - sci_obs_full, bins=bins, alpha=0.6, color='b', label='v5t 23 params')
    axes[1,0].set_xlabel(r'Sci residual = Sci$_{\rm rec}$ - Sci$_{\rm obs}$ (cnt/s)')
    axes[1,0].set_ylabel('count')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    axes[1,0].set_title(f'Sci residual distribution (FULL mission, {len(sci_obs_full)/1e6:.1f}M rows)')
    axes[1,0].set_yscale('log')

    # Side-by-side metrics table
    axes[1,1].axis('off')
    metric_text = (
        f"$\\bf{{Full-mission\\ metrics\\ (forward\\ PHO\\ residual)}}$\n\n"
        f"{'':>10}  {'median':>9}  {'MAD':>7}  {'std':>7}  {'RMS':>7}  {'skew':>5}\n"
        f"{'M7 all':>10}  {classic_stats(r_m7)['median']:+7.2f}  {classic_stats(r_m7)['MAD']:>7.1f}  "
        f"{classic_stats(r_m7)['std']:>7.1f}  {classic_stats(r_m7)['RMS']:>7.1f}  {classic_stats(r_m7)['skewness']:>+5.2f}\n"
        f"{'v5t all':>10}  {classic_stats(r_v5)['median']:+7.2f}  {classic_stats(r_v5)['MAD']:>7.1f}  "
        f"{classic_stats(r_v5)['std']:>7.1f}  {classic_stats(r_v5)['RMS']:>7.1f}  {classic_stats(r_v5)['skewness']:>+5.2f}\n\n"
    )
    if len(r_m7_tr)>0 and len(r_v5_tr)>0:
        metric_text += (
            f"$\\bf{{2017-2019\\ (M7\\ training\\ region)}}$\n"
            f"{'M7':>10}  {classic_stats(r_m7_tr)['median']:+7.2f}  {classic_stats(r_m7_tr)['MAD']:>7.1f}  "
            f"{classic_stats(r_m7_tr)['std']:>7.1f}  {classic_stats(r_m7_tr)['RMS']:>7.1f}  {classic_stats(r_m7_tr)['skewness']:>+5.2f}\n"
            f"{'v5t':>10}  {classic_stats(r_v5_tr)['median']:+7.2f}  {classic_stats(r_v5_tr)['MAD']:>7.1f}  "
            f"{classic_stats(r_v5_tr)['std']:>7.1f}  {classic_stats(r_v5_tr)['RMS']:>7.1f}  {classic_stats(r_v5_tr)['skewness']:>+5.2f}\n\n"
        )
    if len(r_m7_te)>0 and len(r_v5_te)>0:
        metric_text += (
            f"$\\bf{{2020-2026\\ (out-of-sample\\ for\\ M7)}}$\n"
            f"{'M7':>10}  {classic_stats(r_m7_te)['median']:+7.2f}  {classic_stats(r_m7_te)['MAD']:>7.1f}  "
            f"{classic_stats(r_m7_te)['std']:>7.1f}  {classic_stats(r_m7_te)['RMS']:>7.1f}  {classic_stats(r_m7_te)['skewness']:>+5.2f}\n"
            f"{'v5t':>10}  {classic_stats(r_v5_te)['median']:+7.2f}  {classic_stats(r_v5_te)['MAD']:>7.1f}  "
            f"{classic_stats(r_v5_te)['std']:>7.1f}  {classic_stats(r_v5_te)['RMS']:>7.1f}  {classic_stats(r_v5_te)['skewness']:>+5.2f}\n"
        )
    axes[1,1].text(0.02, 0.95, metric_text, family='monospace', va='top', fontsize=10,
                   transform=axes[1,1].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8'))

    fig.suptitle("M7 (90 params, 2017-19 trained) vs v5t (23 params, full mission)\n"
                 r"INVERSE: reconstruct Sci from (PHO, Wide, Large [, Sci_ACD for M7])"
                 "\nSanity filters: PHO>0, Sci>0, L_cyc>1 (no physics filter)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/compare_m7_vs_v5t_inverse.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
