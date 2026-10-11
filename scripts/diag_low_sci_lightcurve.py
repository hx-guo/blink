#!/usr/bin/env python3
"""Find a continuous time-window where Sci_obs is low (<60 cnt/s) and v5t
residual is anomalously large, then plot multi-panel light curve.

Goal: understand what's happening in the leftmost region of the
Sci_rec vs Sci_obs plot where residuals deviate strongly.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","met_sec","PHO","Wide","Large","Sci_1s",
          "L_cycles","Dt","HV","Lat","Lon","ACD_sum","PM_0","PM_1","PM_2"]
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


def compute_v5t_C_and_resid(df, s0_det, beta_v5, t0, w_v5, kc_v5, C0_v5, interp):
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
    C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0_v5
    base = apply_pipeline(pho,lg,wd,sci,lc,dtv,C_v5)
    sci_rec = base - C_v5
    resid = base - sci - C_v5
    return C_v5, sci_rec, resid, am


def main():
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det_v5=cz["s0_det"]; beta_v5=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w_v5=float(cz["w"]); kc_v5=cz["k_coeffs"]; C0_v5=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)

    # === Step 1: scan 2020 cache, find candidate window ===
    print("Scanning 2020 cache for low-Sci high-resid seconds...", flush=True)
    cache_file = os.path.join(CACHE, "clean_relaxed_2020.parquet")
    pf = pq.ParquetFile(cache_file)

    # Search a couple of row groups for candidates
    # Look at row groups across the year
    candidates = []
    for rg in np.unique(np.linspace(0, pf.num_row_groups-1, 8).astype(int)):
        df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
        C_v5, sci_rec, resid, am = compute_v5t_C_and_resid(
            df, s0_det_v5, beta_v5, t0, w_v5, kc_v5, C0_v5, interp)
        sci = df["Sci_1s"].astype(float).values
        # Find seconds where Sci low + residual large
        mask_low = (sci > 5) & (sci < 60) & (np.abs(resid) > 50) & np.isfinite(resid)
        if mask_low.sum() == 0:
            continue
        # Pick a met_sec range with many such low-Sci high-resid points clustered together
        met_secs = df["met_sec"].values[mask_low]
        dates = df["date"].values[mask_low]
        # Find dense clusters: pick most common date among candidates
        from collections import Counter
        date_count = Counter(dates)
        top_date_str, top_count = date_count.most_common(1)[0]
        candidates.append((top_date_str, top_count, rg))
        print(f"  rg {rg}: top date {top_date_str} with {top_count} low-Sci high-resid seconds",
              flush=True)

    if not candidates:
        print("No candidates found"); return

    # Pick the candidate with most concentrated low-Sci high-resid points
    candidates.sort(key=lambda c: c[1], reverse=True)
    target_date, count, _ = candidates[0]
    print(f"\nPicked target date: {target_date} ({count} low-Sci high-resid seconds)")

    # === Step 2: load all rows for that day ===
    print(f"\nReading all 2020 rows for {target_date}...", flush=True)
    chunks=[]
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[b["date"]==target_date]
        if len(b)>0: chunks.append(b)
    df_day = __import__("pandas").concat(chunks, ignore_index=True) if chunks else None
    if df_day is None or len(df_day)==0:
        print("No data for target date"); return
    print(f"  loaded {len(df_day):,} rows for {target_date}", flush=True)

    C_v5, sci_rec, resid, am = compute_v5t_C_and_resid(
        df_day, s0_det_v5, beta_v5, t0, w_v5, kc_v5, C0_v5, interp)
    df_day["C_v5"] = C_v5
    df_day["sci_rec"] = sci_rec
    df_day["resid"] = resid
    df_day["mlat"] = am

    # Compute unwrapped Large for the diagnostic
    pho_d = df_day["PHO"].astype(float).values
    lg_d = df_day["Large"].astype(float).values
    wd_d = df_day["Wide"].astype(float).values
    sci_d = df_day["Sci_1s"].astype(float).values
    lc_d = df_day["L_cycles"].astype(float).values
    dt_d = df_day["Dt"].astype(float).values
    # First pass: unwrap with v5t C
    lv2 = unwrap_v2(pho_d, lg_d, wd_d, sci_d, lc_d, dt_d, C_v5)
    # Second pass: event-balance cap
    LL_d = lc_d*L; lf_d = 1.0 - dt_d/lc_d
    mle = pho_d - ((sci_d+MIN_C_SLACK)*LL_d + wd_d) / lf_d
    n3 = np.round((lv2 - lg_d)/1024).astype(int)
    nmax = np.maximum(np.floor((mle - lg_d)/1024.).astype(int), 0)
    lv5 = lg_d + np.where(n3 > nmax, nmax, n3)*1024.
    df_day["Large_unwrapped"] = lv5
    df_day["N_wraps"] = np.where(n3 > nmax, nmax, n3)  # number of 1024 added

    sci = df_day["Sci_1s"].astype(float).values
    mask_low = (sci > 5) & (sci < 60) & (np.abs(resid) > 50)
    # Pick window: ~1 hour centered on the densest part
    bad_met = df_day["met_sec"].values[mask_low]
    if len(bad_met) == 0:
        print("No bad seconds in target day"); return
    # Find densest 1-hour window
    bad_met_sorted = np.sort(bad_met)
    # Sliding window: count bad in each 3600s window
    best_count = 0; best_center = bad_met_sorted[0]
    for t in bad_met_sorted[::200]:
        c = ((bad_met_sorted >= t-1800) & (bad_met_sorted <= t+1800)).sum()
        if c > best_count:
            best_count = c; best_center = t
    # Align window to the next-lower 10-minute boundary
    t_lo = (best_center // 600) * 600 - 1800
    t_hi = t_lo + 3600
    print(f"\nWindow: met_sec [{t_lo} – {t_hi}] (1 hour, aligned to 10-min boundary)")
    print(f"  bad seconds in window: {best_count}")

    # === Step 3: collect window data, group by met_sec ===
    win = df_day[(df_day["met_sec"]>=t_lo)&(df_day["met_sec"]<=t_hi)].copy()
    print(f"  {len(win):,} rows in window")

    # Per-second aggregation: average over 18 dets for PHO/Wide/Large/Sci/Dt/L_cycles/HV/resid
    grp = win.groupby("met_sec")
    agg = {
        "PHO_sum": grp["PHO"].sum(),     # 18-detector sum
        "Wide_sum": grp["Wide"].sum(),
        "Large_raw_sum": grp["Large"].sum(),
        "Large_unwrap_sum": grp["Large_unwrapped"].sum(),
        "N_wraps_sum": grp["N_wraps"].sum(),
        "Sci_sum": grp["Sci_1s"].sum(),
        "Sci_rec_sum": grp["sci_rec"].sum(),
        "C_v5_sum": grp["C_v5"].sum(),
        "resid_sum": grp["resid"].sum(),
        "HV_avg": grp["HV"].mean(),
        "L_cycles_avg": grp["L_cycles"].mean(),
        "Dt_avg": grp["Dt"].mean(),
        "ACD_sum": grp["ACD_sum"].first(),  # already per-second
        "PM_0": grp["PM_0"].first(),
        "PM_1": grp["PM_1"].first(),
        "PM_2": grp["PM_2"].first(),
        "Lat": grp["Lat"].first(),
        "Lon": grp["Lon"].first(),
        "mlat": grp["mlat"].first(),
    }
    agg_df = __import__("pandas").DataFrame(agg).reset_index().sort_values("met_sec")
    print(f"  {len(agg_df)} unique met_sec in window")

    # === Step 4: Plot multi-panel light curve ===
    met_sec = agg_df["met_sec"].values
    # Convert met_sec to UTC for x-axis. MET epoch = 2012-01-01T00:00 UTC.
    epoch = np.datetime64("2012-01-01T00:00:00")
    time_arr = epoch + (met_sec * np.timedelta64(1, "s"))

    fig, axes = plt.subplots(9, 1, figsize=(14, 24), sharex=True)
    fig.suptitle(
        f"Low-Sci high-residual region light curve — {target_date} (UTC), 1-hour window aligned to 10-min boundary\n"
        f"Sum / average over 18 detectors (3 box × 6 det)",
        fontsize=12, fontweight='bold')

    # Panel 1: PHO / Wide / Large_raw
    ax = axes[0]
    ax.plot(time_arr, agg_df["PHO_sum"], 'b-', lw=0.8, label="PHO (18-det sum)")
    ax.plot(time_arr, agg_df["Wide_sum"], 'r-', lw=0.6, label="Wide (CsI)")
    ax.plot(time_arr, agg_df["Large_raw_sum"], 'g-', lw=0.6, label="Large RAW (0–1023 wrap)")
    ax.set_ylabel("counts / frame", fontsize=11)
    ax.set_yscale("log"); ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Engineering counters (1B HE_Eng)", fontsize=10)

    # New Panel 2: Large raw vs unwrapped + N_wraps
    ax = axes[1]
    ax.plot(time_arr, agg_df["Large_raw_sum"], 'g-', lw=0.6, label="Large RAW (cache, 0–1023 wrap)")
    ax.plot(time_arr, agg_df["Large_unwrap_sum"], 'm-', lw=0.9, label="Large UNWRAPPED (after unwrap_v2 + cap)")
    ax2 = ax.twinx()
    ax2.plot(time_arr, agg_df["N_wraps_sum"], 'k.', ms=2, label="Σ N_wraps (18-det)")
    ax.set_ylabel("Large counts (18-det sum)", fontsize=11)
    ax.set_yscale("symlog", linthresh=100)
    ax2.set_ylabel("Σ N_wraps (1024 multiples added)", fontsize=11, color='k')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Large wrap diagnostic: raw vs unwrapped + N_wraps", fontsize=10)

    # Panel 3: Sci_obs vs Sci_rec (the variables of interest)
    ax = axes[2]
    ax.plot(time_arr, agg_df["Sci_sum"], 'k-', lw=0.8, label=r"Sci$_{\rm obs}$ (1K HE-Evt 1s window)")
    ax.plot(time_arr, agg_df["Sci_rec_sum"], 'm-', lw=0.8, label=r"Sci$_{\rm rec}$ (v5t conservation)")
    ax.plot(time_arr, agg_df["C_v5_sum"], 'c--', lw=0.7, alpha=0.7, label=r"$C$ (v5t baseline)")
    ax.set_ylabel("counts / s", fontsize=11)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title("Sci observed vs Sci reconstructed (v5t)", fontsize=10)

    # Panel 4: residual
    ax = axes[3]
    ax.plot(time_arr, agg_df["resid_sum"], 'b-', lw=0.8, label=r"residual = Sci$_{\rm rec}$ - Sci$_{\rm obs}$ (18-det sum)")
    ax.axhline(0, color='r', lw=1)
    ax.set_ylabel("residual (cnt/s)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("v5t residual", fontsize=10)

    # Panel 5: livetime indicators
    ax = axes[4]
    L_cyc_mean = agg_df["L_cycles_avg"].values
    ax.plot(time_arr, L_cyc_mean, 'b-', lw=0.8, label=r"$L_{\rm cyc}$ avg (16$\mu$s)")
    ax2 = ax.twinx()
    dt_frac = agg_df["Dt_avg"].values / np.maximum(L_cyc_mean, 1)
    ax2.plot(time_arr, dt_frac*100, 'r-', lw=0.8, label="dead time %")
    ax.set_ylabel(r"$L_{\rm cyc}$ (16$\mu$s units)", fontsize=11, color='b')
    ax2.set_ylabel("dead time fraction (%)", fontsize=11, color='r')
    ax.grid(alpha=0.3)
    ax.set_title(r"Live time: $L_{\rm cyc}$ and dead-time fraction", fontsize=10)

    # Panel 6: HV
    ax = axes[5]
    ax.plot(time_arr, agg_df["HV_avg"], 'k-', lw=0.8, label="HV avg (18 PHODets)")
    ax.axhline(-1100, color='r', ls='--', alpha=0.5, label='Stage 1 bounds')
    ax.axhline(-900, color='r', ls='--', alpha=0.5)
    ax.set_ylabel("HV (V)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("PMT high voltage", fontsize=10)

    # Panel 7: ACD, PM
    ax = axes[6]
    ax.plot(time_arr, agg_df["ACD_sum"], 'b-', lw=0.8, label="ACD_sum")
    ax2 = ax.twinx()
    ax2.plot(time_arr, agg_df["PM_0"], 'r-', lw=0.6, alpha=0.7, label="PM_0")
    ax2.plot(time_arr, agg_df["PM_1"], 'g-', lw=0.6, alpha=0.7, label="PM_1")
    ax2.plot(time_arr, agg_df["PM_2"], 'm-', lw=0.6, alpha=0.7, label="PM_2")
    ax.set_ylabel("ACD_sum (cnt/s)", fontsize=11, color='b')
    ax.set_yscale("log")
    ax2.set_ylabel("PM_0/1/2 (cnt/s)", fontsize=11, color='r')
    ax2.set_yscale("symlog", linthresh=1)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("In-situ particle counters", fontsize=10)

    # Panel 8: Geomagnetic lat (mlat)
    ax = axes[7]
    ax.plot(time_arr, agg_df["mlat"], 'k-', lw=0.8, label="|mlat| (AACGM v2)")
    ax.axhline(20, color='r', ls='--', alpha=0.5, label='B threshold (20°)')
    ax.set_ylabel("|mlat| (deg)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Geomagnetic latitude (drives v5t mlat² term)", fontsize=10)

    # Panel 9: Lat, Lon (orbital position)
    ax = axes[8]
    ax.plot(time_arr, agg_df["Lat"], 'b-', lw=0.8, label="Lat")
    ax2 = ax.twinx()
    ax2.plot(time_arr, agg_df["Lon"], 'r-', lw=0.6, label="Lon")
    ax.set_ylabel("Lat (deg)", fontsize=11, color='b')
    ax2.set_ylabel("Lon (deg, [0,360))", fontsize=11, color='r')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Orbital position", fontsize=10)
    ax.set_xlabel("UTC time", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_low_sci_lightcurve.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")

    # Also print summary numbers for the window
    print(f"\n=== Window summary ===")
    print(f"  date: {target_date}, met_sec range: {t_lo} – {t_hi}")
    print(f"  Sci_sum (18-det) median = {np.median(agg_df['Sci_sum']):.1f}, range [{agg_df['Sci_sum'].min():.0f}, {agg_df['Sci_sum'].max():.0f}]")
    print(f"  PHO_sum median = {np.median(agg_df['PHO_sum']):.0f}, range [{agg_df['PHO_sum'].min():.0f}, {agg_df['PHO_sum'].max():.0f}]")
    print(f"  Wide_sum median = {np.median(agg_df['Wide_sum']):.0f}")
    print(f"  residual_sum median = {np.median(agg_df['resid_sum']):.1f}, |max| = {np.abs(agg_df['resid_sum']).max():.0f}")
    print(f"  ACD_sum median = {np.median(agg_df['ACD_sum']):.0f}")
    print(f"  HV_avg range = [{agg_df['HV_avg'].min():.1f}, {agg_df['HV_avg'].max():.1f}]")
    print(f"  |mlat| range = [{agg_df['mlat'].min():.1f}, {agg_df['mlat'].max():.1f}]")
    print(f"  Lat range = [{agg_df['Lat'].min():.1f}, {agg_df['Lat'].max():.1f}]")
    print(f"  Lon range = [{agg_df['Lon'].min():.1f}, {agg_df['Lon'].max():.1f}]")


if __name__ == "__main__":
    main()
