#!/usr/bin/env python3
"""Find a SAA in/out window in a normal day, plot before/after segments together.

Strategy:
  1. Pick a 2019 day (well before SGR FRB 200428).
  2. Load entire day's cleaned cache.
  3. Find the largest met_sec gap > 300 s (SAA period, HV filtered by Stage 1).
  4. Plot ±30 min around gap, so we see "entering SAA" + gap + "leaving SAA".
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","met_sec","PHO","Wide","Large","Sci_1s",
          "L_cycles","Dt","HV","Lat","Lon","ACD_sum"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_2019.parquet"
TARGET_DATE = "2019-08-15"   # arbitrary 2019 day, well before SGR FRB 200428


def main():
    pf = pq.ParquetFile(CACHE)
    print(f"Streaming for {TARGET_DATE}...", flush=True)
    chunks = []
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[b["date"]==TARGET_DATE]
        if len(b)>0: chunks.append(b)
    import pandas as pd
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    if df is None or len(df)==0:
        print(f"No data for {TARGET_DATE}"); return
    print(f"  loaded {len(df):,} rows", flush=True)
    df = df.sort_values("met_sec").reset_index(drop=True)

    # Find SAA gaps: consecutive met_sec jumps > 300s
    unique_met = np.sort(df["met_sec"].unique())
    diffs = np.diff(unique_met)
    big_gaps = np.where(diffs > 300)[0]
    if len(big_gaps) == 0:
        print("No big gaps (>300s) found in day"); return
    print(f"Found {len(big_gaps)} gaps >300s. Sizes (seconds):")
    for i, idx in enumerate(big_gaps[:5]):
        print(f"  gap {i}: met_sec {unique_met[idx]} → {unique_met[idx+1]}  "
              f"({diffs[idx]:.0f}s, {diffs[idx]/60:.1f}min)")

    # Pick the LARGEST gap (likely longest SAA passage)
    largest_idx = big_gaps[np.argmax(diffs[big_gaps])]
    gap_start = int(unique_met[largest_idx])
    gap_end = int(unique_met[largest_idx+1])
    print(f"\nPicked largest gap: {gap_start} → {gap_end}  ({(gap_end-gap_start)/60:.1f}min)")

    # Window: 30 min before gap_start, 30 min after gap_end
    win_lo = gap_start - 1800
    win_hi = gap_end + 1800
    print(f"Plot window: met_sec [{win_lo}, {win_hi}]  ({(win_hi-win_lo)/60:.0f} min total)")

    win = df[(df["met_sec"]>=win_lo)&(df["met_sec"]<=win_hi)].copy()
    print(f"  {len(win):,} rows in plot window", flush=True)

    # Compute v5t C / sci_rec
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)

    am=np.abs(interp(np.column_stack([win["Lat"].values,win["Lon"].values])))
    am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-20.0)**2
    d_arr=np.array([np.datetime64(d) for d in win["date"].values])
    ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
    g_t=1.0-beta*ty
    k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
    box_idx=np.select([win["box"].values==b for b in "ABC"],[0,1,2],default=0)
    detid=box_idx*6+win["det"].values
    C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0
    win["C_v5"]=C_v5

    def unwrap_v2(pho, large, wide, sci, lc, dt, C):
        LL=lc*L; lf=1.0-dt/lc
        pred=pho-(wide+(sci+C)*LL)/lf
        n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
        mx=pho-wide; out=large+n*1024.; ov=out>mx
        if ov.any():
            nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0)
            out=large+np.where(ov,nm,n)*1024.
        return out

    pho=win["PHO"].astype(float).values; lg=win["Large"].astype(float).values
    wd=win["Wide"].astype(float).values; sci=win["Sci_1s"].astype(float).values
    lc=win["L_cycles"].astype(float).values; dtv=win["Dt"].astype(float).values
    LL=lc*L; lf=1.0-dtv/lc
    lv2=unwrap_v2(pho,lg,wd,sci,lc,dtv,C_v5)
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
    n3=np.round((lv2-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
    lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
    base=(pho-lv5)*lf/LL-wd/LL
    win["sci_rec"]=base-C_v5
    win["resid"]=win["sci_rec"]-win["Sci_1s"]

    # Aggregate per (box, met_sec)
    grp = win.groupby(["box","met_sec"]).agg(
        PHO_sum=("PHO","sum"),
        Sci_sum=("Sci_1s","sum"),
        Sci_rec_sum=("sci_rec","sum"),
        resid_sum=("resid","sum"),
        HV_avg=("HV","mean"),
    ).reset_index()
    grp["sci_over_pho"] = grp["Sci_sum"] / np.maximum(grp["PHO_sum"], 1)

    print(f"\n=== Per-box summary for the entire plot window ===")
    for b in "ABC":
        sub = grp[grp["box"]==b]
        print(f"  Box {b}  N_sec={len(sub):>5}  "
              f"PHO_med={sub['PHO_sum'].median():>6.0f}  "
              f"Sci_med={sub['Sci_sum'].median():>6.0f}  "
              f"Sci/PHO={sub['sci_over_pho'].median():.3f}  "
              f"HV_med={sub['HV_avg'].median():.1f}  "
              f"resid_med={sub['resid_sum'].median():+.1f}")

    # Trim transition seconds at gap boundaries (5 s on each side)
    TRIM = 5
    grp = grp[
        ((grp["met_sec"] < gap_start - TRIM) | (grp["met_sec"] > gap_end + TRIM))
    ].copy()

    def break_at_gaps(t, y, met, threshold_s=60):
        """Insert NaN where consecutive met_sec jump > threshold so matplotlib
        does NOT connect across the gap."""
        diffs = np.diff(met)
        gap_pos = np.where(diffs > threshold_s)[0]
        if len(gap_pos) == 0:
            return t, y
        out_t = []; out_y = []
        prev = 0
        for gp in gap_pos:
            out_t.extend(t[prev:gp+1]); out_y.extend(y[prev:gp+1])
            # Use mid-gap time for the NaN marker
            mid_t = t[gp] + (t[gp+1] - t[gp]) / 2
            out_t.append(mid_t); out_y.append(np.nan)
            prev = gp + 1
        out_t.extend(t[prev:]); out_y.extend(y[prev:])
        return np.array(out_t), np.array(out_y)

    # Plot
    mpl.rcParams.update({"font.family":"DejaVu Sans"})
    epoch = np.datetime64("2012-01-01T00:00:00")
    t_gap_start = epoch + gap_start * np.timedelta64(1,"s")
    t_gap_end   = epoch + gap_end   * np.timedelta64(1,"s")

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"SAA in/out comparison — {TARGET_DATE}, ±30 min around the longest SAA gap\n"
        f"Gap (cache empty): {(gap_end-gap_start)/60:.0f} min;  "
        f"shaded grey = SAA passage (Stage 1 filtered out)",
        fontsize=12, fontweight='bold')
    colors = {'A':'blue','B':'green','C':'red'}

    for ax in axes:
        ax.axvspan(t_gap_start, t_gap_end, alpha=0.18, color='gray')

    ax=axes[0]
    for b in "ABC":
        sub = grp[grp["box"]==b].sort_values("met_sec")
        t_raw = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        t, y = break_at_gaps(t_raw, sub["PHO_sum"].values, sub["met_sec"].values)
        ax.plot(t, y, lw=0.6, color=colors[b], label=f"Box {b}")
    ax.set_ylabel("PHO 6-det sum (cnt/frame)", fontsize=11)
    ax.set_yscale("symlog", linthresh=100)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("PHO trigger count (1B engineering counter)", fontsize=11)

    ax=axes[1]
    for b in "ABC":
        sub = grp[grp["box"]==b].sort_values("met_sec")
        t_raw = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        t, y = break_at_gaps(t_raw, sub["Sci_sum"].values, sub["met_sec"].values)
        ax.plot(t, y, lw=0.6, color=colors[b], label=f"Box {b}")
    ax.set_ylabel(r"Sci$_{\rm obs}$ 6-det sum (cnt/s)", fontsize=11)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_title(r"Sci$_{\rm obs}$ (1K HE-Evt 1s window)", fontsize=11)

    ax=axes[2]
    for b in "ABC":
        sub = grp[grp["box"]==b].sort_values("met_sec")
        t_raw = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        t, y = break_at_gaps(t_raw, sub["sci_over_pho"].values, sub["met_sec"].values)
        ax.plot(t, y, lw=0.6, color=colors[b], label=f"Box {b}")
    ax.axhline(0.5, color='k', ls='--', alpha=0.5, label="0.5 (normal)")
    ax.set_ylabel(r"Sci$_{\rm obs}$ / PHO ratio", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Sci/PHO ratio — looking for transient throttle vs normal", fontsize=11)

    ax=axes[3]
    for b in "ABC":
        sub = grp[grp["box"]==b].sort_values("met_sec")
        t_raw = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        t, y = break_at_gaps(t_raw, sub["resid_sum"].values, sub["met_sec"].values)
        ax.plot(t, y, lw=0.6, color=colors[b], label=f"Box {b}")
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel("v5t residual (cnt/s)", fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title(r"residual = Sci$_{\rm rec}$ − Sci$_{\rm obs}$ (per box, 6-det sum)", fontsize=11)

    ax=axes[4]
    for b in "ABC":
        sub = grp[grp["box"]==b].sort_values("met_sec")
        t_raw = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        t, y = break_at_gaps(t_raw, sub["HV_avg"].values, sub["met_sec"].values)
        ax.plot(t, y, lw=0.6, color=colors[b], label=f"Box {b}")
    ax.set_ylabel("HV (V)", fontsize=11)
    ax.set_ylim(-1010, -960)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title("HV (working voltage during in-cache seconds)", fontsize=11)
    ax.set_xlabel("UTC time", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_saa_inout.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
