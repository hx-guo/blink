#!/usr/bin/env python3
"""Zoom-in light curve of the 2020-05-25 spike region.

Single box (Box A) with 6 detectors plotted separately. Focus on 09:17-09:37
where the spike concentration is highest. Includes:
  - PHO, Wide, Large RAW per det
  - Large UNWRAPPED + N_wraps per det
  - Sci_obs vs Sci_rec
  - residual
  - Sci/PHO ratio (the data-quality indicator)
  - ACD_sum, PM (satellite-wide)
  - |mlat|, Lat/Lon
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
          "L_cycles","Dt","HV","Lat","Lon","ACD_sum","PM_0","PM_1","PM_2"]
CACHE_FILE = "/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_2020.parquet"

# Window: 2020-05-25 09:17 to 09:37 (20 min, centered on the spike region)
# MET = seconds since 2012-01-01 UTC.
# 2020-05-25 09:17 UTC = ~265023420
MET_START = 265023420
MET_END   = 265024620  # 09:37


def unwrap_v2_with_n(pho, large, wide, sci, lc, dt, C):
    """Return (Large_unwrapped, N_wraps_final) — N_wraps is post-cap, i.e. actual
    multiples of 1024 added to raw Large."""
    LL = lc * L; lf = 1.0 - dt/lc
    pred = pho - (wide + (sci+C)*LL)/lf
    n1 = np.maximum(np.round((pred-large)/1024.).astype(int), 0)
    mx = pho - wide
    out1 = large + n1*1024.
    ov = out1 > mx
    nm = np.maximum(np.floor((mx-large)/1024.).astype(int), 0)
    n_after_cap1 = np.where(ov, nm, n1)
    out1 = large + n_after_cap1*1024.
    mle = pho - ((sci+MIN_C_SLACK)*LL + wide)/lf
    n3 = np.round((out1-large)/1024).astype(int)
    nmax = np.maximum(np.floor((mle-large)/1024.).astype(int), 0)
    n_final = np.where(n3 > nmax, nmax, n3)
    return large + n_final*1024., n_final


def main():
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)

    print(f"Loading 2020-05-25 window {MET_START}-{MET_END}...", flush=True)
    pf = pq.ParquetFile(CACHE_FILE)
    chunks=[]
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[(b["date"]=="2020-05-25")
              &(b["met_sec"]>=MET_START)&(b["met_sec"]<=MET_END)
              &(b["box"]=="A")]
        if len(b)>0: chunks.append(b)
    import pandas as pd
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    print(f"  loaded {len(df):,} rows (Box A only)", flush=True)
    df = df.sort_values(["det","met_sec"]).reset_index(drop=True)

    # Compute v5t C + unwrap + residual per row
    am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
    am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-20.0)**2
    d_arr=np.array([np.datetime64(d) for d in df["date"].values])
    ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
    g_t=1.0-beta*ty
    k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
    detid=df["det"].values  # Box A so detid = det (0-5)
    box_idx = 0  # A
    detid_global = box_idx*6 + detid
    C_v5 = s0_det[detid_global]*g_t*(1.0+k_t*mt) + C0

    pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
    wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
    lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values

    lv5, n_wraps = unwrap_v2_with_n(pho, lg, wd, sci, lc, dtv, C_v5)
    LL=lc*L; lf=1.0-dtv/lc
    base=(pho-lv5)*lf/LL - wd/LL
    sci_rec = base - C_v5
    resid = sci_rec - sci

    df["C_v5"]=C_v5; df["Large_unwrap"]=lv5; df["N_wraps"]=n_wraps
    df["sci_rec"]=sci_rec; df["resid"]=resid; df["mlat"]=am
    df["sci_over_pho"] = sci/np.maximum(pho,1)

    # MET → UTC time
    epoch = np.datetime64("2012-01-01T00:00:00")
    df["t_utc"] = epoch + df["met_sec"].values * np.timedelta64(1,"s")

    print(f"  Sci/PHO median: {np.median(df['sci_over_pho']):.2f}, "
          f"min: {df['sci_over_pho'].min():.3f}, max: {df['sci_over_pho'].max():.2f}")

    # ─── Plot ───
    mpl.rcParams.update({"font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(8, 1, figsize=(14, 24), sharex=True)
    fig.suptitle(
        f"Box A spike zoom — 2020-05-25, 09:17–09:37 UTC, 6 detectors plotted separately\n"
        f"(spike region: residual goes negative when Sci_obs collapses while PHO stays high)",
        fontsize=12, fontweight='bold')

    det_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))

    # Panel 1: PHO per det
    ax = axes[0]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["PHO"], lw=0.6, color=det_colors[d], label=f"A{d}")
    ax.set_ylabel("PHO (cnt/frame)", fontsize=11)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(loc='upper right', fontsize=9, ncol=6)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("PHO trigger count per detector", fontsize=10)

    # Panel 2: Large RAW per det (with wrap markers — visible as drops back to ~0)
    ax = axes[1]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["Large"], lw=0.6, color=det_colors[d], label=f"A{d}")
    ax.axhline(1024, color='k', ls='--', alpha=0.4, label="1024 (wrap boundary)")
    ax.set_ylabel("Large RAW (cnt, 0–1023)", fontsize=11)
    ax.set_ylim(-50, 1200)
    ax.legend(loc='upper right', fontsize=9, ncol=7)
    ax.grid(alpha=0.3)
    ax.set_title("Large RAW per detector — every drop near 0 means a wrap happened", fontsize=10)

    # Panel 3: Large UNWRAPPED per det
    ax = axes[2]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["Large_unwrap"], lw=0.6, color=det_colors[d], label=f"A{d}")
    ax.set_ylabel("Large UNWRAPPED (cnt/frame)", fontsize=11)
    ax.set_yscale("symlog", linthresh=100)
    ax.legend(loc='upper right', fontsize=9, ncol=6)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Large UNWRAPPED per detector (after unwrap_v2 + event-balance cap)", fontsize=10)

    # Panel 4: N_wraps per det
    ax = axes[3]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["N_wraps"], '.-', lw=0.4, ms=1.5,
                color=det_colors[d], label=f"A{d}")
    ax.set_ylabel("N_wraps (1024 multiples added)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=6)
    ax.grid(alpha=0.3)
    ax.set_title("N_wraps per detector — how many 1024 cycles unwrap_v2 estimated", fontsize=10)

    # Panel 5: Sci/PHO ratio per det (key data-quality indicator)
    ax = axes[4]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["sci_over_pho"], lw=0.6, color=det_colors[d], label=f"A{d}")
    ax.axhline(0.5, color='gray', ls=':', alpha=0.6, label="0.5 (normal, ~50% pass ACD)")
    ax.set_ylabel("Sci$_{obs}$ / PHO", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 10)
    ax.legend(loc='upper right', fontsize=9, ncol=7)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Sci/PHO ratio — drops to ~0 = HE-Evt downlink lost events for that second", fontsize=10)

    # Panel 6: Sci_obs vs Sci_rec per det
    ax = axes[5]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["Sci_1s"], lw=0.6, color=det_colors[d],
                label=f"A{d} Sci$_{{obs}}$", alpha=0.85)
        ax.plot(sub["t_utc"], sub["sci_rec"], lw=0.5, color=det_colors[d],
                ls='--', alpha=0.55)
    ax.set_ylabel("counts / s", fontsize=11)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(loc='upper right', fontsize=9, ncol=6, title="solid: Sci_obs / dashed: Sci_rec")
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Sci observed (solid) vs Sci reconstructed (dashed) per det", fontsize=10)

    # Panel 7: residual per det
    ax = axes[6]
    for d in range(6):
        sub = df[df["det"]==d]
        ax.plot(sub["t_utc"], sub["resid"], lw=0.6, color=det_colors[d], label=f"A{d}")
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel("residual (cnt/s)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=6)
    ax.grid(alpha=0.3)
    ax.set_title(r"residual = Sci$_{rec}$ − Sci$_{obs}$ per detector", fontsize=10)

    # Panel 8: ACD + mlat (satellite-wide context)
    ax = axes[7]
    # Use only det=0 to get unique per-second values
    sub = df[df["det"]==0]
    ax.plot(sub["t_utc"], sub["ACD_sum"], 'b-', lw=0.7, label="ACD_sum")
    ax2 = ax.twinx()
    ax2.plot(sub["t_utc"], sub["mlat"], 'k-', lw=0.7, label="|mlat|")
    ax2.axhline(20, color='gray', ls=':', alpha=0.5)
    ax.set_ylabel("ACD_sum (cnt/s)", color='b', fontsize=11)
    ax.set_yscale("log")
    ax2.set_ylabel("|mlat| (deg)", color='k', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Satellite-wide context: ACD_sum (CR proxy) and geomagnetic latitude", fontsize=10)
    ax.set_xlabel("UTC time", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_lc_box_zoom.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
