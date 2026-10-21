#!/usr/bin/env python3
"""Box A detail around 2019-08-15 SAA in/out: 6 detectors plotted separately,
ALL engineering data on its own panel.

Each Box A detector colored differently; 5 s of transition data trimmed at gap edges.
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
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed/clean_relaxed_2019.parquet"
TARGET_DATE = "2019-08-15"


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    LL=lc*L; lf=1.0-dt/lc
    pred=pho-(wide+(sci+C)*LL)/lf
    n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
    mx=pho-wide; out=large+n*1024.; ov=out>mx
    if ov.any():
        nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0)
        out=large+np.where(ov,nm,n)*1024.
    return out


def main():
    pf = pq.ParquetFile(CACHE)
    print(f"Streaming for {TARGET_DATE} Box A...", flush=True)
    chunks = []
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[(b["date"]==TARGET_DATE)&(b["box"]=="A")]
        if len(b)>0: chunks.append(b)
    import pandas as pd
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    if df is None or len(df)==0:
        print("no data"); return
    df = df.sort_values(["det","met_sec"]).reset_index(drop=True)

    # Find the longest gap in met_sec (likely SAA passage)
    unique_met = np.sort(df["met_sec"].unique())
    diffs = np.diff(unique_met)
    gap_idx = np.argmax(diffs)
    gap_start = int(unique_met[gap_idx])
    gap_end = int(unique_met[gap_idx+1])
    print(f"Largest gap: {gap_start} → {gap_end}  ({(gap_end-gap_start)/60:.1f}min)")
    win_lo = gap_start - 1800
    win_hi = gap_end + 1800
    df = df[(df["met_sec"]>=win_lo)&(df["met_sec"]<=win_hi)].copy()
    print(f"  {len(df):,} rows in ±30 min window")

    # Compute v5t C / Sci_rec / residual
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)
    am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
    am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-20.0)**2
    d_arr=np.array([np.datetime64(d) for d in df["date"].values])
    ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
    g_t=1.0-beta*ty
    k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
    detid = df["det"].values  # Box A so global = 0+det
    C_v5 = s0_det[detid]*g_t*(1.0+k_t*mt) + C0
    df["C_v5"] = C_v5; df["mlat"] = am

    pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
    wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
    lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
    LL=lc*L; lf=1.0-dtv/lc
    lv2 = unwrap_v2(pho, lg, wd, sci, lc, dtv, C_v5)
    mle = pho - ((sci+MIN_C_SLACK)*LL + wd)/lf
    n3 = np.round((lv2-lg)/1024).astype(int)
    nmax = np.maximum(np.floor((mle-lg)/1024.).astype(int), 0)
    n_final = np.where(n3>nmax, nmax, n3)
    df["N_wraps"] = n_final
    df["Large_unwrap"] = lg + n_final*1024.
    base = (pho - df["Large_unwrap"].values)*lf/LL - wd/LL
    df["sci_rec"] = base - C_v5
    df["resid"] = df["sci_rec"] - sci
    df["sci_over_pho"] = sci / np.maximum(pho, 1)
    df["wide_over_pho"] = wd / np.maximum(pho, 1)
    df["large_over_pho"] = lg / np.maximum(pho, 1)
    df["dt_frac"] = dtv / np.maximum(lc, 1) * 100

    # Trim 5 s on each side of the gap
    TRIM = 5
    df = df[(df["met_sec"] < gap_start - TRIM) | (df["met_sec"] > gap_end + TRIM)].copy()

    def break_at_gaps(t, y, met, threshold_s=60):
        diffs = np.diff(met)
        gap_pos = np.where(diffs > threshold_s)[0]
        if len(gap_pos) == 0:
            return t, y
        out_t = []; out_y = []
        prev = 0
        for gp in gap_pos:
            out_t.extend(t[prev:gp+1]); out_y.extend(y[prev:gp+1])
            mid_t = t[gp] + (t[gp+1] - t[gp]) / 2
            out_t.append(mid_t); out_y.append(np.nan)
            prev = gp + 1
        out_t.extend(t[prev:]); out_y.extend(y[prev:])
        return np.array(out_t), np.array(out_y)

    epoch = np.datetime64("2012-01-01T00:00:00")

    # Pick single detector A0 for clean per-second view
    df_a0 = df[df["det"]==0].sort_values("met_sec").reset_index(drop=True)
    print(f"  A0 alone: {len(df_a0):,} rows")

    def plot_a0(ax, col, title, ylabel, yscale=None, ylim=None,
                 log_linthresh=None, extras=None, color='b'):
        t_raw = epoch + df_a0["met_sec"].values * np.timedelta64(1,"s")
        t, y = break_at_gaps(t_raw, df_a0[col].values, df_a0["met_sec"].values)
        ax.plot(t, y, lw=0.5, color=color, label="A0")
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=log_linthresh or 10)
        elif yscale:
            ax.set_yscale(yscale)
        if ylim: ax.set_ylim(*ylim)
        if extras:
            for v, c, lbl in extras:
                ax.axhline(v, color=c, ls='--', alpha=0.5, label=lbl)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3, which='both')
        ax.set_title(title, fontsize=10)

    # ─── Build figure ───
    fig, axes = plt.subplots(17, 1, figsize=(14, 60), sharex=True,
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle(
        f"Detector A0 detail — {TARGET_DATE}, ±30 min around SAA gap\n"
        f"SAA gap shaded grey; transition data (5 s on each side) trimmed",
        fontsize=12, fontweight='bold')

    t_gap_start = epoch + gap_start * np.timedelta64(1,"s")
    t_gap_end   = epoch + gap_end   * np.timedelta64(1,"s")
    for ax in axes:
        ax.axvspan(t_gap_start, t_gap_end, alpha=0.18, color='gray')

    plot_a0(axes[0],  "PHO",         "1. PHO (NaI trigger count, 1B)",
                 "cnt / frame", yscale="symlog", log_linthresh=100)
    plot_a0(axes[1],  "Wide",        "2. Wide (CsI signal count, 1B)",
                 "cnt / frame", yscale="symlog", log_linthresh=10)
    plot_a0(axes[2],  "Large",       "3. Large RAW (0–1023 wrap, 1B)",
                 "cnt", ylim=(-50, 1200))
    plot_a0(axes[3],  "Large_unwrap","4. Large UNWRAPPED (after unwrap_v2 + cap)",
                 "cnt", yscale="symlog", log_linthresh=100)
    plot_a0(axes[4],  "N_wraps",     "5. N_wraps (number of 1024 multiples unwrap_v2 added)",
                 "N")
    plot_a0(axes[5],  "Sci_1s",      r"6. Sci$_{\rm obs}$ (1K HE-Evt 1 s window)",
                 "cnt / s", yscale="symlog", log_linthresh=10)
    plot_a0(axes[6],  "sci_rec",     r"7. Sci$_{\rm rec}$ (v5t conservation)",
                 "cnt / s", yscale="symlog", log_linthresh=10)
    plot_a0(axes[7],  "resid",       r"8. residual = Sci$_{\rm rec}$ − Sci$_{\rm obs}$",
                 "cnt / s", extras=[(0, 'k', "zero")])
    plot_a0(axes[8],  "C_v5",        "9. v5t baseline C (per detector)",
                 "C (cnt/s)")
    plot_a0(axes[9],  "sci_over_pho","10. Sci / PHO ratio", "ratio",
                 yscale="log", ylim=(0.01, 5),
                 extras=[(0.5, 'k', "0.5 normal")])
    plot_a0(axes[10], "wide_over_pho","11. Wide / PHO ratio (CsI/NaI trigger fraction)",
                 "ratio", yscale="log", ylim=(0.001, 1),
                 extras=[(0.3, 'k', "0.3 bright-source")])
    plot_a0(axes[11], "large_over_pho","12. Large / PHO ratio (over-threshold fraction)",
                 "ratio", yscale="log", ylim=(0.0001, 1))
    plot_a0(axes[12], "dt_frac",     "13. Dead-time fraction Dt / L_cyc",
                 "%", ylim=(0, 20))
    plot_a0(axes[13], "L_cycles",    "14. L_cycles (integration cycles)",
                 r"$L_{\rm cyc}$ (16 μs)", ylim=(58700, 59100))
    plot_a0(axes[14], "HV",          "15. HV (per detector, working voltage)",
                 "V", ylim=(-1010, -960))

    # Panel 16: satellite-wide ACD + PM (use df_a0 already filtered)
    ax = axes[15]
    one = df_a0
    t_raw = epoch + one["met_sec"].values * np.timedelta64(1,"s")
    t_acd, y_acd = break_at_gaps(t_raw, one["ACD_sum"].values, one["met_sec"].values)
    ax.plot(t_acd, y_acd, 'k-', lw=0.6, label="ACD_sum (18-ch ASU)")
    ax2 = ax.twinx()
    for pm_col, c, lbl in [("PM_0", 'b', 'PM_0'), ("PM_1", 'g', 'PM_1'), ("PM_2", 'r', 'PM_2')]:
        t_pm, y_pm = break_at_gaps(t_raw, one[pm_col].values, one["met_sec"].values)
        ax2.plot(t_pm, y_pm, color=c, lw=0.5, alpha=0.7, label=lbl)
    ax.set_ylabel("ACD_sum (cnt/s)", fontsize=10, color='k')
    ax.set_yscale("log")
    ax2.set_ylabel("PM_0/1/2 (cnt/s)", fontsize=10, color='b')
    ax2.set_yscale("symlog", linthresh=1)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("16. In-situ particle counters (satellite-wide, APID 0548)", fontsize=10)

    # Panel 17: orbital geometry (|mlat|, Lat, Lon)
    ax = axes[16]
    t_ml, y_ml = break_at_gaps(t_raw, one["mlat"].values, one["met_sec"].values)
    ax.plot(t_ml, y_ml, 'k-', lw=0.6, label="|mlat| (AACGM v2)")
    ax2 = ax.twinx()
    t_la, y_la = break_at_gaps(t_raw, one["Lat"].values, one["met_sec"].values)
    t_lo, y_lo = break_at_gaps(t_raw, one["Lon"].values, one["met_sec"].values)
    ax2.plot(t_la, y_la, 'b-', lw=0.5, alpha=0.7, label="Lat")
    ax2.plot(t_lo, y_lo, 'r-', lw=0.5, alpha=0.7, label="Lon")
    ax.set_ylabel("|mlat| (deg)", fontsize=10, color='k')
    ax2.set_ylabel("Lat / Lon (deg)", fontsize=10, color='b')
    ax.axhline(20, color='gray', ls=':', alpha=0.5, label="B threshold 20°")
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("17. Orbital geometry (geomagnetic + ECEF lat/lon)", fontsize=10)
    ax.set_xlabel("UTC time", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_saa_A0_detail.png"
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
