#!/usr/bin/env python3
"""Compare 3 boxes (A/B/C) during the 2020-05-25 spike window.

Per-box per-second PHO/Sci/Sci_rec, to identify which box (if any) is in
abnormal mode during the SGR J1935+2154 ToO observation.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator

L = 16e-6; MIN_C_SLACK = 50.0
NEEDED = ["date","box","det","met_sec","PHO","Wide","Large","Sci_1s",
          "L_cycles","Dt","HV","Lat","Lon","ACD_sum","PM_0","PM_1","PM_2"]
CACHE_FILE = "/Volumes/Graphite/blink_per_sec_partial/20200525.parquet"  # RAW per-sec (no Stage 1 filter)
# Use exactly the same epoch + np.datetime64 arithmetic as the plotting code,
# so MET aligns with what the panel x-axis shows.
_epoch_for_align = np.datetime64("2012-01-01T00:00:00")
_target_start = np.datetime64("2020-05-25T09:00:00")
MET_START = int((_target_start - _epoch_for_align) / np.timedelta64(1, "s"))
MET_END   = MET_START + 3600   # exactly 1 hour from 09:00 to 10:00 UTC


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
    cz=np.load("n_below_study/v5_npz/v5t_calib.npz")
    s0_det=cz["s0_det"]; beta=float(cz["beta"]); t0=np.datetime64(str(cz["t0"]))
    w=float(cz["w"]); kc=cz["k_coeffs"]; C0=float(cz["C0"])
    grid=np.load("n_below_study/aacgm_grid_2020.npz")
    interp=RegularGridInterpolator((grid["lat_grid"],grid["lon_grid"]),grid["mlat"],
                                   bounds_error=False,fill_value=np.nan)

    pf = pq.ParquetFile(CACHE_FILE)
    chunks=[]
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[(b["met_sec"]>=MET_START)&(b["met_sec"]<=MET_END)]
        if len(b)>0: chunks.append(b)
    import pandas as pd
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    print(f"Loaded {len(df):,} raw rows (no Stage 1 filter)", flush=True)
    # Diagnostic: how many rows have HV in valid range, L_cyc > 50k?
    if df is not None:
        n_hv_ok = ((df["HV"]>-1100)&(df["HV"]<-900)).sum()
        n_lc_ok = (df["L_cycles"]>50000).sum()
        n_both = ((df["HV"]>-1100)&(df["HV"]<-900)&(df["L_cycles"]>50000)).sum()
        print(f"  HV in (-1100,-900): {n_hv_ok:,}  L_cycles>50k: {n_lc_ok:,}  both: {n_both:,}", flush=True)

    # Compute v5t C and Sci_rec
    am=np.abs(interp(np.column_stack([df["Lat"].values,df["Lon"].values])))
    am=np.where(np.isnan(am),0.0,am); mt=np.maximum(0.0,am-20.0)**2
    d_arr=np.array([np.datetime64(d) for d in df["date"].values])
    ty=(d_arr-t0).astype("timedelta64[D]").astype(float)/365.25
    g_t=1.0-beta*ty
    k_t=kc[0]+kc[1]*np.cos(w*ty)+kc[2]*np.sin(w*ty)
    box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
    detid=box_idx*6+df["det"].values
    C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0

    pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
    wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
    lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
    LL=lc*L; lf=1.0-dtv/lc
    lv5=unwrap_v2(pho,lg,wd,sci,lc,dtv,C_v5)
    base=(pho-lv5)*lf/LL-wd/LL
    sci_rec=base-C_v5
    df["sci_rec"]=sci_rec
    df["C_v5"]=C_v5

    # Per-box per-met_sec sums (6-det sum within each box)
    grouped = df.groupby(["box", "met_sec"]).agg(
        PHO_sum=("PHO", "sum"),
        Wide_sum=("Wide", "sum"),
        Large_sum=("Large", "sum"),
        Sci_sum=("Sci_1s", "sum"),
        Sci_rec_sum=("sci_rec", "sum"),
        C_v5_sum=("C_v5", "sum"),
        HV_avg=("HV", "mean"),
        L_cyc_avg=("L_cycles", "mean"),
        Dt_avg=("Dt", "mean"),
        ACD_sum=("ACD_sum", "first"),
        PM_0=("PM_0", "first"),
        PM_1=("PM_1", "first"),
        PM_2=("PM_2", "first"),
        Lat=("Lat", "first"),
        Lon=("Lon", "first"),
    ).reset_index()
    grouped["sci_over_pho"] = grouped["Sci_sum"] / np.maximum(grouped["PHO_sum"], 1)
    grouped["wide_over_pho"] = grouped["Wide_sum"] / np.maximum(grouped["PHO_sum"], 1)
    grouped["large_over_pho"] = grouped["Large_sum"] / np.maximum(grouped["PHO_sum"], 1)
    grouped["dt_frac"] = grouped["Dt_avg"] / np.maximum(grouped["L_cyc_avg"], 1) * 100
    grouped["resid_sum"] = grouped["Sci_rec_sum"] - grouped["Sci_sum"]

    # Print summary per box
    print(f"\n=== Per-box summary (entire 20-min window) ===")
    print(f"{'box':>4}  {'N_sec':>7}  {'PHO_med':>9}  {'Sci_med':>9}  "
          f"{'Sci/PHO_med':>12}  {'HV_med':>9}  {'L_cyc_med':>10}")
    for b in "ABC":
        sub = grouped[grouped["box"]==b]
        print(f"{b:>4}  {len(sub):>7,}  "
              f"{sub['PHO_sum'].median():>9.0f}  "
              f"{sub['Sci_sum'].median():>9.0f}  "
              f"{sub['sci_over_pho'].median():>12.4f}  "
              f"{sub['HV_avg'].median():>9.1f}  "
              f"{sub['L_cyc_avg'].median():>10.0f}")

    # Print sample rows for the worst spike second (around 09:25)
    spike_met = 265024500  # approx 09:25
    nearby = grouped[(grouped["met_sec"]>spike_met-5)&(grouped["met_sec"]<spike_met+5)]
    print(f"\n=== Rows near met_sec ≈ {spike_met} (~09:25 UTC) ===")
    print(nearby.to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    # ─── Plot ───
    epoch = np.datetime64("2012-01-01T00:00:00")
    fig, axes = plt.subplots(14, 1, figsize=(14, 56), sharex=True,
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle(
        "3-box compare 2020-05-25 (during SGR J1935+2154 ToO)\n"
        "HV=0 startup period removed; showing instrument-active window only\n"
        "Each panel: Box A (blue), Box B (green), Box C (red), 6-det sum per box",
        fontsize=12, fontweight='bold')

    colors = {'A':'blue', 'B':'green', 'C':'red'}

    def plot_per_box(ax, ycol, title, ylabel, yscale=None, ylim=None,
                     extras=None, log_linthresh=None):
        for b in "ABC":
            sub = grouped[grouped["box"]==b].sort_values("met_sec")
            t = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
            ax.plot(t, sub[ycol], lw=0.6, color=colors[b], label=f"Box {b}")
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=log_linthresh or 10)
        elif yscale:
            ax.set_yscale(yscale)
        if ylim: ax.set_ylim(*ylim)
        if extras:
            for v, c, lbl in extras:
                ax.axhline(v, color=c, ls='--', alpha=0.5, label=lbl)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='upper right', fontsize=8, ncol=4)
        ax.grid(alpha=0.3, which='both')
        ax.set_title(title, fontsize=10)

    plot_per_box(axes[0], "PHO_sum",  "1. PHO trigger count (1B engineering)",
                 "PHO 6-det sum (cnt/frame)", yscale="symlog", log_linthresh=100)
    plot_per_box(axes[1], "Wide_sum", "2. Wide / CsI signal count (1B engineering)",
                 "Wide 6-det sum (cnt/frame)", yscale="symlog", log_linthresh=10)
    plot_per_box(axes[2], "Large_sum","3. Large RAW counter (0–1023 wrap, 1B engineering)",
                 "Large RAW 6-det sum (cnt)", ylim=(-50, 7000))
    plot_per_box(axes[3], "Sci_sum",  r"4. Sci$_{\rm obs}$ (1K HE-Evt 1 s window)",
                 r"Sci$_{\rm obs}$ 6-det sum (cnt/s)", yscale="symlog", log_linthresh=10)
    plot_per_box(axes[4], "Sci_rec_sum", r"5. Sci$_{\rm rec}$ (v5t conservation reconstruction)",
                 r"Sci$_{\rm rec}$ 6-det sum (cnt/s)", yscale="symlog", log_linthresh=10)
    plot_per_box(axes[5], "resid_sum",   r"6. residual = Sci$_{\rm rec}$ − Sci$_{\rm obs}$",
                 "residual (cnt/s)", extras=[(0, 'k', "zero")])
    plot_per_box(axes[6], "C_v5_sum",    "7. v5t baseline C (per box, 6-det sum)",
                 "C (cnt/s)")
    plot_per_box(axes[7], "sci_over_pho","8. Sci / PHO ratio", "ratio",
                 yscale="log", ylim=(0.01, 5),
                 extras=[(0.5, 'k', "0.5 normal")])
    plot_per_box(axes[8], "wide_over_pho","9. Wide / PHO ratio (CsI / NaI trigger fraction)",
                 "ratio", yscale="log", ylim=(0.001, 1),
                 extras=[(0.3, 'k', "0.3 bright-source")])
    plot_per_box(axes[9], "large_over_pho","10. Large / PHO ratio (over-threshold fraction)",
                 "ratio", yscale="log", ylim=(0.001, 1))
    plot_per_box(axes[10], "dt_frac",  "11. Dead-time fraction Dt/L_cyc",
                 "dead-time %", ylim=(0, 20))
    plot_per_box(axes[11], "L_cyc_avg", "12. L_cycles (integration cycles, 1B)",
                 r"$L_{\rm cyc}$ (16 $\mu$s)", ylim=(58700, 59100))

    # Panel 13: satellite-wide ACD_sum + PM
    ax = axes[12]
    one_box = grouped[grouped["box"]=="A"].sort_values("met_sec")
    t = epoch + one_box["met_sec"].values * np.timedelta64(1,"s")
    ax.plot(t, one_box["ACD_sum"], 'k-', lw=0.6, label="ACD_sum (18-ch ASU)")
    ax2 = ax.twinx()
    ax2.plot(t, one_box["PM_0"], 'b-', lw=0.5, alpha=0.7, label="PM_0")
    ax2.plot(t, one_box["PM_1"], 'g-', lw=0.5, alpha=0.7, label="PM_1")
    ax2.plot(t, one_box["PM_2"], 'r-', lw=0.5, alpha=0.7, label="PM_2")
    ax.set_ylabel("ACD_sum (cnt/s)", fontsize=10, color='k')
    ax.set_yscale("log")
    ax2.set_ylabel("PM_0/1/2 (cnt/s)", fontsize=10, color='b')
    ax2.set_yscale("symlog", linthresh=1)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("13. In-situ particle counters (satellite-wide, APID 0548)", fontsize=10)

    # Panel 14: HV per box
    ax = axes[13]
    for b in "ABC":
        sub = grouped[grouped["box"]==b].sort_values("met_sec")
        t = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        ax.plot(t, sub["HV_avg"], lw=0.6, color=colors[b], label=f"Box {b}")
    ax.set_ylabel("HV (V)", fontsize=10)
    ax.set_ylim(-1010, -960)
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title("14. PMT high voltage (per box)", fontsize=10)
    ax.set_xlabel("UTC time", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    # Trim x-axis to skip the HV=0 (SAA protection) period at the start of the hour.
    # Find first met_sec where HV is in nominal range; use that as the display start.
    hv_ok_mask = (df["HV"] > -1100) & (df["HV"] < -900)
    first_ok = int(df.loc[hv_ok_mask, "met_sec"].min())
    t_lo = epoch + first_ok * np.timedelta64(1, "s")
    t_hi = epoch + MET_END   * np.timedelta64(1, "s")
    for a in axes:
        a.set_xlim(t_lo, t_hi)
    print(f"  x-axis trimmed: HV first valid at met_sec={first_ok} "
          f"({first_ok - MET_START}s after 09:00 UTC)", flush=True)

    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_3box_compare.png"
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
