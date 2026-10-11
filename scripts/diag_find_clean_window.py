#!/usr/bin/env python3
"""Find a clean 1-hour window: all 3 boxes have Sci/PHO ≈ 0.5,
no SGR/FRB ToO contamination, no SAA gaps.

Uses the cleaned cache (already filtered by Stage 1 HV/L_cycles).
Reports the best candidate window with met_sec range + UTC date.

Then plot the 3-box compare for that window.
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
          "L_cycles","Dt","HV","Lat","Lon","ACD_sum"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"


def main():
    # Use 2019 — entirely before SGR J1935+2154 FRB 200428 (2020-04-28)
    fn = os.path.join(CACHE, "clean_relaxed_2019.parquet")
    print(f"Scanning {fn}...", flush=True)
    pf = pq.ParquetFile(fn)
    # Sample row groups; compute per-hour per-box Sci/PHO
    print(f"  num row groups: {pf.num_row_groups}", flush=True)

    candidates = []  # (date_str, hour, sci_pho_A, sci_pho_B, sci_pho_C, met_sec_start)
    # Iterate a sparse sample of row groups
    sample_rgs = np.linspace(0, pf.num_row_groups-1, 30).astype(int)
    sample_rgs = np.unique(sample_rgs)
    for rg in sample_rgs:
        df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
        # Group by (date, hour, box) and median Sci/PHO
        df["hour"] = df["met_sec"].values // 3600
        for (date, hr, box), sub in df.groupby(["date","hour","box"]):
            if len(sub) < 100: continue
            sci = sub["Sci_1s"].astype(float).values
            pho = sub["PHO"].astype(float).values
            ratio_med = np.median(sci / np.maximum(pho, 1))
            candidates.append((date, hr, box, ratio_med, sub["met_sec"].min()))
    print(f"  collected {len(candidates):,} (date, hour, box) summaries", flush=True)

    # Pivot: for each (date, hour), get ratio for A/B/C
    from collections import defaultdict
    grouped = defaultdict(dict)
    grouped_met = {}
    for date, hr, box, ratio, m0 in candidates:
        key = (date, hr)
        grouped[key][box] = ratio
        grouped_met[key] = (m0 // 3600) * 3600  # hour-aligned met_sec

    # Find candidates where all 3 boxes have ratio in [0.4, 0.65]
    clean = []
    for key, ratios in grouped.items():
        if set(ratios.keys()) != {"A","B","C"}: continue
        if all(0.4 < r < 0.65 for r in ratios.values()):
            # Compute closeness: distance from 0.5 for all 3
            score = sum(abs(r - 0.5) for r in ratios.values())
            clean.append((key, ratios, grouped_met[key], score))

    if not clean:
        print("No clean candidates found"); return
    clean.sort(key=lambda x: x[3])
    print(f"\nFound {len(clean)} clean (all-3-box, ratio in [0.4, 0.65]) candidates")
    print(f"Top 10:")
    for (date, hr), ratios, m0, score in clean[:10]:
        print(f"  {date} hour-of-day {hr%24:02d}  "
              f"A={ratios['A']:.3f}  B={ratios['B']:.3f}  C={ratios['C']:.3f}  "
              f"score={score:.3f}  met_sec≈{m0}")

    # Pick the best
    (target_date, target_hour), best_ratios, met_start, _ = clean[0]
    target_met_lo = met_start
    target_met_hi = met_start + 3600
    print(f"\nPicked: {target_date}  met_sec [{target_met_lo}, {target_met_hi}]")

    # Now load all 3-box data for that window from cache
    chunks = []
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=NEEDED).to_pandas()
        b = b[(b["date"]==target_date)&(b["met_sec"]>=target_met_lo)&(b["met_sec"]<=target_met_hi)]
        if len(b)>0: chunks.append(b)
    import pandas as pd
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    print(f"  loaded {len(df):,} rows for plot", flush=True)

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
    box_idx=np.select([df["box"].values==b for b in "ABC"],[0,1,2],default=0)
    detid=box_idx*6+df["det"].values
    C_v5=s0_det[detid]*g_t*(1.0+k_t*mt)+C0
    df["C_v5"]=C_v5

    # unwrap + base
    def unwrap_v2(pho, large, wide, sci, lc, dt, C):
        LL=lc*L; lf=1.0-dt/lc
        pred=pho-(wide+(sci+C)*LL)/lf
        n=np.maximum(np.round((pred-large)/1024.).astype(int),0)
        mx=pho-wide; out=large+n*1024.; ov=out>mx
        if ov.any():
            nm=np.maximum(np.floor((mx-large)/1024.).astype(int),0)
            out=large+np.where(ov,nm,n)*1024.
        return out
    pho=df["PHO"].astype(float).values; lg=df["Large"].astype(float).values
    wd=df["Wide"].astype(float).values; sci=df["Sci_1s"].astype(float).values
    lc=df["L_cycles"].astype(float).values; dtv=df["Dt"].astype(float).values
    LL=lc*L; lf=1.0-dtv/lc
    lv2=unwrap_v2(pho,lg,wd,sci,lc,dtv,C_v5)
    mle=pho-((sci+MIN_C_SLACK)*LL+wd)/lf
    n3=np.round((lv2-lg)/1024).astype(int)
    nmax=np.maximum(np.floor((mle-lg)/1024.).astype(int),0)
    lv5=lg+np.where(n3>nmax,nmax,n3)*1024.
    base=(pho-lv5)*lf/LL-wd/LL
    df["sci_rec"] = base - C_v5

    grouped = df.groupby(["box", "met_sec"]).agg(
        PHO_sum=("PHO", "sum"), Sci_sum=("Sci_1s", "sum"),
        Sci_rec_sum=("sci_rec", "sum"),
        HV_avg=("HV", "mean"),
    ).reset_index()
    grouped["sci_over_pho"] = grouped["Sci_sum"] / np.maximum(grouped["PHO_sum"], 1)
    print(f"\n=== Per-box summary ===")
    for b in "ABC":
        sub = grouped[grouped["box"]==b]
        print(f"  Box {b}  N_sec={len(sub):>5}  PHO_med={sub['PHO_sum'].median():>6.0f}  "
              f"Sci_med={sub['Sci_sum'].median():>6.0f}  "
              f"Sci/PHO={sub['sci_over_pho'].median():.4f}  HV_med={sub['HV_avg'].median():.1f}")

    # Plot
    mpl.rcParams.update({"font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(
        f"Clean window: {target_date} (2019, well before SGR FRB 200428)\n"
        f"3-box compare: Box A (blue), Box B (green), Box C (red)\n"
        f"Sci/PHO ratios:  A = {best_ratios['A']:.3f},  B = {best_ratios['B']:.3f},  C = {best_ratios['C']:.3f}",
        fontsize=12, fontweight='bold')
    colors = {'A':'blue', 'B':'green', 'C':'red'}
    epoch = np.datetime64("2012-01-01T00:00:00")

    ax = axes[0]
    for b in "ABC":
        sub = grouped[grouped["box"]==b].sort_values("met_sec")
        t = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        ax.plot(t, sub["PHO_sum"], lw=0.7, color=colors[b], label=f"Box {b}")
    ax.set_ylabel("PHO 6-det sum (cnt/frame)", fontsize=11)
    ax.set_yscale("symlog", linthresh=100)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("PHO trigger count (1B engineering counter)", fontsize=11)

    ax = axes[1]
    for b in "ABC":
        sub = grouped[grouped["box"]==b].sort_values("met_sec")
        t = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        ax.plot(t, sub["Sci_sum"], lw=0.7, color=colors[b], label=f"Box {b}")
    ax.set_ylabel(r"Sci$_{\rm obs}$ 6-det sum (cnt/s)", fontsize=11)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_title(r"Sci$_{\rm obs}$ (1K HE-Evt 1s window)", fontsize=11)

    ax = axes[2]
    for b in "ABC":
        sub = grouped[grouped["box"]==b].sort_values("met_sec")
        t = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        ax.plot(t, sub["sci_over_pho"], lw=0.7, color=colors[b], label=f"Box {b}")
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label="0.5 (normal)")
    ax.set_ylabel(r"Sci$_{\rm obs}$ / PHO ratio", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_title("Sci/PHO ratio (should be all ~0.5 in clean window)", fontsize=11)

    ax = axes[3]
    for b in "ABC":
        sub = grouped[grouped["box"]==b].sort_values("met_sec")
        t = epoch + sub["met_sec"].values * np.timedelta64(1,"s")
        ax.plot(t, sub["HV_avg"], lw=0.7, color=colors[b], label=f"Box {b}")
    ax.set_ylabel("HV (V)", fontsize=11)
    ax.set_ylim(-1010, -960)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title("HV (V): per-box working voltage", fontsize=11)
    ax.set_xlabel("UTC time", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_3box_clean_window.png"
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
