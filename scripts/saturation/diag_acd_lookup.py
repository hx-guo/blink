#!/usr/bin/env python3
"""Use ACD count-rate lookup as a radiation-environment predictor.

Compare to mlat:
  Panel 1: ACD(lat, lon) map + SAA-keep window overlay
  Panel 2: cache row density on (lat, lon) — see SAA cut footprint
  Panel 3: ACD value distribution in cache (histogram)
  Panel 4: scatter C_truth vs ACD (binned mean)
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as patches
from scipy.interpolate import RegularGridInterpolator

L = 16e-6
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"
ACD_PATH = "/Users/skyair/Developer/ihep/astro_sift/astro_sift/satellites/hxmt/acd.txt"


def load_acd_lookup():
    a = np.loadtxt(ACD_PATH)
    lon_mesh = a[0:180]      # 180 × 360
    lat_mesh = a[180:360]
    acd = a[360:540]
    lat_grid = lat_mesh[:, 0]   # ascending -89.5 .. 89.5
    lon_grid = lon_mesh[0, :]   # ascending -179.5 .. 179.5
    return lat_grid, lon_grid, acd


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
    lat_grid, lon_grid, acd = load_acd_lookup()
    print(f"ACD lookup: lat {lat_grid[0]} to {lat_grid[-1]} (n={len(lat_grid)}), "
          f"lon {lon_grid[0]} to {lon_grid[-1]} (n={len(lon_grid)})")
    print(f"ACD finite={np.isfinite(acd).mean()*100:.1f}%,  "
          f"range {np.nanmin(acd):.1f} - {np.nanmax(acd):.1f} cnt/s")

    # Cache lon is [0, 360). ACD lon is [-180, 180). Convert ACD lon to [0, 360)
    lon_grid_360 = (lon_grid + 360) % 360
    sort_idx = np.argsort(lon_grid_360)
    lon_grid_360 = lon_grid_360[sort_idx]
    acd_360 = acd[:, sort_idx]
    # Interpolator using (lat, lon_360)
    acd_lookup = RegularGridInterpolator((lat_grid, lon_grid_360), acd_360,
                                         bounds_error=False, fill_value=np.nan)

    # ─── Sample cache rows ───
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))[:5]
    NEEDED = ["PHO","Wide","Large","Sci_1s","L_cycles","Dt","Lat","Lon"]
    sample = {"C": [], "ACD_lookup": [], "Lat": [], "Lon": []}
    for f in files:
        pf = pq.ParquetFile(f); rg = pf.num_row_groups // 2
        df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
        pho = df["PHO"].astype(float).values; lg = df["Large"].astype(float).values
        wd  = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
        lc  = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
        LL = lc*L; lf = 1.0 - dtv/lc
        lv = unwrap_v2(pho, lg, wd, sci, lc, dtv, 150.0)
        base = (pho - lv)*lf/LL - wd/LL
        C_truth = base - sci
        lat = df["Lat"].values; lon = df["Lon"].values
        acd_v = acd_lookup(np.column_stack([lat, lon]))
        ok = np.isfinite(C_truth) & (np.abs(C_truth) < 800) & (sci > 50) & np.isfinite(acd_v)
        for k, v in zip(["C","ACD_lookup","Lat","Lon"], [C_truth[ok], acd_v[ok], lat[ok], lon[ok]]):
            sample[k].append(v)
        print(f"  {os.path.basename(f)}: {ok.sum():,} rows kept")
    for k in sample:
        sample[k] = np.concatenate(sample[k])
    n_total = len(sample["C"])
    print(f"\nTotal sample: {n_total:,} rows")

    # ─── Plot ───
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("ACD lookup as a radiation-environment predictor",
                 fontsize=13, fontweight='bold')

    # Panel 1: ACD(lat, lon) map + SAA-keep window
    ax = axes[0, 0]
    acd_plot = acd_360.copy()
    extent = [0, 360, lat_grid[0], lat_grid[-1]]
    im = ax.imshow(acd_plot, origin='lower', aspect='auto', extent=extent,
                   cmap='hot', norm=mc.LogNorm(vmin=1, vmax=np.nanmax(acd_plot)))
    plt.colorbar(im, ax=ax, label='ACD count rate (cnt/s)')
    # SAA-keep window overlay
    ax.axvline(30, ls='--', color='cyan', lw=1.5, label='SAA keep: Lon (30, 270)')
    ax.axvline(270, ls='--', color='cyan', lw=1.5)
    # HXMT lat range
    ax.axhline(43, ls=':', color='lime', lw=1.5, label='HXMT lat ±43°')
    ax.axhline(-43, ls=':', color='lime', lw=1.5)
    ax.set_xlabel("Lon (deg, [0,360))", fontsize=11)
    ax.set_ylabel("Lat (deg)", fontsize=11)
    ax.set_title("1. ACD(lat, lon) map — log-scale, HXMT in-situ flux", fontsize=11)
    ax.legend(fontsize=9, loc='lower right')

    # Panel 2: cache row density on (lat, lon)
    ax = axes[0, 1]
    h, xe, ye = np.histogram2d(sample["Lon"], sample["Lat"],
                                bins=[np.arange(0, 361, 5), np.arange(-50, 51, 2)])
    im = ax.imshow(h.T, origin='lower', aspect='auto',
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap='viridis', norm=mc.LogNorm(vmin=1, vmax=h.max()))
    plt.colorbar(im, ax=ax, label='cache row count')
    ax.axvline(30, ls='--', color='cyan', lw=1.5)
    ax.axvline(270, ls='--', color='cyan', lw=1.5)
    ax.set_xlabel("Lon (deg)", fontsize=11)
    ax.set_ylabel("Lat (deg)", fontsize=11)
    ax.set_title(f"2. Cache row density on (Lat, Lon) — sample, "
                 f"keep Lon (30, 270)", fontsize=11)

    # Panel 3: ACD value distribution in cache
    ax = axes[1, 0]
    ax.hist(sample["ACD_lookup"], bins=np.logspace(0, 3.5, 100),
            color='C0', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel("ACD lookup value (cnt/s)", fontsize=11)
    ax.set_ylabel("# cache rows", fontsize=11)
    ax.set_title(f"3. ACD distribution in cache rows (n={n_total/1e6:.1f}M)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(np.nanmedian(sample["ACD_lookup"]), ls='--', color='red',
               label=f'median = {np.nanmedian(sample["ACD_lookup"]):.0f}')
    ax.legend(fontsize=10)

    # Panel 4: binned C_truth vs ACD
    ax = axes[1, 1]
    acd_bins = np.logspace(0, 3.5, 60)
    bin_idx = np.digitize(sample["ACD_lookup"], acd_bins) - 1
    C_mean = np.full(len(acd_bins)-1, np.nan)
    C_n = np.zeros(len(acd_bins)-1)
    for i in range(len(acd_bins)-1):
        m = bin_idx == i
        if m.sum() > 100:
            C_mean[i] = sample["C"][m].mean()
            C_n[i] = m.sum()
    centers = np.sqrt(acd_bins[:-1] * acd_bins[1:])
    valid = np.isfinite(C_mean)
    ax.plot(centers[valid], C_mean[valid], 'o-', lw=1.5, ms=5, color='black')
    ax.set_xscale('log')
    ax.set_xlabel("ACD lookup (cnt/s)", fontsize=11)
    ax.set_ylabel("⟨C_truth⟩ (cnt/s)", fontsize=11)
    ax.set_title("4. Binned C vs ACD lookup — is ACD a good predictor?",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(centers[valid], C_n[valid], '-', color='C0', alpha=0.5,
             label='row count per bin')
    ax2.set_yscale('log')
    ax2.set_ylabel('row count per bin', fontsize=10, color='C0')

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_acd_lookup.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")

    # Save samples for downstream fit
    np.savez("n_below_study/v5_npz/acd_sample.npz",
             C=sample["C"], ACD=sample["ACD_lookup"],
             Lat=sample["Lat"], Lon=sample["Lon"])
    print("Saved n_below_study/v5_npz/acd_sample.npz")


if __name__ == "__main__":
    main()
