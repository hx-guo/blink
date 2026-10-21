#!/usr/bin/env python3
"""Why does row count and C both jump at |mlat|=16°?

Look at:
  - N(|mlat|) full 0-60° at 0.2° resolution — is 16° the only jump?
  - 2D Lat × |mlat| density map — does 16° correspond to a geographic Lat boundary?
  - Lon-stratified N(|mlat|) — does SAA cut explain it?
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.interpolate import RegularGridInterpolator

CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"


def main():
    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)
    files = sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))[:5]

    # 0.2° mlat × 1° Lat × 30° Lon strata
    mlat_edges = np.arange(0, 60.001, 0.2)
    lat_edges = np.arange(-50, 50.001, 1)
    lon_edges = np.arange(0, 360.001, 30)
    n_mlat = len(mlat_edges) - 1
    n_lat = len(lat_edges) - 1
    n_lon = len(lon_edges) - 1

    H_lat = np.zeros((n_lat, n_mlat), dtype=np.int64)
    H_lon = np.zeros((n_lon, n_mlat), dtype=np.int64)
    N_mlat = np.zeros(n_mlat, dtype=np.int64)

    print("Scanning sample row groups...")
    for f in files:
        pf = pq.ParquetFile(f)
        rg_pick = pf.num_row_groups // 2
        df = pf.read_row_group(int(rg_pick), columns=["Lat", "Lon"]).to_pandas()
        am = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
        ok = np.isfinite(am)
        am = am[ok]
        lat = df["Lat"].values[ok]
        lon = df["Lon"].values[ok]
        mi = np.digitize(am, mlat_edges) - 1
        la_i = np.digitize(lat, lat_edges) - 1
        lo_i = np.digitize(lon, lon_edges) - 1
        m_ok = (mi >= 0) & (mi < n_mlat)
        l_ok = m_ok & (la_i >= 0) & (la_i < n_lat)
        o_ok = m_ok & (lo_i >= 0) & (lo_i < n_lon)
        np.add.at(N_mlat, mi[m_ok], 1)
        np.add.at(H_lat, (la_i[l_ok], mi[l_ok]), 1)
        np.add.at(H_lon, (lo_i[o_ok], mi[o_ok]), 1)
        print(f"  {os.path.basename(f)}: {m_ok.sum():,} valid rows")

    mlat_centers = 0.5 * (mlat_edges[:-1] + mlat_edges[1:])
    print(f"\nTotal rows: {N_mlat.sum():,}")

    # ─── Plot ───
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Origin of the 16° jump — is it selection or physical?",
                 fontsize=13, fontweight='bold')

    # Panel 1: N(|mlat|) 0-60°
    ax = axes[0, 0]
    ax.plot(mlat_centers, N_mlat, '-', lw=1.2, color='black')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("row count per 0.2° bin", fontsize=11)
    ax.set_title("1. Sample density N(|mlat|) — is 16° special?",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Panel 2: 2D Lat × |mlat|
    ax = axes[0, 1]
    extent = [mlat_edges[0], mlat_edges[-1], lat_edges[0], lat_edges[-1]]
    im = ax.imshow(H_lat, origin='lower', aspect='auto', extent=extent,
                   cmap='viridis', norm=mc.LogNorm(vmin=1, vmax=H_lat.max()))
    ax.axvline(16, ls='--', color='red', alpha=0.7, lw=1.5)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("geographic Lat (deg)", fontsize=11)
    ax.set_title("2. Lat × |mlat| density (log) — does 16° map to a Lat band?",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label='row count')

    # Panel 3: Lon-stratified
    ax = axes[1, 0]
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    cmap = plt.cm.tab12 if hasattr(plt.cm, 'tab12') else plt.cm.tab10
    for j, lc in enumerate(lon_centers):
        ax.plot(mlat_centers, H_lon[j], '-', lw=1.5,
                color=plt.cm.viridis(j/(n_lon-1)),
                label=f'Lon {lon_edges[j]:.0f}-{lon_edges[j+1]:.0f}°')
    ax.axvline(16, ls='--', color='red', alpha=0.7, lw=1.5)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("row count per (Lon-bin, mlat-bin)", fontsize=11)
    ax.set_title("3. N(|mlat|) by Lon — same jump in all longitudes?",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Panel 4: derivative of log N vs mlat (find any sharp jumps)
    ax = axes[1, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        logN = np.log10(np.maximum(N_mlat, 1))
    dlogN = np.gradient(logN, mlat_centers)
    ax.plot(mlat_centers, dlogN, '-', lw=1.2, color='black')
    ax.axvline(16, ls='--', color='C2', alpha=0.7, lw=1.5, label='|mlat|=16°')
    ax.axhline(0, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel("|mlat| (deg)", fontsize=11)
    ax.set_ylabel("d(log10 N) / d|mlat|", fontsize=11)
    ax.set_title("4. d(log N)/d|mlat| — peaks = sharp density jumps",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_mlat_16_origin.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")

    # Print N around 16°
    print("\n=== N(|mlat|) around 16° ===")
    for i, m in enumerate(mlat_centers):
        if 13 <= m <= 19:
            print(f"  |mlat|={m:5.1f}°  N={N_mlat[i]:>8,}")


if __name__ == "__main__":
    main()
