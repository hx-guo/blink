#!/usr/bin/env python3
"""Isolate the three C-model factors one at a time (control-variable study):
  A. geomagnetic latitude  — fix time window (2020), residual vs |mlat|
  B. PMT outgassing g(t)    — fix equator (|mlat|<5, mlat-term=0), residual vs time
  C. solar cycle k(t)       — fix high mlat, high/equator ratio vs time

residual = base_v2 - Sci  (C=150 coarse unwrap)  ~=  C(det,|mlat|,t).
If a panel shows a clear trend, that factor is needed; if flat, it can be dropped.

Output: plots/diag_factor_isolation.png
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

L_CYCLES_TO_SEC = 16e-6
B_THRESHOLD = 20.0
NEEDED = ["date", "box", "det", "PHO", "Wide", "Large", "Sci_1s", "L_cycles", "Dt", "Lat", "Lon"]
CACHE = "/Volumes/Graphite/work/blink/blink_clean_relaxed"


def unwrap_large_v2(pho, large, wide, sci, l_cycles, dt, C):
    pho = np.asarray(pho, float); large = np.asarray(large, float)
    wide = np.asarray(wide, float); sci = np.asarray(sci, float)
    L = np.asarray(l_cycles, float) * L_CYCLES_TO_SEC
    lf = 1.0 - np.asarray(dt, float) / np.asarray(l_cycles, float)
    predicted = pho - (wide + (sci + C) * L) / lf
    n = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    maxa = pho - wide; lc = large + n * 1024.0; over = lc > maxa
    if over.any():
        nmax = np.maximum(np.floor((maxa - large) / 1024.0).astype(int), 0)
        lc = large + np.where(over, nmax, n) * 1024.0
    return lc


def main():
    grid = np.load("n_below_study/aacgm_grid_2020.npz")
    interp = RegularGridInterpolator((grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                                     bounds_error=False, fill_value=np.nan)
    cz = np.load("n_below_study/v5_npz/v5t_calib.npz")
    g_A, g_tau, w, kc = float(cz["g_A"]), float(cz["g_tau"]), float(cz["w"]), cz["k_coeffs"]
    t0 = np.datetime64(str(cz["t0"]))

    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "clean_relaxed_20*.parquet")))
             if "sample" not in f]
    cols = {k: [] for k in ["res", "amlat", "ty", "year"]}
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in np.unique(np.linspace(0, pf.num_row_groups - 1, 6).astype(int)):
            df = pf.read_row_group(int(rg), columns=NEEDED).to_pandas()
            am = np.abs(interp(np.column_stack([df["Lat"].values, df["Lon"].values])))
            am = np.where(np.isnan(am), 0.0, am)
            pho = df["PHO"].astype(float).values; large = df["Large"].astype(float).values
            wide = df["Wide"].astype(float).values; sci = df["Sci_1s"].astype(float).values
            lc = df["L_cycles"].astype(float).values; dtv = df["Dt"].astype(float).values
            L = lc * L_CYCLES_TO_SEC; lf = 1.0 - dtv / lc
            lv2 = unwrap_large_v2(pho, large, wide, sci, lc, dtv, 150.0)
            res = (pho - lv2) * lf / L - wide / L - sci
            ok = (wide / np.maximum(pho, 1) < 0.3) & (sci > 100) & np.isfinite(res) & (np.abs(res) < 2000)
            ty = (pd.to_datetime(df["date"]).values.astype("datetime64[D]") - t0).astype("timedelta64[D]").astype(float) / 365.25
            cols["res"].append(res[ok]); cols["amlat"].append(am[ok])
            cols["ty"].append(ty[ok]); cols["year"].append(df["date"].str[:4].astype(int).values[ok])
        print(f"  scanned {os.path.basename(f)}", flush=True)
    for k in cols:
        cols[k] = np.concatenate(cols[k])
    res, amlat, ty, year = cols["res"], cols["amlat"], cols["ty"], cols["year"]
    print(f"total {len(res):,} clean rows")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # --- A: mlat, fix time = 2020 ---
    ax = axes[0]
    m = (year == 2020)
    mb = np.linspace(0, 55, 23)
    mc = 0.5 * (mb[:-1] + mb[1:])
    med = np.array([np.median(res[m & (amlat >= mb[i]) & (amlat < mb[i+1])])
                    if (m & (amlat >= mb[i]) & (amlat < mb[i+1])).sum() > 100 else np.nan
                    for i in range(len(mc))])
    ax.plot(mc, med, "o-", color="#1f77b4", label="median residual (2020)")
    base = np.nanmedian(res[m & (amlat < 20)])
    C0 = float(cz["C0"])
    ty20 = 2020.5 - 2017.48                       # mid-2020 in mission years
    k20 = (kc[0] + kc[1]*np.cos(w*ty20) + kc[2]*np.sin(w*ty20)
           + kc[3]*np.cos(2*w*ty20) + kc[4]*np.sin(2*w*ty20))
    fitx = np.linspace(0, 55, 100)
    # C(mlat) = C_eq + (C_eq - C0)*k*max(0,|mlat|-20)^2 ; C_eq = base
    ax.plot(fitx, base + (base - C0) * k20 * np.maximum(0, fitx - 20) ** 2, "r--",
            label=fr"calib (k(2020)={k20:.5f}, $C_0$={C0:.1f})")
    ax.axvline(20, color="gray", ls=":", lw=1)
    ax.set_xlabel("|mlat| (deg)"); ax.set_ylabel("residual ~ C (cnt/s)")
    ax.set_title("A. mlat factor (time fixed = 2020)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # --- B: g(t), fix equator |mlat|<5 ---
    ax = axes[1]
    eq = amlat < 5
    tb = np.linspace(0, 9, 28)
    tcen = 0.5 * (tb[:-1] + tb[1:])
    eqmed = np.array([np.median(res[eq & (ty >= tb[i]) & (ty < tb[i+1])])
                      if (eq & (ty >= tb[i]) & (ty < tb[i+1])).sum() > 100 else np.nan
                      for i in range(len(tcen))])
    norm0 = np.nanmedian(eqmed[:3])
    ax.plot(2017.48 + tcen, eqmed / norm0, "o-", color="#2ca02c", label="equatorial residual (norm)")
    gfit = g_A + (1 - g_A) * np.exp(-tcen / g_tau)
    ax.plot(2017.48 + tcen, gfit, "r--", label=f"calib $g(t)$")
    ax.set_xlabel("year"); ax.set_ylabel("residual / residual(t0)")
    ax.set_title("B. PMT outgassing g(t) (equator fixed)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0.5, 1.2)

    # --- C: k(t), robust LS slope per half-year bin (all high-mlat data, C0-corrected) ---
    ax = axes[2]
    C0c = float(cz["C0"])
    tb2 = np.linspace(0, 9, 19)          # half-year bins
    tc2 = 0.5 * (tb2[:-1] + tb2[1:])
    keff, kx = [], []
    for i in range(len(tc2)):
        mt = (ty >= tb2[i]) & (ty < tb2[i + 1])
        eq = res[mt & (amlat < 5)]
        hm = mt & (amlat >= 25)
        if len(eq) < 100 or hm.sum() < 300:
            continue
        eqm = np.median(eq)              # C_eq = s0*g + C0
        w2 = np.maximum(0.0, amlat[hm] - 20) ** 2
        yv = (res[hm] - C0c) / (eqm - C0c) - 1.0   # = k * w^2  (C0 removed)
        g2 = w2 > 25
        if g2.sum() < 200:
            continue
        kfit = np.sum(w2[g2] * yv[g2]) / np.sum(w2[g2] ** 2)  # LS slope through origin
        keff.append(kfit); kx.append(2017.48 + tc2[i])
    kx = np.array(kx); keff = np.array(keff)
    ax.plot(kx, keff, "o-", color="#ff7f0e", lw=1.6, label="k (LS slope, half-year)")
    tcc = kx - 2017.48
    kcal = kc[0] + kc[1]*np.cos(w*tcc) + kc[2]*np.sin(w*tcc) + kc[3]*np.cos(2*w*tcc) + kc[4]*np.sin(2*w*tcc)
    ax.plot(kx, kcal, "r--", lw=2.0, label="calib $k(t)$")
    rk = np.corrcoef(keff, kcal)[0, 1] if len(keff) > 3 else np.nan
    ax.set_xlabel("year"); ax.set_ylabel("effective k")
    ax.set_title(f"C. solar-cycle k(t)  (LS slope, half-year;  r={rk:.2f})", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    print(f"\nC. k(t): half-year LS vs calib  r={rk:.3f},  k range {np.min(keff):.5f}-{np.max(keff):.5f}")

    fig.suptitle("Factor-isolation study: each factor with the other two fixed", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/diag_factor_isolation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved plots/diag_factor_isolation.png")

    # numeric signal strength
    print("\n=== signal strength ===")
    print(f"A. mlat: residual(|mlat|45) / residual(eq) = {np.nanmedian(res[(year==2020)&(amlat>=43)&(amlat<47)])/base:.2f}x")
    print(f"B. g(t): equator residual change {(1-np.nanmin(eqmed)/norm0)*100:.0f}% over mission")
    print(f"C. k(t): effective k range {np.nanmin(keff):.5f} - {np.nanmax(keff):.5f} ({(np.nanmax(keff)/np.nanmin(keff)-1)*100:.0f}% swing)")


if __name__ == "__main__":
    main()
