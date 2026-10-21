#!/usr/bin/env python3
"""Fit B(|mlat|) — the cosmic ray secondary background term as a function of
AACGM magnetic latitude. Try several functional forms.

Output: plots/B_mlat_fit.png
"""
from __future__ import annotations

import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import aacgmv2

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from unwrap_large_v2 import unwrap_large_v2, CONF_LOW

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")
OUT_PNG = Path("plots/B_mlat_fit.png")
L_CYCLES_TO_SEC = 16e-6

USE_COLS = ["date", "box", "det", "L_cycles", "Dt", "PHO", "Large", "Wide", "Sci_1s", "Lat", "Lon"]


def build_aacgm_grid(date=datetime.datetime(2020, 6, 15), alt_km=540.0):
    print(f"Building AACGM grid for {date.date()} at alt={alt_km}km...")
    lat_grid = np.linspace(-45, 45, 91)
    lon_grid = np.linspace(0, 360, 181)
    mlat_arr = np.zeros((len(lat_grid), len(lon_grid)))
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):  # silence "undefined near equator"
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                mlat, _, _ = aacgmv2.get_aacgm_coord(float(lat), float(lon), alt_km, date)
                mlat_arr[i, j] = mlat if not np.isnan(mlat) else 0.0
    return lat_grid, lon_grid, mlat_arr


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE, columns=USE_COLS)
    print(f"  rows: {len(df):,}")

    lat_grid, lon_grid, mlat_grid = build_aacgm_grid()
    interp = RegularGridInterpolator((lat_grid, lon_grid), mlat_grid,
                                      bounds_error=False, fill_value=np.nan)
    print("Interpolating mlat for each row...")
    pts = np.column_stack([df["Lat"].values, df["Lon"].values])
    mlat = interp(pts)
    abs_mlat = np.abs(mlat)

    pho = df["PHO"].values; large_raw = df["Large"].values
    wide = df["Wide"].values; sci = df["Sci_1s"].values.astype("float64")
    lc = df["L_cycles"].values; dtv = df["Dt"].values
    L = lc.astype("float64") * L_CYCLES_TO_SEC
    lf = 1.0 - dtv.astype("float64") / lc.astype("float64")

    large_corr, conf = unwrap_large_v2(pho, large_raw, wide, sci, lc, dtv, C=195.0,
                                        return_confidence=True)
    n_wraps = ((large_corr - large_raw.astype("float64")) / 1024).round().astype(int)
    base = ((pho.astype("float64") - large_corr) * lf - wide.astype("float64")) / L
    residual = (base - sci).values if hasattr(base, "values") else (base - sci)

    wide_pho = wide / np.maximum(pho, 1)
    is_clean = ((conf > CONF_LOW) & (wide_pho < 0.3) & (sci > 100)
                & (n_wraps == 0) & np.isfinite(residual) & ~np.isnan(mlat))
    print(f"  Clean rows: {is_clean.sum():,}")

    # Per-det baseline at |mlat|<5°
    is_eq = is_clean & (abs_mlat < 5)
    C_det_per_row = np.zeros(len(df))
    for box in "ABC":
        for det in range(6):
            m_dt = ((df["box"] == box) & (df["det"] == det)).values
            m = m_dt & is_eq
            cval = float(np.mean(residual[m])) if m.sum() > 100 else 120.0
            C_det_per_row[m_dt] = cval
    B_per_row = residual - C_det_per_row

    # Fine binning by |mlat|, fit functions
    edges = np.linspace(0, 45, 22)  # 2° bins
    centers = 0.5 * (edges[:-1] + edges[1:])

    B_mean = np.full(len(centers), np.nan)
    B_err = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = is_clean & (abs_mlat >= edges[i]) & (abs_mlat < edges[i + 1])
        if m.sum() < 500:
            continue
        B_mean[i] = float(np.mean(B_per_row[m]))
        B_err[i] = float(np.std(B_per_row[m])) / np.sqrt(m.sum())

    valid = ~np.isnan(B_mean)
    x = centers[valid]
    y = B_mean[valid]
    yerr = np.maximum(B_err[valid], 0.5)

    # Try several functional forms:
    # 1. Polynomial 2nd order
    p2 = np.polyfit(x, y, 2)
    y_p2 = np.polyval(p2, centers)
    chi2_p2 = np.sum(((y - np.polyval(p2, x)) / yerr)**2) / (len(x) - 3)

    # 2. Polynomial 3rd order
    p3 = np.polyfit(x, y, 3)
    y_p3 = np.polyval(p3, centers)
    chi2_p3 = np.sum(((y - np.polyval(p3, x)) / yerr)**2) / (len(x) - 4)

    # 3. Power law: B = A · |mlat|^n (with small offset to handle near-zero)
    def power_law(lat, A, n):
        return A * np.maximum(lat, 0.01)**n
    try:
        popt_pl, _ = curve_fit(power_law, x[x > 5], y[x > 5], p0=[0.001, 3], maxfev=5000)
        y_pl = power_law(centers, *popt_pl)
        chi2_pl = np.sum(((y[x > 5] - power_law(x[x > 5], *popt_pl)) / yerr[x > 5])**2) / (len(x[x > 5]) - 2)
    except Exception as e:
        print(f"  Power law fit failed: {e}")
        popt_pl = None; chi2_pl = np.nan

    # 4. sin²-form: B = A · sin²(|mlat|·factor)
    def sin2_form(lat, A, factor):
        return A * np.sin(np.radians(lat * factor))**2
    try:
        popt_sin, _ = curve_fit(sin2_form, x, y, p0=[200, 1.0], maxfev=5000)
        y_sin = sin2_form(centers, *popt_sin)
        chi2_sin = np.sum(((y - sin2_form(x, *popt_sin)) / yerr)**2) / (len(x) - 2)
    except Exception as e:
        popt_sin = None; chi2_sin = np.nan

    # 5. Stoermer-inspired: B = A · (1/cos^n(mlat·π/180) - 1)
    def stoermer(lat, A, n):
        cosval = np.maximum(np.cos(np.radians(lat)), 0.1)
        return A * (cosval**(-n) - 1)
    try:
        popt_st, _ = curve_fit(stoermer, x, y, p0=[1.0, 10.0], maxfev=5000)
        y_st = stoermer(centers, *popt_st)
        chi2_st = np.sum(((y - stoermer(x, *popt_st)) / yerr)**2) / (len(x) - 2)
    except Exception as e:
        popt_st = None; chi2_st = np.nan

    # 6. Simple piecewise quadratic: B = a · max(0, |mlat| - threshold)²
    def piecewise(lat, a, threshold):
        return a * np.maximum(0, lat - threshold)**2
    try:
        popt_pw, _ = curve_fit(piecewise, x, y, p0=[0.3, 22.0], maxfev=5000)
        chi2_pw = np.sum(((y - piecewise(x, *popt_pw)) / yerr)**2) / (len(x) - 2)
    except Exception as e:
        popt_pw = None; chi2_pw = np.nan

    # Print fits
    print("\n=== Fit comparison ===")
    print(f"  Polynomial 2nd: {p2[0]:+.5f}·x² + {p2[1]:+.5f}·x + {p2[2]:+.2f}   χ²/dof = {chi2_p2:.2f}")
    print(f"  Polynomial 3rd: {p3[0]:+.6f}·x³ + {p3[1]:+.5f}·x² + {p3[2]:+.5f}·x + {p3[3]:+.2f}   χ²/dof = {chi2_p3:.2f}")
    if popt_pl is not None:
        print(f"  Power law: {popt_pl[0]:.6f} · |mlat|^{popt_pl[1]:.2f}   χ²/dof = {chi2_pl:.2f}")
    if popt_sin is not None:
        print(f"  sin² form: {popt_sin[0]:.2f} · sin²({popt_sin[1]:.3f}·mlat)   χ²/dof = {chi2_sin:.2f}")
    if popt_st is not None:
        print(f"  Stoermer:  {popt_st[0]:.4f} · (cos^{popt_st[1]:.2f}(mlat) − 1)/(...)   χ²/dof = {chi2_st:.2f}")
        # cleaner display
        print(f"             {popt_st[0]:.3f} · (1/cos^{popt_st[1]:.2f}(mlat) − 1)   χ²/dof = {chi2_st:.2f}")
    if popt_pw is not None:
        print(f"  PIECEWISE: {popt_pw[0]:.4f} · max(0, |mlat| − {popt_pw[1]:.2f}°)²   χ²/dof = {chi2_pw:.2f}  ← SIMPLE")

    # === Plot ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    ax.errorbar(x, y, yerr=yerr, fmt="ko", markersize=6, lw=1.5, capsize=3, label="data")
    xx = np.linspace(0, 45, 200)
    ax.plot(xx, np.polyval(p2, xx), "b-", lw=1.5, alpha=0.7, label=f"poly 2nd  (χ²/dof={chi2_p2:.1f})")
    ax.plot(xx, np.polyval(p3, xx), "g-", lw=2.0, label=f"poly 3rd  (χ²/dof={chi2_p3:.1f})")
    if popt_pl is not None:
        ax.plot(xx, power_law(xx, *popt_pl), "r-", lw=1.5, alpha=0.7,
                 label=fr"power law: {popt_pl[0]:.3g}·|mlat|^{popt_pl[1]:.2f}  (χ²/dof={chi2_pl:.1f})")
    if popt_sin is not None:
        ax.plot(xx, sin2_form(xx, *popt_sin), "m--", lw=1.5,
                 label=fr"sin²: {popt_sin[0]:.0f}·sin²({popt_sin[1]:.2f}·mlat)  (χ²/dof={chi2_sin:.1f})")
    if popt_st is not None:
        ax.plot(xx, stoermer(xx, *popt_st), "c-", lw=1.5, alpha=0.5,
                 label=fr"Stoermer: {popt_st[0]:.2f}·(1/cos^{popt_st[1]:.1f}(mlat) − 1)  (χ²/dof={chi2_st:.1f})")
    if popt_pw is not None:
        ax.plot(xx, piecewise(xx, *popt_pw), color="orange", lw=3.0,
                 label=fr"$\bf{{piecewise}}$: {popt_pw[0]:.2f}·max(0, |mlat|−{popt_pw[1]:.1f}°)²  (χ²/dof={chi2_pw:.1f})  ← SIMPLEST")
    ax.axhline(0, color="k", ls=":", lw=0.7)
    ax.set_xlabel("|AACGM mlat| (deg)", fontsize=12)
    ax.set_ylabel("B (cnt/s, environmental background above equatorial baseline)", fontsize=12)
    ax.set_title("B(|mlat|) — cosmic ray secondary background term, with fits", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 45)
    ax.set_ylim(-20, 200)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")

    print("\n=== Numerical table ===")
    print(f"  {'|mlat|':<8}{'B (data)':>10}{'std/√N':>10}")
    for i in range(len(centers)):
        if valid[i]:
            print(f"  {centers[i]:>5.1f}°    {B_mean[i]:>+8.1f}   {B_err[i]:>+6.1f}")


if __name__ == "__main__":
    main()
