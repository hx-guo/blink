#!/usr/bin/env python3
"""Check HXMT pointing during 2020-10-10..16 anomaly hours.

If HXMT was doing target-of-opportunity follow-up of magnetar SGR J1830-0645
(RA=277.5°, Dec=-6.75°), the pointing direction during anomaly hours should
cluster near those coordinates.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("n_below_study/clean_relaxed_2020_sample05.parquet")

# Target sources to check
SGR_J1830_RA, SGR_J1830_DEC = 277.5, -6.75   # the new magnetar
FOURU_2206_RA, FOURU_2206_DEC = 331.98, 54.52  # 4U 2206+54 (HXMT observation paper mentions Oct 10-11)


def angular_sep(ra1, dec1, ra2, dec2):
    """Angular separation in degrees between two points on sphere."""
    ra1r = np.radians(ra1); dec1r = np.radians(dec1)
    ra2r = np.radians(ra2); dec2r = np.radians(dec2)
    cos_sep = np.sin(dec1r)*np.sin(dec2r) + np.cos(dec1r)*np.cos(dec2r)*np.cos(ra1r-ra2r)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sep))


def stats(name, x):
    if len(x) == 0:
        return f"{name}: empty"
    return (f"{name}: n={len(x):>10,}  median={np.median(x):>9.2f}  "
            f"Q05={np.quantile(x, 0.05):>9.2f}  Q95={np.quantile(x, 0.95):>9.2f}")


def main():
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df):,} rows")
    print(f"  Available pointing cols: Ra, Dec, Delta_Ra, Delta_Dec, Delta")

    pho = df["PHO"].astype("float64").values
    wide = df["Wide"].astype("float64").values
    wide_pho = np.where(pho > 0, wide / pho, np.nan)

    ra = df["Ra"].values
    dec = df["Dec"].values

    # Sanity: print Ra/Dec range
    print(f"\nRa range: {np.nanmin(ra):.1f} to {np.nanmax(ra):.1f}")
    print(f"Dec range: {np.nanmin(dec):.1f} to {np.nanmax(dec):.1f}")

    # Compute angular separation to SGR J1830 and 4U 2206
    sep_sgr = angular_sep(ra, dec, SGR_J1830_RA, SGR_J1830_DEC)
    sep_4u = angular_sep(ra, dec, FOURU_2206_RA, FOURU_2206_DEC)

    # Anomaly window
    in_week = ((df["date"] >= "2020-10-10") & (df["date"] <= "2020-10-16")).values
    anom = in_week & (wide_pho > 0.3) & np.isfinite(wide_pho)
    nonanom = in_week & (wide_pho <= 0.3) & np.isfinite(wide_pho)

    print(f"\n2020-10-10..16 week:")
    print(f"  Anom (W/P>0.3):    {anom.sum():>8,}")
    print(f"  Non-anom (W/P≤0.3): {nonanom.sum():>8,}")

    # === Pointing of anomaly seconds vs non-anomaly ===
    print("\n=== Pointing during 2020-10-10..16 ===")
    print(stats("Ra Anom",      ra[anom]))
    print(stats("Ra Non-anom",  ra[nonanom]))
    print(stats("Dec Anom",     dec[anom]))
    print(stats("Dec Non-anom", dec[nonanom]))

    # === Distance to SGR J1830 ===
    print(f"\n=== Angular separation to SGR J1830-0645 (RA={SGR_J1830_RA}°, Dec={SGR_J1830_DEC}°) ===")
    print(stats("Sep Anom",     sep_sgr[anom]))
    print(stats("Sep Non-anom", sep_sgr[nonanom]))
    print(f"  Anom rows within 5° of SGR J1830:  {((sep_sgr < 5) & anom).sum():>8,}  ({((sep_sgr < 5) & anom).sum() / max(anom.sum(), 1) * 100:.2f}%)")
    print(f"  Anom rows within 10° of SGR J1830: {((sep_sgr < 10) & anom).sum():>8,}  ({((sep_sgr < 10) & anom).sum() / max(anom.sum(), 1) * 100:.2f}%)")

    # === Distance to 4U 2206+54 ===
    print(f"\n=== Angular separation to 4U 2206+54 (RA={FOURU_2206_RA}°, Dec={FOURU_2206_DEC}°) ===")
    print(stats("Sep Anom",     sep_4u[anom]))
    print(stats("Sep Non-anom", sep_4u[nonanom]))
    print(f"  Anom rows within 5° of 4U 2206:    {((sep_4u < 5) & anom).sum():>8,}  ({((sep_4u < 5) & anom).sum() / max(anom.sum(), 1) * 100:.2f}%)")

    # === Show top pointing centers during anomaly ===
    print("\n=== Top 10 most-frequent (Ra, Dec) bins for anomaly seconds ===")
    anom_df = df.loc[anom, ["Ra", "Dec"]].copy()
    anom_df["Ra_bin"] = (anom_df["Ra"] / 5).round() * 5
    anom_df["Dec_bin"] = (anom_df["Dec"] / 5).round() * 5
    top_anom = anom_df.groupby(["Ra_bin", "Dec_bin"]).size().sort_values(ascending=False).head(10)
    print(top_anom.to_string())

    print("\n=== Top 10 most-frequent (Ra, Dec) bins for non-anomaly seconds in same week ===")
    nonanom_df = df.loc[nonanom, ["Ra", "Dec"]].copy()
    nonanom_df["Ra_bin"] = (nonanom_df["Ra"] / 5).round() * 5
    nonanom_df["Dec_bin"] = (nonanom_df["Dec"] / 5).round() * 5
    top_nonanom = nonanom_df.groupby(["Ra_bin", "Dec_bin"]).size().sort_values(ascending=False).head(10)
    print(top_nonanom.to_string())

    # === Per-day pointing during 2020-10-10..16 ===
    print("\n=== Per-day pointing during anomaly week ===")
    for d in ["2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16"]:
        m = (df["date"] == d).values
        if m.sum() == 0:
            continue
        ra_d = ra[m]
        dec_d = dec[m]
        anom_m = m & (wide_pho > 0.3) & np.isfinite(wide_pho)
        nonanom_m = m & (wide_pho <= 0.3) & np.isfinite(wide_pho)
        print(f"\n  {d}: total={m.sum()}, anom={anom_m.sum()}, non-anom={nonanom_m.sum()}")
        if anom_m.sum() > 0:
            print(f"    Anom Ra med={np.median(ra[anom_m]):.1f}, Dec med={np.median(dec[anom_m]):.1f}, "
                  f"sep_SGR med={np.median(sep_sgr[anom_m]):.1f}°")
        if nonanom_m.sum() > 0:
            print(f"    Non-anom Ra med={np.median(ra[nonanom_m]):.1f}, Dec med={np.median(dec[nonanom_m]):.1f}, "
                  f"sep_SGR med={np.median(sep_sgr[nonanom_m]):.1f}°")


if __name__ == "__main__":
    main()
