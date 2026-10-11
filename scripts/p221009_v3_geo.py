#!/usr/bin/env python3
"""GRB 221009A v3: when is the burst occulted by the Earth for Insight-HXMT?

The v2 freeze quoted T0+1698 s (HXMT geometric limb, p221009_v2_bkg.orbit) and
the background calibration window started at T0+1780 s; Zhang+2025 and An+2023
quote T0+1884 s for HXMT and T0+1780 s for GECAM-C. This script recomputes the
limb crossing from the ephemerides, for several Earth-limb definitions, and
validates the method on GECAM-C (whose posatt file has both J2000 and WGS84
positions, so the frame handling is checked as well). It also computes when
each spacecraft enters the Earth's (cylindrical) shadow from the Sun, since
"entered the Earth's shadow" is the wording of both papers.

Outputs: data/p221009/v3/geo.txt, geo.csv
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, ITRS, GCRS, CartesianRepresentation, get_sun
import astropy.units as u

OUT = Path("data/p221009/v3")
OUT.mkdir(parents=True, exist_ok=True)
T0_UTC = "2022-10-09T13:17:00"
T0_HXMT = 339945423.0
T0_GECAM = 55862220.0
ORBIT = "data/hxmt_aux/HXMT_20221009T13_Orbit_FFFFFF_V1_1K.FITS"
GC_POSATT = "data/p221009/timing_in/gc_posatt_221009_13_v07.fits"
RA, DEC = 288.265, 19.773
R_EQ, R_POL = 6378.137e3, 6356.752e3
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def grb_unit(frame_times=None, itrs=False):
    c = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg, distance=1e9 * u.pc, frame="icrs")
    if not itrs:
        v = c.cartesian.xyz.value
        return (v / np.linalg.norm(v))[:, None]
    v = c.transform_to(ITRS(obstime=frame_times)).cartesian.xyz.value
    return v / np.linalg.norm(v, axis=0)


def elevation(pos, uvec, h_atm, oblate=False):
    """Angle of the source above the Earth limb (deg), Earth radius + h_atm.
    oblate: use the local geocentric radius of the ellipsoid along the
    tangent direction (approximated by the radius at the sub-satellite
    latitude), otherwise the equatorial radius."""
    r = np.linalg.norm(pos, axis=0)
    if oblate:
        lat = np.arcsin(pos[2] / r)
        Re = R_EQ * R_POL / np.sqrt((R_POL * np.cos(lat)) ** 2 + (R_EQ * np.sin(lat)) ** 2)
    else:
        Re = R_EQ
    nad = np.degrees(np.arccos(np.clip(-(pos * uvec).sum(0) / r, -1, 1)))
    return nad - np.degrees(np.arcsin((Re + h_atm) / r))


def tangent_altitude(pos, uvec):
    """Minimum altitude (km, above the equatorial sphere) of the line of sight."""
    r = np.linalg.norm(pos, axis=0)
    cosz = (pos * uvec).sum(0) / r          # >0: source above local horizontal
    d = r * np.sqrt(np.clip(1 - cosz ** 2, 0, None))
    return np.where(cosz > 0, (r - R_EQ) / 1e3, (d - R_EQ) / 1e3)


def crossing(t, y):
    """Times where y changes sign from + to - (linear interpolation)."""
    i = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    return t[i] + y[i] / (y[i] - y[i + 1]) * (t[i + 1] - t[i])


def in_shadow(pos_gcrs, times):
    """Cylindrical umbra test (Sun at infinity)."""
    s = get_sun(times).cartesian.xyz.to(u.m).value
    s = s / np.linalg.norm(s, axis=0)
    proj = (pos_gcrs * s).sum(0)
    perp = np.linalg.norm(pos_gcrs - proj * s, axis=0)
    return (proj < 0) & (perp < R_EQ)


def main():
    rows = []
    # ---------- GECAM-C (validation: An+2023 quotes T0+1780 s) ----------
    g = fits.open(GC_POSATT)[1].data
    tg = g["TIME"] - T0_GECAM
    pj = np.vstack([g["X_J2000"], g["Y_J2000"], g["Z_J2000"]]).astype(float)
    pw = np.vstack([g["X_WGS84"], g["Y_WGS84"], g["Z_WGS84"]]).astype(float)
    tt = Time(T0_UTC, scale="utc") + TimeDelta(tg, format="sec")
    say("== GECAM-C (posatt v07) ==")
    for h in (0.0, 50e3, 100e3):
        ej = elevation(pj, grb_unit(), h)
        ew = elevation(pw, grb_unit(tt, itrs=True), h)
        cj, cw = crossing(tg, ej), crossing(tg, ew)
        say(f"  limb + {h/1e3:4.0f} km: ingress (J2000 frame) {np.round(cj, 1)}   (WGS84 frame) {np.round(cw, 1)}")
        rows.append(("GECAM-C", f"limb+{h/1e3:.0f}km", cj[0] if len(cj) else np.nan))
    sh = in_shadow(pj, tt)
    ish = tg[np.where(~sh[:-1] & sh[1:])[0] + 1] if (~sh[:-1] & sh[1:]).any() else []
    say(f"  enters Earth's shadow (Sun) at T0+{np.round(ish, 1)} s")
    say(f"  An+2023 Methods: GECAM-C occultation ingress T0+1780 s")
    rows.append(("GECAM-C", "sun shadow", ish[0] if len(ish) else np.nan))

    # ---------- HXMT ----------
    o = fits.open(ORBIT)[1].data
    th = o["Time"] - T0_HXMT
    pe = np.vstack([o["X"], o["Y"], o["Z"]]).astype(float)
    lon_xyz = np.degrees(np.arctan2(pe[1], pe[0]))
    say("\n== Insight-HXMT (Orbit 1K, X/Y/Z) ==")
    say(f"  frame check: atan2(Y,X) - Lon column, max |diff| = {np.max(np.abs(((lon_xyz - o['Lon'] + 180) % 360) - 180)):.4f} deg"
        "  -> X/Y/Z are Earth-fixed (WGS84/ITRS)")
    th_s = th[::5]
    ts = Time(T0_UTC, scale="utc") + TimeDelta(th_s, format="sec")
    ui = grb_unit(ts, itrs=True)
    pe_s = pe[:, ::5]
    # GCRS position of HXMT (for the shadow test and an inertial cross-check)
    itrs = ITRS(CartesianRepresentation(pe_s * u.m), obstime=ts)
    pg = itrs.transform_to(GCRS(obstime=ts)).cartesian.xyz.to(u.m).value
    for h in (0.0, 50e3, 100e3):
        for obl in (False, True):
            e1 = elevation(pe_s, ui, h, oblate=obl)
            c1 = crossing(th_s, e1)
            tag = f"limb+{h/1e3:.0f}km{' (ellipsoid)' if obl else ''}"
            say(f"  {tag:22s} ingress T0+{np.round(c1, 1)} s")
            rows.append(("HXMT", tag, c1[0] if len(c1) else np.nan))
    e_in = elevation(pg, grb_unit(), 0.0)
    say(f"  inertial cross-check (GCRS position, ICRS source), geometric limb: T0+{np.round(crossing(th_s, e_in), 1)} s")
    ta = tangent_altitude(pe_s, ui)
    for alt in (0, -50, -100):
        c = crossing(th_s, ta - alt)
        say(f"  line-of-sight tangent altitude reaches {alt:+d} km at T0+{np.round(c, 1)} s")
    sh = in_shadow(pg, ts)
    ish = th_s[np.where(~sh[:-1] & sh[1:])[0] + 1]
    say(f"  HXMT enters Earth's shadow (Sun) at T0+{np.round(ish, 1)} s")
    rows.append(("HXMT", "sun shadow", ish[0] if len(ish) else np.nan))
    for t in (1600, 1698, 1780, 1884, 1930, 2000):
        i = np.argmin(np.abs(th_s - t))
        say(f"  T0+{t}: GRB elevation above geometric limb {elevation(pe_s[:, i:i+1], ui[:, i:i+1], 0.0)[0]:+6.2f} deg, "
            f"line-of-sight tangent altitude {ta[i]:7.0f} km, in Sun shadow {bool(sh[i])}")
    say("  Zhang+2025 / An+2023: HXMT 'entered the Earth's shadow' at T0+1884 s (their T0 is 0.05 s later);")
    say("  An+2023 Methods: 'earth eclipse time of Insight-HXMT observing GRB 221009A is 1883-3864 s'")
    import csv
    with open(OUT / "geo.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spacecraft", "definition", "t_rel_T0"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.1f}"])
    (OUT / "geo.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
