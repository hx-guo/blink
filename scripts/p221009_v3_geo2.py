#!/usr/bin/env python3
"""GRB 221009A v3: probe of the ~185 s offset between our HXMT occultation
(T0+1699..3686 s from the 1K Orbit file) and the literature (1883..3864 s).

1. What the HXMT orbit product is: header keywords of the 1K Orbit files
   (time system, MJDREF, units, creator) and the columns of the 1K Att file.
2. Independent time check of the orbit file from the data themselves: the HE
   particle background is a function of geomagnetic position, so the ACD-tag
   rate and Cnt_LargeEvt must line up with the dipole L computed from the
   orbit at zero lag if the orbit time axis is right. We scan a lag of
   -600..+600 s (L(t + lag)) and record the rms of a quadratic background-vs-L
   relation over T12-T14 (burst interval T0-10..+1000 s excluded). A 185 s
   error of the orbit time would put the minimum at |lag| ~ 185 s.
3. The same lag test on GECAM-C (posatt v07 + T14 file; LG background rate
   from BTIME outside the burst) as a control.
4. GECAM-C occultation interval (ingress and egress) from posatt T13+T14, to
   test whether the literature "HXMT" 1883-3864 s is in fact GECAM-C's.

Output: data/p221009/v3/geo2.txt
"""
import glob
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from p221009_v3_geo import elevation, grb_unit, crossing  # noqa: E402

OUT = Path("data/p221009/v3")
T0 = 339945423.0
T0G = 55862220.0
K1 = "data/1K/Y202210/20221009-1943"
POSATT_DIR = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily/2022/10/09/POSATT"
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def dipole_L(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    pl, pn = np.radians(80.65), np.radians(-72.68)
    m = np.array([np.cos(pl) * np.cos(pn), np.cos(pl) * np.sin(pn), np.sin(pl)])
    s = (m[0] * x + m[1] * y + m[2] * z) / r
    return (r / 6371.2e3) / (1 - s * s)


def lag_scan(t_obs, y, t_orb, L, lags, deg=2):
    """rms of y - poly(L(t+lag)) for each lag."""
    out = []
    for lag in lags:
        Ls = np.interp(t_obs + lag, t_orb, L)
        c = np.polyfit(Ls, y, deg)
        out.append(np.sqrt(np.mean((y - np.polyval(c, Ls)) ** 2)))
    return np.array(out)


def refine(lags, rms):
    i = int(np.argmin(rms))
    if 0 < i < len(lags) - 1:
        a, b, c = rms[i - 1], rms[i], rms[i + 1]
        return lags[i] + 0.5 * (a - c) / (a - 2 * b + c) * (lags[1] - lags[0])
    return lags[i]


def main():
    # ---------- 1. the HXMT orbit product ----------
    say("== HXMT orbit product ==")
    for h in ("13",):
        fn = f"{K1}/HXMT_20221009T{h}_Orbit_FFFFFF_V1_1K.FITS"
        with fits.open(fn) as f:
            for ext in (0, 1):
                hd = f[ext].header
                keys = [k for k in ("TELESCOP", "INSTRUME", "ORIGIN", "CREATOR", "DATE", "FILENAME",
                                    "TIMESYS", "TIMEREF", "TIMEUNIT", "MJDREFI", "MJDREFF", "MJDREF",
                                    "TIMEZERO", "TSTART", "TSTOP", "DATE-OBS", "DATE-END", "EXTNAME") if k in hd]
                say(f"  {Path(fn).name} [{ext}] " + "  ".join(f"{k}={hd[k]}" for k in keys))
            d = f[1].data
            say(f"  columns {f[1].columns.names}; first Time {d['Time'][0]:.3f} = T0{d['Time'][0] - T0:+.3f} s, "
                f"step {np.median(np.diff(d['Time'])):.3f} s")
    att = sorted(glob.glob(f"{K1}/HXMT_20221009T13_Att_*1K.FITS"))
    if att:
        with fits.open(att[0]) as f:
            say(f"  {Path(att[0]).name}: extensions {[x.name for x in f]}; columns "
                f"{f[1].columns.names}; TIMESYS={f[1].header.get('TIMESYS')}")
    say("  no other HXMT orbit product (L1P / 1T) for 2022-10-09 was found on the farm "
        "(1L holds per-ObsID L1 data not indexed by date; 1T has no 2022 entries); the independent "
        "checks below therefore test the 1K Orbit time axis against the HE data themselves")

    # ---------- 2. lag test, HXMT ----------
    o = [fits.open(f"{K1}/HXMT_20221009T{h}_Orbit_FFFFFF_V1_1K.FITS")[1].data for h in ("12", "13", "14")]
    to = np.concatenate([x["Time"] for x in o]) - T0
    L = dipole_L(*(np.concatenate([x[c] for x in o]).astype(float) for c in ("X", "Y", "Z")))
    H = np.load(OUT / "hours_1s.npz")
    t = H["t"] + 0.5
    from scipy.ndimage import uniform_filter1d, binary_dilation
    usable = np.isfinite(H["f"]) & np.isfinite(H["LE"]) & (H["n"] > 3000)
    usable &= ~binary_dilation(~usable, iterations=60)
    usable &= ~((t > -10) & (t < 1000))
    lags = np.arange(-600, 601, 5.0)
    say("\n== orbit-time test: HE background tracers vs dipole L(t + lag), T12-T14, burst excluded ==")
    for name, y in (("ACD-tag rate", uniform_filter1d(H["acd"].astype(float), 21)),
                    ("Cnt_LargeEvt", uniform_filter1d(np.nan_to_num(H["LE"]), 21)),
                    ("1K rate", uniform_filter1d(H["n"].astype(float), 21))):
        for seg, m in (("all usable", usable), ("T12 only", usable & (t < -1020)),
                       ("T14 only", usable & (t > 2580))):
            rms = lag_scan(t[m], y[m], to, L, lags)
            best = refine(lags, rms)
            r0 = rms[lags == 0][0]
            r185 = np.interp(185, lags, rms)
            rm185 = np.interp(-185, lags, rms)
            say(f"  {name:13s} {seg:11s}: best lag {best:+6.1f} s; rms at 0 / +185 / -185 s = "
                f"{r0:7.1f} / {r185:7.1f} / {rm185:7.1f} ({m.sum()} s)")

    # ---------- 3. GECAM-C control and 4. occultation interval ----------
    pa = [f"{POSATT_DIR}/gc_posatt_221009_13_v07.fits"]
    p14 = sorted(glob.glob(f"{POSATT_DIR}/gc_posatt_221009_14_v*.fits"))
    p12 = sorted(glob.glob(f"{POSATT_DIR}/gc_posatt_221009_12_v*.fits"))
    files = ([p12[-1]] if p12 else []) + pa + ([p14[-1]] if p14 else [])
    g = [fits.open(f)[1].data for f in files]
    tg = np.concatenate([x["TIME"] for x in g]) - T0G
    order = np.argsort(tg)
    tg = tg[order]
    pj = np.vstack([np.concatenate([x[c] for x in g])[order] for c in ("X_J2000", "Y_J2000", "Z_J2000")]).astype(float)
    pw = np.vstack([np.concatenate([x[c] for x in g])[order] for c in ("X_WGS84", "Y_WGS84", "Z_WGS84")]).astype(float)
    keep = np.concatenate([[True], np.diff(tg) > 0])
    tg, pj, pw = tg[keep], pj[:, keep], pw[:, keep]
    say(f"\n== GECAM-C posatt ({', '.join(Path(f).name for f in files)}), T0{tg[0]:+.0f}..{tg[-1]:+.0f} s ==")
    e = elevation(pj, grb_unit(), 0.0)
    ing = crossing(tg, e)
    egr = crossing(tg, -e)
    say(f"  geometric occultation: ingress T0+{np.round(ing, 1)} s, egress T0+{np.round(egr, 1)} s")
    if len(ing) and len(egr):
        say(f"  -> interval {ing[0]:.0f}-{egr[egr > ing[0]][0]:.0f} s, duration "
            f"{egr[egr > ing[0]][0] - ing[0]:.0f} s")
    e100 = elevation(pj, grb_unit(), 100e3)
    say(f"  limb + 100 km: ingress {np.round(crossing(tg, e100), 1)}, egress {np.round(crossing(tg, -e100), 1)}")
    # HXMT interval from the same orbit set, for the side-by-side
    from astropy.time import Time, TimeDelta
    po = np.vstack([np.concatenate([x[c] for x in o]).astype(float) for c in ("X", "Y", "Z")])
    sub = slice(None, None, 5)
    ts = Time("2022-10-09T13:17:00", scale="utc") + TimeDelta(to[sub], format="sec")
    eh = elevation(po[:, sub], grb_unit(ts, itrs=True), 0.0)
    say(f"  HXMT (1K Orbit T12-T14), geometric: ingress {np.round(crossing(to[sub], eh), 1)}, "
        f"egress {np.round(crossing(to[sub], -eh), 1)}")
    # GECAM-C lag control: LG background (BTIME, ch9-12) vs its own dipole L
    try:
        b = fits.open("data/gecam_c/gcg_btime_221009_13_v12.fits")["SPECTRUM01"].data
        tb = b["STARTTIME"] - T0G
        cb = b["COUNTS"][:, 9:13].sum(1).astype(float)
        e1 = np.arange(np.floor(tb[0]), np.ceil(tb[-1]))
        y = np.histogram(tb, e1, weights=cb)[0] / np.maximum(np.histogram(tb, e1)[0] * 0.05, 1e-9)
        tc = e1[:-1] + 0.5
        Lg = dipole_L(*pw)
        m = (np.histogram(tb, e1)[0] == 20) & ~((tc > -10) & (tc < 1000))
        y = uniform_filter1d(y, 21)
        m &= y < np.nanpercentile(y[m], 99)
        rms = lag_scan(tc[m], y[m], tg, Lg, lags)
        say(f"  GECAM-C control: LG background vs its dipole L(t + lag): best lag {refine(lags, rms):+.1f} s "
            f"({m.sum()} s; rms at 0 / +185 s = {rms[lags == 0][0]:.1f} / {np.interp(185, lags, rms):.1f})")
    except Exception as ex:  # noqa: BLE001
        say(f"  GECAM-C control not run: {ex}")
    # ---------- 5. occultation of the HE pointing direction (not the GRB) ----------
    say("\n== hypothesis: the literature interval is the occultation of the HE boresight, not of the GRB ==")
    from astropy.coordinates import SkyCoord, ITRS
    import astropy.units as u
    ra, dec, ta = [], [], []
    for h in ("13", "14"):
        fa = sorted(glob.glob(f"{K1}/HXMT_20221009T{h}_Att_*1K.FITS"))[0]
        a_ = fits.open(fa)["ATT_Pointing"].data
        ta.append(a_["Time"] - T0); ra.append(a_["Ra"]); dec.append(a_["Dec"])
    ta, ra, dec = (np.concatenate(v) for v in (ta, ra, dec))
    sel = (ta > 1000) & (ta < 4500)
    ta, ra, dec = ta[sel][::5], ra[sel][::5], dec[sel][::5]
    tsa = Time("2022-10-09T13:17:00", scale="utc") + TimeDelta(ta, format="sec")
    pos = np.vstack([np.interp(ta, to, po[k]) for k in range(3)])
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1e9 * u.pc)
    v = c.transform_to(ITRS(obstime=tsa)).cartesian.xyz.value
    v = v / np.linalg.norm(v, axis=0)
    ep = elevation(pos, v, 0.0)
    gsep = SkyCoord(ra=288.265 * u.deg, dec=19.773 * u.deg).separation(SkyCoord(ra=ra * u.deg, dec=dec * u.deg)).deg
    say(f"  HE pointing RA {np.median(ra):.2f}..., Dec {np.median(dec):.2f}; separation from GRB "
        f"{gsep.min():.1f}-{gsep.max():.1f} deg over T0+1000..4500 s")
    say(f"  pointing occultation (geometric limb): ingress T0+{np.round(crossing(ta, ep), 1)} s, "
        f"egress T0+{np.round(crossing(ta, -ep), 1)} s")
    (OUT / "geo2.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
