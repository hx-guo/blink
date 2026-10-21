#!/usr/bin/env python3
"""GRB 221009A timing study: data preparation.

Builds compact .npz caches under data/p221009/timing/ so that the analysis
scripts (p221009_timing_*.py) do not re-parse the 1 GB csv:

  hxmt_events.npz   t (s rel. T0, HXMT frame), fill (bool), ch (uint8), box (uint8)
  eng_1hz.npz       t (s rel. T0, 1-Hz), flive (18-det mean), dtau (gross D/tau evt/s)
  gecam_c.npz       t (s rel. T0, HXMT frame, GECAM light-travel corrected),
                    det (1..12), gain (0=HG,1=LG), pi, dead (us)
  geometry.json     light-travel numbers HXMT vs GECAM-C at T0+200 s

Time frame: everything is referred to the HXMT spacecraft, T0 = 2022-10-09T13:17:00 UTC
(MET_HXMT 339945423.0, astropy). GECAM-C event times are shifted to the HXMT
spacecraft with t_hxmt = t_gecam + (r_gecam - r_hxmt) . n_src / c  (photons reach
the satellite closer to the source first; see feedback on light-travel sign).

Usage (from blink root):  .venv/bin/python scripts/grb221009a/p221009_timing_data.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, ITRS, GCRS, CartesianRepresentation
import astropy.units as u

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../talks",)]  # sibling and shared script modules
from make_f10 import load_eng_dtau, TRIGGER_MET_ASTROPY, CACHE

OUT = Path("data/p221009/timing")
T0_UTC = "2022-10-09T13:17:00"
GECAM_EVT = "data/gecam_c/gcg_evt_221009_13_v09.fits"
GECAM_POS = OUT / "gc_posatt_221009_13_v07.fits"
HXMT_ORB = "data/hxmt_aux/HXMT_20221009T13_Orbit_FFFFFF_V1_1K.FITS"
GECAM_MJDREF = 59215 + 0.00080074074          # TT
SRC = SkyCoord(288.2643 * u.deg, 19.7712 * u.deg)  # GRB 221009A (Swift/XRT)
C_LIGHT = 299792458.0


def hxmt_events():
    df = pd.read_csv(CACHE, usecols=["box", "type", "met", "channel"],
                     dtype={"box": "category", "type": "category"})
    t = df["met"].to_numpy() - TRIGGER_MET_ASTROPY
    fill = (df["type"] == "FILL_GAP").to_numpy()
    box = df["box"].map({"A": 0, "B": 1, "C": 2}).to_numpy().astype(np.uint8)
    ch = df["channel"].to_numpy().astype(np.uint8)
    o = np.argsort(t, kind="stable")
    np.savez(OUT / "hxmt_events.npz", t=t[o], fill=fill[o], ch=ch[o], box=box[o])
    print(f"HXMT: {len(t):,} events ({fill.sum():,} fill), "
          f"t = {t.min():.1f} .. {t.max():.1f} s")


def eng():
    t, dtau_net, flive, tau_us, bkg = load_eng_dtau(50.0, 2500.0)
    np.savez(OUT / "eng_1hz.npz", t=t, flive=flive, dtau=dtau_net + bkg,
             tau_us=tau_us)
    print(f"ENG: {len(t)} s, flive min {np.nanmin(flive):.3f} "
          f"at {t[np.nanargmin(flive)]:.0f} s")


def ecef_at(fn, tcol, cols, met_to_time, t_query):
    d = fits.open(fn)[1].data
    tt = d[tcol].astype(float)
    return np.array([np.interp(t_query, tt, d[c].astype(float)) for c in cols])


def geometry():
    """Light-travel offset (GECAM-C minus HXMT path) projected on the source."""
    t0 = Time(T0_UTC, scale="utc")
    out = {}
    for trel in (0.0, 200.0, 250.0, 500.0):
        tq = t0 + trel * u.s
        hx_met = TRIGGER_MET_ASTROPY + trel
        r_hx = ecef_at(HXMT_ORB, "Time", ["X", "Y", "Z"], None, hx_met)
        gc_met = (tq.tt - Time(GECAM_MJDREF, format="mjd", scale="tt")).sec
        r_gc = ecef_at(GECAM_POS, "TIME", ["X_WGS84", "Y_WGS84", "Z_WGS84"],
                       None, gc_met)
        # source unit vector in ITRS (Earth-fixed) at this instant
        n = SRC.transform_to(ITRS(obstime=tq)).cartesian.xyz.value
        n = n / np.linalg.norm(n)
        dproj = float(np.dot(r_gc - r_hx, n))
        out[f"T0+{trel:.0f}"] = {
            "r_hxmt_km": (r_hx / 1e3).round(1).tolist(),
            "r_gecam_km": (r_gc / 1e3).round(1).tolist(),
            "sep_km": float(np.linalg.norm(r_gc - r_hx) / 1e3),
            # positive: GECAM-C is closer to the source -> sees photons earlier
            "dt_gecam_minus_hxmt_ms": -dproj / C_LIGHT * 1e3,
        }
    (OUT / "geometry.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"  {k}: sep {v['sep_km']:.0f} km, photon arrival "
              f"GECAM - HXMT = {v['dt_gecam_minus_hxmt_ms']:+.2f} ms")
    return out


def gecam(geo):
    t0 = Time(T0_UTC, scale="utc")
    gc_t0 = (t0.tt - Time(GECAM_MJDREF, format="mjd", scale="tt")).sec
    # arrival-time difference varies by < 0.1 ms over the burst; interpolate linearly
    tk = np.array([0.0, 200.0, 250.0, 500.0])
    dk = np.array([geo[f"T0+{x:.0f}"]["dt_gecam_minus_hxmt_ms"] for x in tk]) / 1e3
    f = fits.open(GECAM_EVT, memmap=True)
    T, D, G, P, DT = [], [], [], [], []
    for i in range(1, 13):
        ev = f[f"EVENTS{i:02d}"].data
        tr = ev["TIME"] - gc_t0
        m = (tr > -60) & (tr < 2400)
        tr = tr[m]
        # GECAM time -> time the same wavefront reaches HXMT
        tr = tr - np.interp(tr, tk, dk)
        T.append(tr); D.append(np.full(m.sum(), i, np.uint8))
        G.append(ev["GAIN_TYPE"][m].astype(np.uint8)); P.append(ev["PI"][m].astype(np.int16))
        DT.append(ev["DEAD_TIME"][m].astype(np.float32))
    t = np.concatenate(T); o = np.argsort(t, kind="stable")
    np.savez(OUT / "gecam_c.npz", t=t[o], det=np.concatenate(D)[o],
             gain=np.concatenate(G)[o], pi=np.concatenate(P)[o],
             dead=np.concatenate(DT)[o])
    eb = f["EBOUNDS"].data
    np.savez(OUT / "gecam_ebounds.npz", ch=eb["CHANNEL"], emin=eb["E_MIN"], emax=eb["E_MAX"])
    print(f"GECAM-C: {len(t):,} events")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    geo = geometry()
    gecam(geo)
    eng()
    hxmt_events()
