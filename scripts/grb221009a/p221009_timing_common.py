"""Shared loaders for the GRB 221009A timing study (p221009_timing_*.py).

All times are seconds relative to T0 = 2022-10-09T13:17:00 UTC at the HXMT
spacecraft. Run p221009_timing_data.py first to build the .npz caches.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits

OUT = Path("data/p221009/timing")
GECAM_BTIME = "data/gecam_c/gcg_btime_221009_13_v12.fits"
GECAM_BTIME_BKG = "data/gecam_c/gcg_btime_221014_13_v06.fits"
GECAM_T0_MET = 55862220.0      # 2022-10-09T13:17:00 UTC in GECAM MET (TT-based)
SAA_OFF = [(229.2, 249.0), (267.0, 277.0)]   # HE detector-off (SAA mode), [lo, hi), all boxes
HE_BKG_WIN = (-50.0, -10.0)

# HE raw-channel bands. HE ran in its low-gain mode during the burst and the
# excess is CsI-dominated, so these are channel ranges, roughly 0.2-0.5 /
# 0.5-1 / 1-2.5 MeV deposited energy (same bands as the energy-band study).
HE_BANDS = {"ch30-53": (30, 53), "ch53-99": (53, 99), "ch99-224": (99, 224)}


def he_events():
    d = np.load(OUT / "hxmt_events.npz")
    return d["t"], d["fill"], d["ch"], d["box"]


def eng_1hz():
    d = np.load(OUT / "eng_1hz.npz")
    return d["t"], d["flive"], d["dtau"]


def light_travel_ms():
    import json
    return json.loads((OUT / "geometry.json").read_text())


def he_rate(t, edges, sel=None):
    """Gross HE count rate (obs + fill) on edges, optionally for a mask."""
    tt = t if sel is None else t[sel]
    return np.histogram(tt, bins=edges)[0] / np.diff(edges)


def flive_on(edges, et, flive):
    """Hold the 1-Hz live fraction onto arbitrary bins (by bin centre)."""
    c = 0.5 * (edges[:-1] + edges[1:])
    k = np.floor(c).astype(int)
    m = {int(round(x)): v for x, v in zip(et, flive)}
    return np.array([m.get(int(kk), np.nan) for kk in k])


def saa_mask(edges):
    """True where a bin overlaps an HE detector-off interval."""
    lo, hi = edges[:-1], edges[1:]
    bad = np.zeros(len(lo), bool)
    for a, b in SAA_OFF:
        bad |= (hi > a) & (lo < b)
    return bad


def gecam_btime(det=1, chans=range(9, 13), subtract_bkg=False, deadtime=True):
    """GECAM-C BTIME 50 ms series for one GRD.

    Returns (t_left, counts, live, rate) where live is the per-bin exposure of
    the gain used (HG for ch 0-5, LG for ch 7-12) and rate = counts/live when
    deadtime is True. t_left is shifted to the HXMT spacecraft frame using the
    light-travel table of geometry.json.
    """
    chans = list(chans)
    f = fits.open(GECAM_BTIME)
    d = f[f"SPECTRUM{det:02d}"].data
    t = d["STARTTIME"] - GECAM_T0_MET
    c = d["COUNTS"][:, chans].sum(axis=1).astype(float)
    gi = 1 if min(chans) >= 7 else 0
    live = d["EXPOSURE"][:, gi].astype(float)
    geo = light_travel_ms()
    tk = np.array([0.0, 200.0, 250.0, 500.0])
    dk = np.array([geo[f"T0+{x:.0f}"]["dt_gecam_minus_hxmt_ms"] for x in tk]) / 1e3
    t = t - np.interp(t, tk, dk)
    if subtract_bkg:
        fb = fits.open(GECAM_BTIME_BKG)
        db = fb[f"SPECTRUM{det:02d}"].data
        off = fb[0].header["TSTART"] - f[0].header["TSTART"]
        tb = db["STARTTIME"] - off - GECAM_T0_MET
        cb = db["COUNTS"][:, chans].sum(axis=1).astype(float)
        c = c - np.interp(t, tb, cb, left=np.nan, right=np.nan)
    rate = c / live if deadtime else c / 0.05
    return t, c, live, rate


def rebin_series(t_left, y, edges, mode="mean"):
    """Average (or sum) a regular fine series into coarser bins by bin start."""
    idx = np.digitize(t_left + 1e-6, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(y)
    s = np.bincount(idx[ok], weights=y[ok], minlength=len(edges) - 1)
    n = np.bincount(idx[ok], minlength=len(edges) - 1)
    if mode == "sum":
        return s, n
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, n
