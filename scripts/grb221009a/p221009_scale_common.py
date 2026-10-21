"""Shared loaders for the GRB 221009A x200-scale / spectral-drift study
(p221009_scale_*.py). Run from the blink repo root.

Times are seconds relative to T0 = 2022-10-09T13:17:00 UTC. HXMT events come
from data/cache_221009a_reconstruct.csv (the cache behind make_f10.py); GECAM-C
from the GRD01 BTIME (50 ms, 14 channels) and the event file.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../talks",)]  # sibling and shared script modules
from make_f10 import load_eng_dtau, to_grid  # noqa: E402

OUT = Path("data/p221009/scale")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = "data/cache_221009a_reconstruct.csv"
TRIGGER_MET_ASTROPY = 339945423.0
GECAM_T0_MET = 55862220.0
GECAM_EVT = "data/gecam_c/gcg_evt_221009_13_v09.fits"
GECAM_BTIME = "data/gecam_c/gcg_btime_221009_13_v12.fits"
GECAM_BTIME_BKG = "data/gecam_c/gcg_btime_221014_13_v06.fits"
SCALE = 200.0
SAA_OFF = [(229.0, 249.0), (267.0, 277.0)]  # HE off incl. partial seconds (data: last evt 229.2, first 248.x; 267.0-276.x)

# Phases (s after T0). P1/P2 and valley follow the HE light curve; the HE-off
# SAA intervals are cut separately where HE enters.
PHASES = {
    "calib 175-212": (175, 212),
    "calib-clean 182-205": (182, 205),  # excludes the GECAM HG particle burst at 171-181
    "P1 rise 212-226": (212, 226),
    "P1 peak 226-231": (226, 231),
    "P2 255-268": (255, 268),
    "valley 277-300": (277, 300),
    "flare 442-593": (442, 593),
    "flare peak 500-520": (500, 520),
    "tail 600-700": (600, 700),
}


def hxmt_events():
    """Cached per-event arrays: t (rel T0), fill flag, channel, pulse width, box."""
    npz = OUT / "hxmt_events.npz"
    if not npz.exists():
        import pandas as pd
        df = pd.read_csv(CACHE, usecols=["box", "type", "met", "channel", "pulse_width"],
                         dtype={"box": "category", "type": "category",
                                "channel": np.uint8, "pulse_width": np.uint8})
        np.savez(npz, t=df["met"].to_numpy() - TRIGGER_MET_ASTROPY,
                 fill=(df["type"] != "EVT").to_numpy(),
                 ch=df["channel"].to_numpy(), pw=df["pulse_width"].to_numpy(),
                 box=df["box"].cat.codes.to_numpy().astype(np.int8))
    d = np.load(npz)
    return d["t"], d["fill"], d["ch"], d["pw"], d["box"]


def saa_mask(x, w=1.0):
    bad = np.zeros(len(x), bool)
    for a, b in SAA_OFF:
        bad |= (x + w > a) & (x < b)
    return bad


def gecam_btime_channels(det=1, bkg=True):
    """50-ms GRD BTIME: t (rel T0), net counts per channel (14), exposure (2)."""
    f = fits.open(GECAM_BTIME)
    s = f[f"SPECTRUM{det:02d}"].data
    t = s["STARTTIME"] - GECAM_T0_MET
    c = s["COUNTS"].astype(float)
    e = s["EXPOSURE"].astype(float)
    cb = np.zeros_like(c)
    if bkg:
        fb = fits.open(GECAM_BTIME_BKG)
        sb = fb[f"SPECTRUM{det:02d}"].data
        off = fb[0].header["TSTART"] - f[0].header["TSTART"]
        tb = sb["STARTTIME"] - off - GECAM_T0_MET
        for k in range(c.shape[1]):
            cb[:, k] = np.interp(t, tb, sb["COUNTS"][:, k].astype(float),
                                 left=np.nan, right=np.nan)
    return t, c, cb, e


def rebin_sum(t, y, edges):
    """Sum 50-ms rows whose start time falls in each bin (y 1-D or 2-D)."""
    idx = np.searchsorted(edges, t, side="right") - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    if y.ndim == 1:
        return np.bincount(idx[ok], weights=y[ok], minlength=len(edges) - 1)
    return np.stack([np.bincount(idx[ok], weights=y[ok, k], minlength=len(edges) - 1)
                     for k in range(y.shape[1])], axis=1)


def hxmt_1s(edges):
    """Event-level HE rates on 1-s edges, as in make_f10: gross obs, gross rec,
    background B, live fraction, dead-time-corrected net."""
    t, fill, ch, pw, box = hxmt_events()
    x = edges[:-1]
    r_obs = np.histogram(t[~fill], bins=edges)[0].astype(float)
    r_all = np.histogram(t, bins=edges)[0].astype(float)
    B = r_all[x < -10].mean()
    et, dtau, flive, tau_us, B_eng = load_eng_dtau(50, 700)
    fl = to_grid(x, et, flive)
    with np.errstate(divide="ignore", invalid="ignore"):
        net_dt = np.where(fl > 0.05, r_all / fl - B, np.nan)
    return dict(x=x, r_obs=r_obs, r_all=r_all, B=B, flive=fl,
                net=r_all - B, net_dt=net_dt)


def robust(r):
    r = r[np.isfinite(r)]
    if not len(r):
        return np.nan, np.nan, 0
    q75, q25 = np.percentile(r, [75, 25])
    return np.median(r), (q75 - q25) / 1.349, len(r)
