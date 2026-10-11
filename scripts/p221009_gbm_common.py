"""Shared helpers for the GRB 221009A Fermi-GBM analysis (p221009_gbm_*.py).

Runs on the IHEP farm with the gdt-fermi venv in
/scratchfs2/gecam/guohx/p221009/gbm/venv.  All times are seconds relative to
the GBM trigger time T0 = MET 687014224.98753 (2022-10-09T13:16:59.9875 UTC).
The HXMT event cache and the paper use 13:17:00.000 UTC as zero; with the
0.0017 s Fermi-HXMT light-travel difference, t_paper = t_GBM - 0.0108 s
(figs_v4.md section 6; runs before 2026-10-04 used 0.06-0.062 s).
"""
import os
import glob
import numpy as np
import pandas as pd
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.headers import TteHeaders
from gdt.missions.fermi.gbm.response import GbmRsp2
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.data_primitives import EventList, Ebounds, Gti
from astropy.io import fits

ROOT = os.environ.get("GBM_ROOT", "/scratchfs2/gecam/guohx/p221009/gbm")
TTE_DIR = os.environ.get("GBM_TTE_DIR", "/scratchfs2/gecam/guohx/p221009/sci/gbm")
TRIG_DIR = os.path.join(ROOT, "trig")
OUT = os.environ.get("GBM_OUT", os.path.join(ROOT, "out"))

T0 = 687014224.98753
HXMT_DT = 0.0108                # t_HXMT = t_GBM - 0.0108 s (was 0.062 before 2026-10-04)
RA, DEC = 288.26452, 19.77350   # Swift-XRT
NAI = ["n%x" % i for i in range(12)]
BGO = ["b0", "b1"]
DETS = NAI + BGO
BTI = [(218.501, 277.894), (507.275, 514.443)]
KEV2ERG = 1.602176634e-9

# energy windows used in spectral fits (Lesage+2023: NaI 20-1000 keV, BGO > 400 keV)
NAI_ERANGE = (20.0, 900.0)
BGO_ERANGE = (400.0, 38000.0)


def outdir(*sub):
    d = os.path.join(OUT, *sub)
    os.makedirs(d, exist_ok=True)
    return d


def open_tte(det, trange=(-400.0, 800.0)):
    """Continuous (hourly) TTE for one detector, cut to trange and re-referenced
    to T0 (gdt then works in relative seconds, as for trigger TTE)."""
    f = glob.glob(os.path.join(TTE_DIR, "glg_tte_%s_221009_13z_v*.fit*" % det))[0]
    with fits.open(f) as h:
        eb = Ebounds.from_bounds(h["EBOUNDS"].data["E_MIN"], h["EBOUNDS"].data["E_MAX"])
        t = h["EVENTS"].data["TIME"] - T0
        m = (t >= trange[0]) & (t < trange[1])
        ev = EventList(times=t[m], channels=h["EVENTS"].data["PHA"][m].astype(int), ebounds=eb)
        dead = float(h["EVENTS"].header.get("EVT_DEAD", 2.6e-6))
    return GbmTte.from_data(ev, gti=Gti.from_bounds([trange[0]], [trange[1]]),
                            event_deadtime=dead, overflow_deadtime=1.0e-5, headers=TteHeaders())


def rel(t):
    return np.asarray(t, float)


def rsp2(det):
    return GbmRsp2.open(glob.glob(os.path.join(TRIG_DIR, "glg_cspec_%s_bn221009553_v*.rsp2" % det))[-1])


def response_for(det, t1, t2):
    """DRM interpolated (in pointing) to the interval centre; rsp2 times are
    relative to the trigger."""
    return rsp2(det).interpolate(0.5 * (t1 + t2))


def phaii(tte, dt=1.024, trange=(-400.0, 800.0)):
    """Bin the TTE into a 128-channel PHAII with deadtime-corrected exposure;
    bins phased to T0 (as CSPEC/LLE)."""
    return tte.to_phaii(bin_by_time, dt, time_ref=0.0, tstart=trange[0], tstop=trange[1])


def poly_bkg(ph, bkg_ranges, order):
    """gdt polynomial background, fitted channel by channel over bkg_ranges
    (relative s).  Returns the BackgroundFitter."""
    bf = BackgroundFitter.from_phaii(ph, Polynomial, time_ranges=list(bkg_ranges))
    bf.fit(order=order)
    return bf


def erange_for(det):
    return BGO_ERANGE if det.startswith("b") else NAI_ERANGE


def lightcurve(times_rel, pha, edges, chan_lo, chan_hi, deadtime=(2.6e-6, 1.0e-5), nchan=128):
    """Counts and deadtime-corrected exposure per time bin for channels
    [chan_lo, chan_hi]; the deadtime uses all events in the bin (overflow
    channel 127 costs 10 us)."""
    allc = np.histogram(times_rel, edges)[0]
    ovf = np.histogram(times_rel[pha == nchan - 1], edges)[0]
    sel = (pha >= chan_lo) & (pha <= chan_hi)
    cnt = np.histogram(times_rel[sel], edges)[0]
    expo = np.diff(edges) - deadtime[0] * (allc - ovf) - deadtime[1] * ovf
    return cnt, expo


def chan_range(ebounds, e1, e2):
    lo = np.asarray(ebounds.low_edges()); hi = np.asarray(ebounds.high_edges())
    c1 = int(np.searchsorted(lo, e1, side="right") - 1)
    c2 = int(np.searchsorted(hi, e2, side="left"))
    return max(c1, 0), min(c2, len(lo) - 1)


def write_csv(df, *path):
    p = os.path.join(outdir(*path[:-1]), path[-1])
    pd.DataFrame(df).to_csv(p, index=False, float_format="%.6g")
    return p


# ---------------------------------------------------------------- orbital background geometry
DAILY = os.path.join(ROOT, "daily")
DAY = {-30: "221007", -15: "221008", 0: "221009", 15: "221010", 30: "221011"}


def _poshist(day):
    with fits.open(glob.glob(os.path.join(DAILY, "glg_poshist_all_%s_v*.fit" % day))[0]) as h:
        d = h[1].data
        return np.array(d["SCLK_UTC"], float), np.array(d["SC_LAT"], float), np.array(d["SC_LON"], float)


def _geo_at(ph, t):
    s, la, lo = ph
    i = np.clip(np.searchsorted(s, t), 1, len(s) - 1)
    a, b = np.radians(la[i]), np.radians(lo[i])
    return np.stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)], -1)


def orbital_shifts(ks=(-30, -15, 15, 30)):
    """Time shifts (s) to the same ground track k orbits away: k x the orbital
    period (ascending nodes on the burst day), refined by minimising the mean
    ground-track offset over T0-400...+800 s.  Returns (period, {k: (shift,
    mean offset deg)})."""
    ph0 = _poshist(DAY[0])
    tw = T0 + np.arange(-400.0, 800.0, 10.0)
    v0 = _geo_at(ph0, tw)
    s, la, _ = ph0
    asc = s[1:][(la[:-1] < 0) & (la[1:] >= 0)]
    P = np.median(np.diff(asc))
    shifts = {}
    for k in ks:
        phk = _poshist(DAY[k])
        grid = k * P + np.arange(-900.0, 900.0, 1.0)
        dist = [np.degrees(np.arccos(np.clip((_geo_at(phk, tw + g) * v0).sum(-1), -1, 1))).mean() for g in grid]
        j = int(np.argmin(dist))
        shifts[k] = (grid[j], dist[j])
    return P, shifts
