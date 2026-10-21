"""Shared paths and helpers for the GRB 221009A HE variability scouting
(p221009_sci_*.py). Runs on the IHEP farm; all paths are absolute."""
import numpy as np

BASE = "/scratchfs2/gecam/guohx/p221009"
OUT = f"{BASE}/sci/out"
EVT_REC = f"{BASE}/blink/data/p221009/scale/hxmt_events.npz"   # t rel T0, fill, ch, pw, box
EVT_PKT = f"{BASE}/inject/out/221009a_events.npz"               # per box MET, pkt, SEC flag
HE1K = f"{BASE}/cache/he1k_hour.npz"
LC_V3 = f"{BASE}/runs/v3full2/data/p221009/v3/lc_221009a_v3_1s.csv"
GECAM_EVT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily/2022/10/09/GRD_EVT/gcg_evt_221009_13_v09.fits"
GECAM_BTIME = f"{BASE}/blink/data/gecam_c/gcg_btime_221009_13_v12.fits"
T0_MET = 339945423.0          # HXMT MET of 2022-10-09T13:17:00 UTC
GECAM_MJDREF = 59215 + 0.00080074074


def gecam_t0():
    """GECAM MET of T0 (TT based, as in p221009_timing_data)."""
    from astropy.time import Time
    t0 = Time("2022-10-09T13:17:00", scale="utc")
    return (t0.tt - Time(GECAM_MJDREF, format="mjd", scale="tt")).sec
