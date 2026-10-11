"""Shared loaders for the GRB 221009A v2 number/figure freeze (p221009_v2_*.py).

Run from the blink repo root with .venv/bin/python. Deterministic; no random
numbers. Times are seconds relative to T0 = 2022-10-09T13:17:00.000 UTC
(HXMT MET 339945423, GECAM MET 55862220.0); 1-s bins are [k, k+1).

Conventions frozen here (see docs/team/freeze.md of the paper repo):
  * HE light curve = all recorded events in the reconstruction cache
    (observed + gap filler, NaI and CsI, no channel or pulse-width cut).
  * Box live fraction f_b = 1 - <D>_b / L_cyc from the 1-Hz HE_Eng frame
    labelled round(MET - T0); f = mean of the three boxes (= 18-det mean).
  * HE-off mask: a bin is invalid if it overlaps by more than 10 ms the
    union over boxes of (last event before, first event after) the two
    SAA-mode shut-offs, measured from the observed events (the one
    channel-0 / pw-0 marker per box per integer second is not an event).
  * GECAM-C GRD01 LG = BTIME ch9-12 (0.38-6.28 MeV); 50-ms rows are summed
    into the 1-s bin that contains the row start (T0 = 55862220.0 exactly);
    dead-time correction per row with EXPOSURE[:,1]; background from the
    5-d revisit file with its own EXPOSURE.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION  # noqa: E402

OUT = Path("data/p221009/v2")
OUT.mkdir(parents=True, exist_ok=True)
T0 = 339945423.0
GECAM_T0 = 55862220.0
SCALE = 200.0
BEFORE, AFTER = 50, 700
EDGES = np.arange(-BEFORE, AFTER + 1, 1.0)
X = EDGES[:-1]
EVENTS_NPZ = "data/p221009/scale/hxmt_events.npz"   # from p221009_scale_common
GECAM_BTIME = "data/gecam_c/gcg_btime_221009_13_v12.fits"
GECAM_BKG = "data/gecam_c/gcg_btime_221014_13_v06.fits"
HE_1K = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"
ORBIT = "data/hxmt_aux/HXMT_20221009T13_Orbit_FFFFFF_V1_1K.FITS"
OFF_TOL = 0.010     # s; overlap with the HE-off interval tolerated in a valid bin


def he_events():
    """t (rel T0), fill flag, channel, pulse width, box code (0,1,2 = A,B,C)."""
    d = np.load(EVENTS_NPZ)
    return d["t"], d["fill"], d["ch"], d["pw"], d["box"]


def is_marker(t, fill, ch, pw):
    """The once-per-second channel-0 / pw-0 record each box writes at integer MET."""
    return (~fill) & (ch == 0) & (pw == 0) & (np.abs(t - np.round(t)) < 1e-6)


def he_counts(edges=EDGES):
    """Per-box observed and observed+filler counts per bin (markers kept, as
    in the paper pipeline; they are 1 per box per second and cancel in B)."""
    t, fill, ch, pw, box = he_events()
    out = {}
    for i, b in enumerate("ABC"):
        m = box == i
        out["all_" + b] = np.histogram(t[m], edges)[0].astype(float)
        out["obs_" + b] = np.histogram(t[m & ~fill], edges)[0].astype(float)
    out["n_all"] = out["all_A"] + out["all_B"] + out["all_C"]
    out["n_obs"] = out["obs_A"] + out["obs_B"] + out["obs_C"]
    return out


def he_off_intervals():
    """Per box (last event before, first event after) each SAA-mode shut-off,
    and the union over boxes. Observed events only, markers excluded."""
    t, fill, ch, pw, box = he_events()
    ev = (~fill) & ~is_marker(t, fill, ch, pw)
    per_box = {}
    for i, b in enumerate("ABC"):
        tb = np.sort(t[ev & (box == i)])
        rows = []
        for lo, hi in ((225.0, 252.0), (262.0, 281.0)):
            s = tb[(tb > lo) & (tb < hi)]
            j = np.argmax(np.diff(s))
            rows.append((s[j], s[j + 1]))
        per_box[b] = rows
    union = [(min(per_box[b][k][0] for b in "ABC"), max(per_box[b][k][1] for b in "ABC"))
             for k in range(2)]
    inter = [(max(per_box[b][k][0] for b in "ABC"), min(per_box[b][k][1] for b in "ABC"))
             for k in range(2)]
    return per_box, union, inter


def off_mask(x=X, w=1.0):
    """True for bins overlapping the union HE-off intervals by > OFF_TOL."""
    _, union, _ = he_off_intervals()
    bad = np.zeros(len(x), bool)
    for a, b in union:
        ov = np.clip(np.minimum(x + w, b) - np.maximum(x, a), 0, None)
        bad |= ov > OFF_TOL
    return bad


def eng_frames():
    """Per box, per 1-Hz HE_Eng frame over the whole hour: label k = round(MET - T0),
    box-mean D, L_cyc, the six Cnt_PHODet, and summed Cnt_LargeEvt."""
    out = {}
    for box, code in BOX_CODE.items():
        fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(
            f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
        with fits.open(fn, memmap=True) as fe:
            d = fe["HE_Eng"].data
            off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
            met = d["Time"].astype(float) + off + MET_CORRECTION
            g = [BOX_OFFSET[box] + j for j in range(6)]
            out[box] = dict(
                k=np.round(met - T0).astype(int),
                D=np.mean([d[f"DeadTime_PHODet_{i}"].astype(float) for i in g], 0),
                L=d["Length_Time_Cycle"].astype(float),
                P=np.array([d[f"Cnt_PHODet_{i}"].astype(float) for i in g]),
                LE=np.sum([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in g], 0))
    return out


def eng_on_grid(x, frames=None):
    """Box live fractions, box-mean D (16-us units), L, summed P, P max, LargeEvt on x."""
    frames = frames or eng_frames()
    n = len(x)
    res = {}
    for b, fr in frames.items():
        idx = {int(k): i for i, k in enumerate(fr["k"])}
        f = np.full(n, np.nan); D = np.full(n, np.nan); L = np.full(n, np.nan)
        P = np.full(n, np.nan); Pmax = np.full(n, np.nan); LE = np.full(n, np.nan)
        for j, xx in enumerate(x):
            i = idx.get(int(round(xx)))
            if i is None or fr["L"][i] <= 0:
                continue
            f[j] = 1 - fr["D"][i] / fr["L"][i]
            D[j], L[j] = fr["D"][i], fr["L"][i]
            P[j], Pmax[j] = fr["P"][:, i].sum(), fr["P"][:, i].max()
            LE[j] = fr["LE"][i]
        res[b] = dict(f=f, D=D, L=L, P=P, Pmax=Pmax, LE=LE)
    res["f"] = np.nanmean([res[b]["f"] for b in "ABC"], 0)
    res["LE"] = sum(res[b]["LE"] for b in "ABC")
    return res


def gecam_lg(edges=EDGES):
    """GECAM-C GRD01 LG net rate per bin, uncorrected and LG-dead-time corrected,
    with Poisson errors and the mean LG live fraction."""
    fb, fk = fits.open(GECAM_BTIME), fits.open(GECAM_BKG)
    b, k = fb["SPECTRUM01"].data, fk["SPECTRUM01"].data
    off = fk[0].header["TSTART"] - fb[0].header["TSTART"]
    tb = b["STARTTIME"] - GECAM_T0
    tk = k["STARTTIME"] - off - GECAM_T0
    cb = b["COUNTS"][:, 9:13].sum(1).astype(float)
    ck = np.interp(tb, tk, k["COUNTS"][:, 9:13].sum(1).astype(float),
                   left=np.nan, right=np.nan)
    eb = b["EXPOSURE"][:, 1] / 0.05
    ek = np.interp(tb, tk, k["EXPOSURE"][:, 1] / 0.05, left=np.nan, right=np.nan)
    idx = np.floor(tb + 1e-6).astype(int) - int(edges[0])
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(ck)

    def s(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb)[:nb]
    n = s(np.ones_like(cb))
    T = n * 0.05
    with np.errstate(divide="ignore", invalid="ignore"):
        net = (s(cb) - s(ck)) / T
        sig = np.sqrt(s(cb) + s(ck)) / T
        net_dc = (s(cb / eb) - s(ck / ek)) / T
        sig_dc = np.sqrt(s(cb / eb**2) + s(ck / ek**2)) / T
        live = s(eb) / n
    return dict(net=net, sig=sig, net_dc=net_dc, sig_dc=sig_dc, live=live, nrow=n)


def robust(r):
    r = np.asarray(r)[np.isfinite(r)]
    if not len(r):
        return np.nan, np.nan, 0
    q75, q25 = np.percentile(r, [75, 25])
    return np.median(r), (q75 - q25) / 1.349, len(r)
