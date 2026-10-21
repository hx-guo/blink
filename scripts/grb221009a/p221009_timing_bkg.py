"""HE orbital-background model for the GRB 221009A timing study.

The HE event rate outside the burst follows the orbital particle background,
which rises by a factor ~2.5 between T0 and T0+1000 s on this orbit. We model
the gross 1-s HE event rate as a linear function of the engineering
Cnt_LargeEvt counter (events above the upper discriminator, dominated by
charged particles), calibrated on two source-free windows:

  pre-burst      T0-1020 .. T0-10 s
  Earth-occulted T0+1780 .. T0+2580 s  (GRB below HXMT's Earth limb from ~T0+1720 s)

Calibration is a closed-form least-squares line. The proxy is used only where
Cnt_LargeEvt lies inside its calibrated range (<= the occulted-window maximum);
outside that range the model is extrapolated and flagged.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION
from p221009_timing_common import OUT

T0_MET = 339945423.0
HE1K = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"
CAL_WINDOWS = [(-1020.0, -10.0), (1780.0, 2580.0)]


def eng_counter(name="LargeEvt"):
    """Sum of Cnt_<name>_<det> over the 18 detectors, keyed by integer T0-relative second."""
    acc = {}
    for box, code in BOX_CODE.items():
        fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(
            f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
        d = fits.open(fn)["HE_Eng"].data
        off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        t = np.round(d["Time"].astype(float) + off + MET_CORRECTION - T0_MET).astype(int)
        c = sum(d[f"Cnt_{name}_{BOX_OFFSET[box] + j}"].astype(float) for j in range(6))
        for tt, cc in zip(t, c):
            acc[tt] = acc.get(tt, 0.0) + cc
    ts = np.array(sorted(acc))
    return ts.astype(float), np.array([acc[k] for k in ts])


def he1k_hour():
    p = OUT / "he1k_hour.npz"
    if not p.exists():
        d = fits.open(HE1K, memmap=True)[1].data
        np.savez(p, t=d["Time"] - T0_MET, det=d["Det_ID"], ch=d["Channel"],
                 pw=d["Pulse_Width"])
    d = np.load(p)
    return d["t"], d["ch"]


def background_model(chmin=0, windows=CAL_WINDOWS):
    """Return (t_left, gross 1-s rate, model, in_range, coeffs, resid_std)."""
    t, ch = he1k_hour()
    e = np.arange(-1020.0, 2581.0, 1.0)
    r = np.histogram(t[ch >= chmin], e)[0].astype(float)
    tl, lg = eng_counter("LargeEvt")
    lm = dict(zip(tl.astype(int), lg))
    x = np.array([lm.get(int(k), np.nan) for k in e[:-1]])
    cal = np.zeros(len(r), bool)
    for a, b in windows:
        cal |= (e[:-1] >= a) & (e[:-1] < b)
    cal &= np.isfinite(x)
    coef = np.polyfit(x[cal], r[cal], 1)
    model = np.polyval(coef, x)
    resid = np.std(r[cal] - model[cal])
    in_range = x <= x[cal].max()
    return e[:-1], r, model, in_range, coef, resid


# Cnt_LargeEvt also counts source photons above the upper discriminator during
# the brightest phases (and drops to zero while HE is off), so the proxy is
# contaminated there; across these windows the model is replaced by a straight
# line between the clean values at the window edges.
CONTAMINATED = [(174.0, 300.0), (498.0, 540.0)]


def background_clean(chmin=0, windows=CAL_WINDOWS):
    tl, r, model, ok, coef, resid = background_model(chmin=chmin, windows=windows)
    m = model.copy()
    for a, b in CONTAMINATED:
        ia = int(np.searchsorted(tl, a)); ib = int(np.searchsorted(tl, b))
        ya = model[ia - 5:ia].mean(); yb = model[ib:ib + 5].mean()
        m[ia:ib] = ya + (yb - ya) * (tl[ia:ib] - tl[ia - 3]) / (tl[ib + 2] - tl[ia - 3])
    return tl, r, m, ok, coef, resid


if __name__ == "__main__":
    tc, r, m, ok, c, s = background_model()
    print(f"bkg = {c[1]:.0f} + {c[0]:.4f} * LargeEvt ; resid std {s:.0f} evt/s")
    for a, b in [(-1020, -10), (0, 10), (100, 170), (300, 440), (600, 750),
                 (750, 1000), (1000, 1300), (1300, 1700), (1780, 2580)]:
        w = (tc >= a) & (tc < b)
        print(f"  {a:5d}..{b:5d}  gross {r[w].mean():7.0f}  model {m[w].mean():7.0f}"
              f"  excess {np.mean(r[w] - m[w]):7.0f}  in-range {ok[w].mean():.2f}")
    for wins, nm in [([CAL_WINDOWS[0]], "pre only"), ([CAL_WINDOWS[1]], "occ only")]:
        _, _, _, _, cc, ss = background_model(windows=wins)
        print(f"  calib {nm}: {cc[1]:.0f} + {cc[0]:.4f} * LargeEvt (resid {ss:.0f})")

