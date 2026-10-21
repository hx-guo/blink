"""Referee round 2, R4: structure of the HE quiet-period light curve
(GRB 221009A).  Is the +50 counts/s at 10-70 s and the deficit at 85-115 s
signal (precursor tail) or the LargeEvt-background bias pattern?

1. Net CsI ch 20-255 rate with the adopted model (slope 0.982 from the
   reference passes, offset from T0-420...-10 s), per segment 12-60, 60-85,
   85-115, 115-172 s and in 10-s bins.
2. Expected background bias per segment from the blind-extrapolation tests
   (m3_blind_meta.txt): residual regressed on dLE (segment-mean LE minus
   calibration-mean LE) and dLE/dt in the segment; prediction and the scatter
   about the regression.  Also the residual on the burst day's own pre-burst
   tests with the same lag structure.
3. Expected source counts from the precursor tail: the gbm group's
   time-resolved COMP fits (8.576-13, 13-20, 20-30, 30-43 s; prec_fits.csv)
   folded through the HE DRM.

    python p221009_resp_r4.py [OUTDIR]"""
import csv
import sys

import numpy as np

from p221009_resp_spec import cpl, group_drm, largeevt, load_events, photon_flux, pl

BASE = "/scratchfs2/gecam/guohx/p221009"
SLOPE = 0.982
CAL = (-420, -10)
SEGS = [(12, 60), (60, 85), (85, 115), (115, 172), (121.159, 164.168), (12, 172)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    te, che = load_events()
    tl = np.arange(-1020, 2580)
    c = np.histogram(te[che < 256], np.append(tl, 2580))[0].astype(float)
    lm = largeevt()
    le = np.array([lm.get(int(k), np.nan) for k in tl])
    ok = np.isfinite(le)
    le = np.interp(tl, tl[ok], le[ok])
    cal = (tl >= CAL[0]) & (tl < CAL[1])
    a = np.mean(c[cal] - SLOPE * le[cal])
    net = c - (a + SLOPE * le)
    lec = le[cal].mean()
    # bias regression from blind tests
    M = np.loadtxt(f"{out}/m3_blind_meta.txt")          # t0 LE_cal dLE dLEdt resid
    X = np.column_stack([np.ones(len(M)), M[:, 2], M[:, 3]])
    coef = np.linalg.lstsq(X, M[:, 4], rcond=None)[0]
    scat = np.std(M[:, 4] - X @ coef)
    X1 = np.column_stack([np.ones(len(M)), M[:, 2]])
    coef1 = np.linalg.lstsq(X1, M[:, 4], rcond=None)[0]
    scat1 = np.std(M[:, 4] - X1 @ coef1)
    lines = [f"bias model resid = {coef[0]:+.1f} + {coef[1]:+.4f} dLE + {coef[2]:+.2f} dLE/dt  (scatter {scat:.1f}); "
             f"dLE only: {coef1[0]:+.1f} + {coef1[1]:+.4f} dLE (scatter {scat1:.1f}); N={len(M)}",
             f"calibration-window mean LE {lec:.0f}"]
    # precursor-tail prediction from GBM time-resolved fits
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    R = group_drm(z["precursor_per"].astype(float))
    pred = np.zeros(tl.size)
    for r in csv.DictReader(open(f"{BASE}/gbm/runs/prec4/precursor/prec_fits.csv")):
        if r["tag"] != "tr" or r["model"] != "COMP" or r["dets"] != "n4+n8+b1" or r["bkg"] != "two":
            continue
        t1, t2 = float(r["t1"]) - 0.06, float(r["t2"]) - 0.06
        if t2 <= 8:
            continue
        rate = (photon_flux(cpl, [np.log(float(r["A"])), float(r["index"]), np.log(float(r["Epeak"]))], elo, ehi) @ R).sum()
        m = (tl + 0.5 >= t1) & (tl + 0.5 < t2)
        pred[m] = rate * 1.12            # HE/GBM area ratio on the precursor
        lines.append(f"GBM COMP [{r['t1']},{r['t2']}] a={float(r['index']):.2f} Ep={float(r['Epeak']):.0f}: "
                     f"predicted HE {rate:.0f} counts/s (x1.12 = {rate * 1.12:.0f})")
    lines.append("SEGMENTS: net (stat) | expected bias from blind tests (+-scatter) | net - bias | GBM precursor-tail prediction")
    for s0, s1 in SEGS:
        m = (tl >= s0) & (tl < s1)
        if s0 % 1:
            w = np.clip(np.minimum(tl + 1, s1) - np.maximum(tl, s0), 0, 1)
            m = w > 0
        else:
            w = m.astype(float)
        dur = w.sum()
        n = (w * net).sum() / dur
        st = np.sqrt((w * c).sum()) / dur
        dle = (w * le).sum() / dur - lec
        sl = np.polyfit(tl[m], le[m], 1)[0]
        b = coef @ [1, dle, sl]
        b1 = coef1 @ [1, dle]
        p = (w * pred).sum() / dur
        lines.append(f"  {s0:>7}-{s1:<7}: net {n:+6.1f} +- {st:4.1f}; LE mean {lec + dle:.0f} (dLE {dle:+.0f}), dLE/dt {sl:+.1f}/s; "
                     f"bias {b:+6.1f} (dLE-only {b1:+6.1f}) +- {scat:.0f}; net-bias {n - b:+6.1f}; GBM tail {p:5.1f}")
    rows = []
    for t0 in range(-100, 180, 10):
        m = (tl >= t0) & (tl < t0 + 10)
        dle = le[m].mean() - lec
        sl = np.polyfit(tl[m], le[m], 1)[0]
        rows.append((t0, net[m].mean(), np.sqrt(c[m].sum()) / 10, le[m].mean(), sl, coef @ [1, dle, sl], pred[m].mean()))
    np.savetxt(f"{out}/r4_bins10.txt", rows, fmt="%.0f %.1f %.1f %.0f %.2f %.1f %.1f",
               header="t0 net stat LE dLEdt bias_pred gbm_tail_pred (counts/s)")
    for r in rows:
        lines.append("  bin " + " ".join(f"{x:.1f}" for x in r))
    with open(f"{out}/r4_summary.txt", "w") as fo:
        for l in lines:
            print(l, flush=True)
            print(l, file=fo)


if __name__ == "__main__":
    main()
