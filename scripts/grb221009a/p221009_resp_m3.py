"""Referee2 M3(a,b): does Cnt_LargeEvt respond to the source, and how large
is the systematic error of the quiet-period background (GRB 221009A)?

(a) Source response of LargeEvt: excess of the 1-s LargeEvt sum over a
    time-linear baseline (fitted on flanking source-free seconds) divided by
    the HE/CsI source counts (ch 20-255, observed, dead-time corrected) for
    the precursor (0-9 s) and the clean early main emission (176-216 s).
    Converted to the quiet-period case for the GBM soft spectra with the
    DRM ratio of deposits >3.1 MeV to deposits in 0.17-3.1 MeV.
(b) Blind-extrapolation residuals of the adopted background model
    (CsI = a + s * LE; s from source-free reference passes, a from a 400-s
    window ending 10 s before; prediction for the window +12...+170 s
    later, as for the quiet period) on every source-free stretch available:
    burst-day Earth-occulted data and the three reference passes, LE < 5500.

    python p221009_resp_m3.py [OUTDIR]"""
import sys

import numpy as np

from p221009_resp_adjorb_ana import load, sel
from p221009_resp_adjorb_ana2 import cnts
from p221009_resp_spec import cpl, largeevt, load_events, pl, photon_flux

BASE = "/scratchfs2/gecam/guohx/p221009"
SLOPE = {"d1008": 0.982, "d1007": 0.979, "d1003": 0.985}     # p221009_resp_adjorb_lever.py


def lin_excess(tr, y, flank, win):
    m = np.zeros(tr.size, bool)
    for w in flank:
        m |= sel(tr, w)
    c = np.polyfit(tr[m], y[m], 1)
    resid_sd = np.std(y[m] - np.polyval(c, tr[m]))
    w = sel(tr, win)
    return (y[w] - np.polyval(c, tr[w])).sum(), resid_sd * np.sqrt(w.sum()), c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    lines = []
    tr, gb, _ = load("burst")
    cb = gb.sum(1)
    le = cnts("burst", tr)["le"]
    # live fraction (box mean) for the source counts
    import csv
    fl = {}
    for r in csv.DictReader(open(f"{BASE}/resp/in/ts_1s.csv")):
        fl.setdefault(int(float(r["t"])), []).append(float(r["flive"]))
    f = np.array([np.mean(fl.get(int(k), [1.0])) for k in tr])
    lines.append("(a) LargeEvt response to the source")
    cases = {"precursor 0-9": ([(-60, -3), (15, 60)], (0, 9)),
             "precursor -1-12": ([(-60, -3), (15, 60)], (-1, 12)),
             "onset 176-200": ([(120, 172)], (176, 200)),
             "early main 176-216": ([(120, 172)], (176, 216))}
    eps = {}
    for name, (flank, win) in cases.items():
        dle, sle, _ = lin_excess(tr, le, flank, win)
        dcs, scs, _ = lin_excess(tr, cb / f, flank, win)
        e, se = dle / dcs, sle / dcs
        eps[name] = (e, se)
        lines.append(f"  {name:18s}: dLE {dle:+.0f} +- {sle:.0f}, dCsI(src, /f) {dcs:.0f}; eps = dLE/dCsI = {e:+.4f} +- {se:.4f}")
    # DRM: fraction of deposits above the upper discriminator (> ch 275 ~ 3.39 MeV; MC deposits stop at 4 MeV,
    # so this is a lower bound) relative to ch 20-255, for the precursor and the GBM quiet spectra
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    dep = z["rise_dep"].astype(float)            # incident x deposited (4-keV bins, 0-4000 keV), 18 det
    mid = np.arange(dep.shape[1]) * 4 + 2
    lo_m = (mid >= 174) & (mid < 3130)
    hi_m = mid >= 3390
    shapes = {"precursor PL -2.12": (pl, [0.0, -2.12]), "GBM II PL -2.05 (to MeV)": (pl, [0.0, -2.05]),
              "GBM II COMP Ep=54 a=-1.88": (cpl, [0.0, -1.88, np.log(54)]),
              "GBM Lesage II COMP Ep=5.7MeV a=-0.4 (hard)": (cpl, [0.0, -0.4, np.log(5700)])}
    rat = {}
    for k, (m, p) in shapes.items():
        ph = photon_flux(m, p, elo, ehi)
        a = (ph[:, None] * dep[:, hi_m]).sum()
        b = (ph[:, None] * dep[:, lo_m]).sum()
        rat[k] = a / b
        lines.append(f"  DRM deposits >3.39 MeV / 0.17-3.13 MeV for {k}: {rat[k]:.4f}")
    e_p, se_p = eps["precursor 0-9"]
    e_m, se_m = eps["early main 176-216"]
    for k in shapes:
        sc = rat[k] / rat["precursor PL -2.12"]
        lines.append(f"  quiet-period eps for {k}: precursor-scaled {e_p * sc:+.4f} +- {se_p * sc:.4f}; "
                     f"95% upper {max(e_p, 0) * sc + 1.96 * se_p * sc:.4f}; "
                     f"net-rate suppression factor (1 - 0.98 eps) >= {1 - 0.98 * (max(e_p, 0) * sc + 1.96 * se_p * sc):.3f}")
    # (b) blind extrapolation residuals
    lines.append("(b) blind extrapolation residuals (net counts/s over a 158-s window 12-170 s after a 400-s calibration)")
    res_all = {}
    meta = []
    # burst day over the full cached hour (T0-1020...+2580) from the 1K cache and 1B LargeEvt
    te, che = load_events()
    tl = np.arange(-1020, 2580)
    cl = np.histogram(te[che < 256], np.append(tl, 2580))[0].astype(float)
    lm = largeevt()
    ll = np.array([lm.get(int(k), np.nan) for k in tl])
    okl = np.isfinite(ll)
    ll = np.interp(tl, tl[okl], ll[okl])
    sources = [("burst pre-burst", "burstlong", None, (-1020, -5)), ("burst occulted", "burstlong", None, (1890, 2580))] \
        + [(r, r, SLOPE[r], (-420, 2620)) for r in SLOPE]
    for name, lab, s, rng in sources:
        if lab == "burstlong":
            c, l, trr = cl, ll, tl
            sl = np.mean(list(SLOPE.values()))
        else:
            _, g, _ = load(lab)
            c = g.sum(1)
            l = cnts(lab, tr)["le"]
            sl = np.mean([v for k, v in SLOPE.items() if k != lab])      # leave-one-out slope
            trr = tr
        res = []
        for t0 in range(rng[0] + 410, rng[1] - 170, 10):
            cal = sel(trr, (t0 - 410, t0 - 10))
            tst = sel(trr, (t0 + 12, t0 + 170))
            if l[cal].max() > 5500 or l[tst].max() > 5500 or c[cal].min() <= 0 or c[tst].min() <= 0:
                continue
            a = np.mean(c[cal] - sl * l[cal])
            res.append(np.mean(c[tst] - (a + sl * l[tst])))
            meta.append((name, t0, l[cal].mean(), l[tst].mean() - l[cal].mean(),
                         np.polyfit(trr[tst], l[tst], 1)[0], res[-1]))
        res = np.array(res)
        res_all[name] = res
        if res.size:
            lines.append(f"  {name:16s} (slope {sl:.3f}): N={res.size:3d} (overlapping, ~{max(res.size // 16, 1)} independent); "
                         f"mean {res.mean():+.1f}, rms {np.sqrt(np.mean(res ** 2)):.1f}, "
                         f"|res| 68% {np.percentile(np.abs(res), 68):.1f}, 95% {np.percentile(np.abs(res), 95):.1f} counts/s")
        else:
            lines.append(f"  {name}: no usable windows")
    allr = np.concatenate([v for v in res_all.values() if v.size])
    stat = np.sqrt(np.mean(cb[sel(tr, (-400, -10))]) / 158)       # Poisson of a 158-s mean
    lines.append(f"  ALL: N={allr.size}, mean {allr.mean():+.1f}, rms {np.sqrt(np.mean(allr ** 2)):.1f}, "
                 f"|res| 68% {np.percentile(np.abs(allr), 68):.1f}, 95% {np.percentile(np.abs(allr), 95):.1f} counts/s "
                 f"(Poisson part of a 158-s mean incl. calibration: ~{stat * np.sqrt(1 + 158 / 400):.1f})")
    sysv = np.sqrt(max(np.mean(allr ** 2) - stat ** 2 * (1 + 158 / 400), 0))
    lines.append(f"  -> systematic (rms minus Poisson): {sysv:.1f} counts/s")
    # stratify: tests resembling the quiet period (burst day: cal LE 2844, test - cal = +640, dLE/dt ~ +6/s)
    M = np.array([m[1:] for m in meta], float)
    for lab, msk in (("LE_cal < 3300 and 300 < dLE < 1000 (quiet-like)", (M[:, 1] < 3300) & (M[:, 2] > 300) & (M[:, 2] < 1000)),
                     ("dLE > 300 (rising)", M[:, 2] > 300), ("|dLE| < 300 (flat)", np.abs(M[:, 2]) < 300),
                     ("dLE < -300 (falling)", M[:, 2] < -300)):
        r = M[msk, 4]
        if r.size:
            lines.append(f"  stratum {lab}: N={r.size}, mean {r.mean():+.1f}, rms {np.sqrt(np.mean(r ** 2)):.1f} counts/s")
    cc = np.corrcoef(M[:, 2], M[:, 4])[0, 1]
    k = np.polyfit(M[:, 2], M[:, 4], 1)
    lines.append(f"  residual vs dLE: corr {cc:+.2f}, slope {k[0]:+.4f} counts/s per LE unit, intercept {k[1]:+.1f}")
    k2 = np.polyfit(M[:, 3], M[:, 4], 1)
    lines.append(f"  residual vs dLE/dt in test window: corr {np.corrcoef(M[:, 3], M[:, 4])[0, 1]:+.2f}, slope {k2[0]:+.2f} "
                 f"(counts/s per LE/s), intercept {k2[1]:+.1f}; quiet period dLE/dt and dLE -> predicted residual "
                 f"{np.polyval(k, 640):+.1f} (dLE fit)")
    np.savetxt(f"{out}/m3_blind_meta.txt", np.column_stack([M]), fmt="%.2f",
               header="t0 LE_cal dLE dLEdt_test resid (source order: " + ",".join(dict.fromkeys(m[0] for m in meta)) + ")")
    with open(f"{out}/m3_summary.txt", "w") as fo:
        for l_ in lines:
            print(l_, flush=True)
            print(l_, file=fo)
    np.savez(f"{out}/m3_blind_resid.npz", **{k.replace(' ', '_'): v for k, v in res_all.items()})


if __name__ == "__main__":
    main()
