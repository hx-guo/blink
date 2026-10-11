"""Quiet-period (T0+12...+170 s) limits from HE/CsI low gain (GRB 221009A
response study).

Background: the 1-s CsI (pw 76-255, ch 20-255) rate is modelled as a linear
function of the engineering Cnt_LargeEvt counter (orbital particle proxy, as
in p221009_timing_bkg), calibrated on the pre-burst window and on the
Earth-occulted window (source occulted from T0+1884 s; An+2023, Zhang+2025).
For comparison a time-linear model fitted on T0-400...-5 s is also given.
Each model is validated by leaving out T0-100...-10 s and predicting it.
Count limits are converted to 0.3-3 MeV energy flux with the precursor DRM.

    python p221009_resp_quiet.py [OUTDIR]"""
import os
import sys

import numpy as np

BASE = "/scratchfs2/gecam/guohx/p221009"
os.chdir(f"{BASE}/blink")                    # data/1B relative paths of p221009_timing_bkg
from p221009_timing_bkg import eng_counter   # noqa: E402
from p221009_resp_spec import (band, counts, cpl, fit, group_drm, load_events,  # noqa: E402
                               photon_flux, pl)
from p221009_resp_prec import efluxband  # noqa: E402

PRE = (-1020, -10)
OCC = (1900, 2580)
QUIET = [(12, 170), (12, 60), (60, 120), (120, 170), (12, 165)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    t, ch = load_events()
    s = ch < 256
    edges = np.arange(-1020, 2581, 1.0)
    r = np.histogram(t[s], edges)[0].astype(float)
    tc = edges[:-1]
    tl, lg = eng_counter("LargeEvt")
    lm = dict(zip(tl.astype(int), lg))
    x = np.array([lm.get(int(k), np.nan) for k in tc])

    def win(ws):
        m = np.zeros(tc.size, bool)
        for a, b in ws:
            m |= (tc >= a) & (tc < b)
        return m & np.isfinite(x) & (r > 0)

    models = {}
    # (a) LargeEvt proxy
    for name, cal in (("proxy", [PRE, OCC]), ("proxy_val", [(-1020, -110), OCC])):
        m = win(cal)
        c = np.polyfit(x[m], r[m], 1)
        models[name] = np.polyval(c, x)
    # (b) time-linear, pre-burst only
    for name, cal in (("tlin", [(-400, -5)]), ("tlin_val", [(-400, -110)])):
        m = win(cal)
        c = np.polyfit(tc[m], r[m], 1)
        models[name] = np.polyval(c, tc)
    lines = []
    vb = [(a, a + 10) for a in range(-100, -10, 10)]
    sysv = {}
    for base in ("proxy", "tlin"):
        mv = models[base + "_val"]
        res = [np.nansum(r[win([w])] - mv[win([w])]) for w in vb]
        stat = np.sqrt(np.mean([r[win([w])].sum() for w in vb]))
        rms = np.sqrt(np.mean(np.square(res)))
        sysv[base] = np.sqrt(max(rms ** 2 - stat ** 2, 0))
        lines.append(f"validation {base}: 10-s residuals over T0-100..-10 (model fitted without them): "
                     f"mean {np.mean(res):+.0f}, rms {rms:.0f}, stat {stat:.0f} -> extra sys {sysv[base]:.0f} counts/10 s")
    # flux conversion with the precursor DRM
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    R = group_drm(z["precursor_per"].astype(float))
    src, bkg, bvar, expo = counts(t, ch, [(0, 9)])
    pp, _ = fit(pl, [-3, -2.0], R, elo, ehi, src, bkg, bvar, expo, 1.0)
    shapes = {"PL(prec index)": (pl, [0.0, pp[1]]),
              "Band(-1,-2.3,1MeV)": (band, [0.0, -1.0, np.log(1000), -2.3]),
              "CPL(-1,300keV)": (cpl, [0.0, -1.0, np.log(300)])}
    conv = {k: (photon_flux(m, p, elo, ehi) @ R).sum() / efluxband(m, p, 300, 3000) for k, (m, p) in shapes.items()}
    lines.append("conversion (counts/s per erg cm-2 s-1 in 0.3-3 MeV): "
                 + ", ".join(f"{k} {v:.3e}" for k, v in conv.items()))
    prec_net = src.sum() - bkg.sum()
    lines.append(f"precursor 0-9 s net counts {prec_net:.0f} (PL index {pp[1]:.3f})")
    for base in ("proxy", "tlin"):
        mdl = models[base]
        for w in QUIET:
            m = win([w])
            net = np.sum(r[m] - mdl[m])
            nb = m.sum() / 10
            sig = np.sqrt(r[m].sum() + (sysv[base] * nb) ** 2)
            ul = max(net, 0) + 3 * sig
            dur = m.sum()
            lines.append(f"{base} quiet {w}: net {net:+.0f} +- {sig:.0f} ({net / sig:+.1f} sig), 3sig UL {ul:.0f} counts;"
                         f" time-avg 0.3-3 MeV flux UL: " + ", ".join(f"{k} {ul / dur / v:.2e}" for k, v in conv.items())
                         + f"; UL/precursor-counts {ul / prec_net:.2f}")
    lc = [(a, np.sum(r[win([(a, a + 10)])] - models["proxy"][win([(a, a + 10)])]),
           np.sum(r[win([(a, a + 10)])] - models["tlin"][win([(a, a + 10)])])) for a in range(-100, 180, 10)]
    np.savetxt(f"{out}/quiet_net_10s.txt", lc, fmt="%.0f %.1f %.1f", header="t_start net_proxy net_tlin (CsI ch20-255)")
    with open(f"{out}/quiet_summary.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            print(l, file=f)


if __name__ == "__main__":
    main()
