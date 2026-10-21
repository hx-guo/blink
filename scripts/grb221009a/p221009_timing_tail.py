#!/usr/bin/env python3
"""GRB 221009A decaying emission after the main pulses, and the flare extent,
with the orbital-background model of p221009_timing_bkg.

1. Net HE rate with (a) the paper's constant pre-burst background B and (b)
   the LargeEvt-proxy background; difference vs time (how much of the late
   'tail' is background drift).
2. Decay of the post-main-pulse emission, T0+280..380 s (before the flare
   envelope starts near T0+385 s): power law in (t - t_ref) for
   t_ref = 0 and t_ref = 226 s, and an exponential; weighted least squares in
   log space is avoided -- fits are done on the linear rate with the 1-s errors
   of p221009_timing_pulses (Poisson/f_live + background-model floor, no 12%
   systematic since f_live > 0.9 here).
3. Flare T90 on the proxy-net curve with the tail continuum modelled by the
   decay fit (alternative to the paper's local linear continuum).
4. Integrated net counts over T0+180..300 s for both backgrounds.

Outputs: data/p221009/timing/tail.{json,png}
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from p221009_timing_common import OUT, he_events, eng_1hz, flive_on
from p221009_timing_bkg import background_clean as background_model

B_PAPER = 9332.0


def main():
    t, fill, ch, box = he_events()
    keep = ~((ch == 0) & (np.abs(t - np.round(t)) < 1e-7))
    t = t[keep]
    et, fl, _ = eng_1hz()
    e = np.arange(170.0, 752.5, 1.0)
    n = np.histogram(t, e)[0].astype(float)
    f = flive_on(e, et, fl)
    tl, r, model, ok, coef, resid = background_model()
    bm = dict(zip(tl.astype(int), model)); okm = dict(zip(tl.astype(int), ok))
    b = np.array([bm[int(x)] for x in e[:-1]])
    inr = np.array([okm[int(x)] for x in e[:-1]])
    # alternative proxy calibrations (pre-only, occulted-only) for a systematic band
    alt = []
    for w in ([(-1020.0, -10.0)], [(1780.0, 2580.0)]):
        tl2, _, m2, _, _, _ = background_model(windows=w)
        mm = dict(zip(tl2.astype(int), m2))
        alt.append(np.array([mm[int(x)] for x in e[:-1]]))
    tc = e[:-1] + 0.5
    gross = n / f
    net_c = gross - B_PAPER
    net_p = gross - b
    sys_b = np.abs(alt[0] - alt[1]) / 2
    err = np.hypot(np.sqrt(n) / f, resid)
    res = {"bkg_coef": coef.tolist(), "bkg_resid": float(resid)}

    rows = []
    for lo, hi in [(280, 300), (300, 340), (340, 380), (380, 440), (440, 600),
                   (600, 650), (650, 700), (700, 752)]:
        w = (tc >= lo) & (tc < hi)
        rows.append(dict(win=(lo, hi), net_const=float(net_c[w].mean()),
                         net_proxy=float(net_p[w].mean()),
                         proxy_sys=float(sys_b[w].mean()),
                         bkg_rise=float(b[w].mean() - B_PAPER),
                         proxy_in_range=float(inr[w].mean())))
        print(f"{lo}-{hi}: net(const B) {rows[-1]['net_const']:8.0f}  net(proxy) "
              f"{rows[-1]['net_proxy']:8.0f} ± {rows[-1]['proxy_sys']:.0f} (calib)  "
              f"bkg rise {rows[-1]['bkg_rise']:7.0f}  proxy in range {rows[-1]['proxy_in_range']:.2f}")
    res["net_by_window"] = rows

    # decay fits on 280-380 s
    w = (tc >= 280) & (tc < 380)
    fits = {}
    for nm, fn, p0 in [
        ("pl_tref0", lambda x, A, a: A * (x / 300.0) ** (-a), (1e4, 5)),
        ("pl_tref226", lambda x, A, a: A * ((x - 226.0) / 74.0) ** (-a), (1e4, 2)),
        ("exp", lambda x, A, tau: A * np.exp(-(x - 300.0) / tau), (1e4, 20)),
    ]:
        for bn, y in [("proxy", net_p), ("const", net_c)]:
            p, cov = curve_fit(fn, tc[w], y[w], p0=p0, sigma=err[w], absolute_sigma=True,
                               maxfev=20000)
            chi2 = float(np.sum(((fn(tc[w], *p) - y[w]) / err[w]) ** 2))
            dof = int(w.sum() - 2)
            pe = np.sqrt(np.diag(cov)) * np.sqrt(max(chi2 / dof, 1))
            # background-calibration systematic: refit with the two alternative models
            ps = []
            for a in alt:
                pa, _ = curve_fit(fn, tc[w], (gross - a)[w], p0=p, sigma=err[w], maxfev=20000)
                ps.append(pa[1])
            fits[f"{nm}|{bn}"] = dict(p=p.tolist(), perr=pe.tolist(), chi2=chi2, dof=dof,
                                     index_calib_range=[float(min(ps)), float(max(ps))])
            print(f"decay {nm:11s} bkg={bn:5s}: index {p[1]:.3f} ± {pe[1]:.3f} "
                  f"(calib {min(ps):.3f}..{max(ps):.3f})  chi2/dof {chi2:.0f}/{dof}")
    res["decay_fits"] = fits

    # flare T90 on the proxy-net curve minus the extrapolated tail decay
    pl = fits["pl_tref226|proxy"]["p"]
    tail = pl[0] * ((tc - 226.0) / 74.0) ** (-pl[1])
    out_t90 = {}
    for nm, cont in [("tail_powerlaw", tail), ("zero", 0 * tail)]:
        wf = (tc >= 380) & (tc < 700)
        exc = (net_p - cont)[wf]
        cum = np.cumsum(exc); tt = tc[wf]
        tot = cum[(tt > 640) & (tt < 700)].mean() if nm == "zero" else cum[(tt > 620) & (tt < 660)].mean()
        t05 = tt[np.argmax(cum >= 0.05 * tot)]; t95 = tt[np.argmax(cum >= 0.95 * tot)]
        out_t90[nm] = dict(t05=float(t05), t95=float(t95), T90=float(t95 - t05),
                           counts=float(tot))
        print(f"flare T90 (continuum={nm}): {t05:.1f}..{t95:.1f} = {t95-t05:.1f} s, counts {tot:.3g}")
    res["flare_t90"] = out_t90

    w = (tc >= 180) & (tc < 300)
    res["integral_180_300"] = dict(const=float(net_c[w].sum()), proxy=float(net_p[w].sum()))
    print("integral 180-300:", res["integral_180_300"])
    (OUT / "tail.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(tc, net_c, lw=0.8, label="net, constant pre-burst background (paper)")
    ax[0].plot(tc, net_p, lw=0.8, label="net, LargeEvt orbital-background model")
    ax[0].fill_between(tc, net_p - sys_b, net_p + sys_b, color="C1", alpha=0.25, lw=0)
    ax[0].set_yscale("symlog", linthresh=1e3); ax[0].set_ylim(-3e3, 1e7)
    ax[0].axhline(0, color="0.5", lw=0.5); ax[0].legend(fontsize=7)
    ax[0].set_ylabel("evt/s")
    ax[1].plot(tc, b, color="C3", label="background model")
    ax[1].axhline(B_PAPER, color="C0", ls=":", label="constant B = 9332")
    ax[1].plot(tc, np.where(inr, np.nan, b), color="k", lw=2, alpha=0.4,
               label="proxy outside calibrated range")
    ax[1].set_ylabel("background evt/s"); ax[1].set_xlabel("T - T0 (s)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "tail.png", dpi=130)


if __name__ == "__main__":
    main()
