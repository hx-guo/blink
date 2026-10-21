#!/usr/bin/env python3
"""Pulse structure of the GRB 221009A main emission and flare from the HE
1-s dead-time-corrected light curve.

Light curve: gross HE rate (observed + gap-filled, all channels, as in the
paper) divided by the 1-Hz engineering live fraction, minus the orbital
background model of p221009_timing_bkg (LargeEvt proxy). Bins overlapping the
HE detector-off intervals (first/last partial seconds included) are dropped.

Per-bin uncertainty: Poisson of the gross counts / f_live, plus a 12%
fractional systematic (the robust scatter of the paper's HXMT/GECAM ratio on
the main pulse), added in quadrature.

Analyses:
  1. Bayesian blocks ('measures' fitness, p0 = 0.01) over T0+170..700 s.
  2. Sum-of-Norris-pulse fits (Norris et al. 2005,
     I = A*lam*exp(-tau1/(t-ts) - (t-ts)/tau2), lam = exp(2 sqrt(tau1/tau2)))
     to the main emission (T0+174..340 s) and to the flare (T0+440..600 s),
     least squares (scipy), 1-sigma from the covariance matrix. Derived:
     t_peak = ts + sqrt(tau1 tau2), width w = tau2 sqrt(1 + 4 sqrt(tau1/tau2)),
     asymmetry k = (1 + 4 sqrt(tau1/tau2))^-1/2.
  3. The same pulse model fitted to the GECAM-C GRD01 LG 1-s curve (no gaps),
     to show what the HE detector-off intervals hide.

Outputs: data/p221009/timing/pulses.{json,png}
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from astropy.stats import bayesian_blocks

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from p221009_timing_common import (OUT, he_events, eng_1hz, flive_on, gecam_btime,
                                   rebin_series)
from p221009_timing_bkg import background_clean as background_model

OFF = [(229.0, 249.0), (267.0, 277.0)]     # whole seconds touched by HE-off
SYS = 0.12


def he_lc(lo=170.0, hi=752.0):
    t, fill, ch, box = he_events()
    keep = ~((ch == 0) & (np.abs(t - np.round(t)) < 1e-7))
    t = t[keep]
    et, fl, _ = eng_1hz()
    e = np.arange(lo, hi + 0.5, 1.0)
    n = np.histogram(t, e)[0].astype(float)
    f = flive_on(e, et, fl)
    tl, r, model, ok, coef, resid = background_model()
    bmap = dict(zip(tl.astype(int), model))
    b = np.array([bmap[int(x)] for x in e[:-1]])
    rate = n / f - b
    err = np.hypot(np.sqrt(n) / f, SYS * np.abs(rate))
    err = np.hypot(err, resid)                 # background-model floor
    good = np.ones(len(rate), bool)
    for a, c in OFF:
        good &= ~((e[:-1] >= a) & (e[:-1] < c))
    return e[:-1] + 0.5, rate, err, good, f


def norris(t, A, ts, t1, t2):
    x = np.clip(t - ts, 1e-9, None)
    lam = np.exp(2 * np.sqrt(t1 / t2))
    y = A * lam * np.exp(-t1 / x - x / t2)
    return np.where(t > ts, y, 0.0)


def model(t, p):
    return sum(norris(t, *p[4 * i:4 * i + 4]) for i in range(len(p) // 4))


def fit(t, y, e, p0, lo_b, hi_b):
    r = least_squares(lambda p: (model(t, p) - y) / e, p0, bounds=(lo_b, hi_b),
                      x_scale="jac", max_nfev=20000)
    J = r.jac
    dof = len(y) - len(p0)
    chi2 = float(np.sum(r.fun ** 2))
    try:
        cov = np.linalg.inv(J.T @ J) * max(chi2 / dof, 1.0)
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = None
        perr = np.full(len(p0), np.nan)
    return r.x, perr, chi2, dof, cov


def derived(p, cov):
    out = []
    for i in range(len(p) // 4):
        A, ts, t1, t2 = p[4 * i:4 * i + 4]
        mu = np.sqrt(t1 / t2)
        tp = ts + np.sqrt(t1 * t2)
        w = t2 * np.sqrt(1 + 4 * mu)
        k = 1 / np.sqrt(1 + 4 * mu)
        # linear error propagation for t_peak and width
        g_tp = np.array([0, 1, 0.5 * np.sqrt(t2 / t1), 0.5 * np.sqrt(t1 / t2)])
        dw_dt1 = t2 * 0.5 / np.sqrt(1 + 4 * mu) * 4 * 0.5 / np.sqrt(t1 * t2)
        dw_dt2 = np.sqrt(1 + 4 * mu) + t2 * 0.5 / np.sqrt(1 + 4 * mu) * 4 * (-0.5) * np.sqrt(t1) * t2 ** -1.5
        g_w = np.array([0, 0, dw_dt1, dw_dt2])
        if cov is not None:
            C = cov[4 * i:4 * i + 4, 4 * i:4 * i + 4]
            etp = float(np.sqrt(g_tp @ C @ g_tp)); ew = float(np.sqrt(g_w @ C @ g_w))
        else:
            etp = ew = np.nan
        # rise (peak - half max before) / decay (half max after - peak), numerically
        tt = np.linspace(ts, ts + 20 * w, 200001)
        yy = norris(tt, A, ts, t1, t2)
        ym = yy.max(); ip = int(np.argmax(yy))
        above = np.where(yy >= 0.5 * ym)[0]
        rise = tt[ip] - tt[above[0]]; decay = tt[above[-1]] - tt[ip]
        out.append(dict(A=float(A), t_peak=float(tp), t_peak_err=etp,
                        fwhm_width=float(w), width_err=ew, asym=float(k),
                        rise_hwhm=float(rise), decay_hwhm=float(decay),
                        ts=float(ts), tau1=float(t1), tau2=float(t2)))
    return out


MAIN_P0 = [  # A, ts, tau1, tau2   (seed pulses read off the light curve)
    (6.0e4, 175.0, 60.0, 3.0),      # T0+187 pulse
    (2.5e4, 185.0, 30.0, 6.0),      # T0+191 shoulder / tail
    (9.0e5, 212.0, 50.0, 2.0),      # T0+222 sub-peak
    (4.5e6, 215.0, 120.0, 3.0),     # P1 (peaks at or after T0+227)
    (2.3e6, 245.0, 150.0, 4.0),     # P2 (T0+262)
    (9.0e4, 240.0, 300.0, 25.0),    # broad underlying / decay component
]
FLARE_P0 = [
    (5.0e3, 380.0, 500.0, 40.0),    # broad flare envelope
    (1.2e4, 440.0, 300.0, 10.0),    # T0+470 sub-pulse
    (1.8e5, 490.0, 200.0, 1.5),     # T0+510 spike
    (2.0e4, 500.0, 300.0, 15.0),    # post-spike shoulder
    (5.0e3, 540.0, 500.0, 15.0),    # T0+580 bump
]


def run_fit(tc, y, e, good, lo, hi, p0):
    m = good & (tc >= lo) & (tc < hi)
    p = np.array(p0, float).ravel()
    lo_b = np.array([[0, lo - 120, 1e-3, 0.05] for _ in p0]).ravel()
    hi_b = np.array([[1e8, hi, 1e4, 500] for _ in p0]).ravel()
    p = np.clip(p, lo_b + 1e-6, hi_b - 1e-6)
    x, xe, chi2, dof, cov = fit(tc[m], y[m], e[m], p, lo_b, hi_b)
    return x, xe, chi2, dof, cov, m


def crossing(tc, y, i0, level, direction):
    """Linear-interpolated time where y crosses level, walking from i0."""
    i = i0
    step = -1 if direction == "before" else 1
    while 0 <= i + step < len(y) and y[i + step] >= level:
        i += step
    j = i + step
    if not (0 <= j < len(y)):
        return np.nan
    return float(tc[j] + (level - y[j]) / (y[i] - y[j]) * (tc[i] - tc[j]))


def empirical(tc, y, good):
    """Model-free pulse metrics on stretches without HE detector-off seconds.

    Only quoted when the half-maximum crossing lies on contiguous good bins.
    """
    out = {}
    for nm, lo, hi in [("pulse_187", 176, 210), ("P1_rise", 210, 229),
                       ("P2", 249, 267), ("flare_spike", 495, 540)]:
        w = np.where(good & (tc >= lo) & (tc < hi))[0]
        i0 = w[np.argmax(y[w])]
        pk = y[i0]
        d = dict(t_peak_bin=float(tc[i0]), peak=float(pk))
        sub_t, sub_y = tc[w], y[w]
        k0 = int(np.argmax(sub_y))
        for frac in (0.5, 0.1):
            tb = crossing(sub_t, sub_y, k0, frac * pk, "before")
            ta = crossing(sub_t, sub_y, k0, frac * pk, "after")
            d[f"t_before_{frac}"] = tb
            d[f"t_after_{frac}"] = ta
        d["fwhm"] = d["t_after_0.5"] - d["t_before_0.5"]
        out[nm] = d
    return out


def main():
    tc, y, e, good, f = he_lc()
    res = {}
    res["empirical"] = empirical(tc, y, good)
    for k, v in res["empirical"].items():
        print(k, {a: round(b, 2) for a, b in v.items()})
    # 1. Bayesian blocks on the dead-time-corrected 1-s curve
    m = good & (tc < 700)
    bb = bayesian_blocks(tc[m], y[m], e[m], fitness="measures", p0=0.01)
    blocks = []
    for a, b in zip(bb[:-1], bb[1:]):
        w = m & (tc >= a) & (tc <= b)
        blocks.append((float(a), float(b), float(np.average(y[w], weights=1 / e[w] ** 2))))
    res["bayesian_blocks"] = blocks

    # 2. pulse fits
    out = {}
    for nm, (lo, hi, p0) in {"main": (174.0, 340.0, MAIN_P0),
                             "flare": (380.0, 640.0, FLARE_P0)}.items():
        x, xe, chi2, dof, cov, mm = run_fit(tc, y, e, good, lo, hi, p0)
        out[nm] = dict(params=x.tolist(), perr=xe.tolist(), chi2=chi2, dof=dof,
                       pulses=derived(x, cov))
        print(f"{nm}: chi2/dof = {chi2:.1f}/{dof}")
        for i, d in enumerate(out[nm]["pulses"]):
            print(f"   pulse {i}: t_peak {d['t_peak']:.2f}±{d['t_peak_err']:.2f}  "
                  f"peak {norris(np.array([d['t_peak']]), d['A'], d['ts'], d['tau1'], d['tau2'])[0]:.3g}  "
                  f"FWHM {d['fwhm_width']:.2f}±{d['width_err']:.2f}  rise {d['rise_hwhm']:.2f} "
                  f"decay {d['decay_hwhm']:.2f}  k {d['asym']:.2f}")
    res["fits"] = out

    # 3. GECAM-C LG 1-s, same main-emission model (no gaps), scaled x200
    tg, cg, lg, rg = gecam_btime(1, range(9, 13), subtract_bkg=True)
    eg = np.arange(170.0, 753.0, 1.0)
    g, ng = rebin_series(tg, rg, eg)
    gc_t = eg[:-1] + 0.5
    cnt = g * 1.0
    ge = np.hypot(np.sqrt(np.abs(cnt) / 0.95), 0.05 * np.abs(g))
    p0g = np.array(out["main"]["params"]).reshape(-1, 4).copy()
    p0g[:, 0] /= 200.0
    xg, xge, chi2g, dofg, covg, _ = run_fit(gc_t, g, ge, np.isfinite(g), 174.0, 340.0, p0g)
    res["gecam_lg_main"] = dict(params=xg.tolist(), chi2=chi2g, dof=dofg,
                                pulses=derived(xg, covg))
    print("GECAM LG main pulses:")
    for d in res["gecam_lg_main"]["pulses"]:
        print(f"   t_peak {d['t_peak']:.2f}  FWHM {d['fwhm_width']:.2f} rise {d['rise_hwhm']:.2f} decay {d['decay_hwhm']:.2f}")
    # brightest seconds
    res["brightest_second"] = {
        "HE_observed": float(tc[good][np.argmax(np.where(tc[good] < 340, y[good], -1))]),
        "GECAM_LG": float(gc_t[np.nanargmax(np.where(gc_t < 340, g, -1))]),
        "GECAM_LG_in_HE_off": [float(gc_t[np.nanargmax(np.where((gc_t > 229) & (gc_t < 249), g, -1))]),
                               float(np.nanmax(np.where((gc_t > 229) & (gc_t < 249), g, -1)))],
        "GECAM_LG_at_227.5": float(g[np.argmin(np.abs(gc_t - 227.5))]),
    }
    print("brightest:", res["brightest_second"])
    (OUT / "pulses.json").write_text(json.dumps(res, indent=1))

    # figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for a, (lo, hi, nm) in zip(ax, [(174, 340, "main"), (380, 640, "flare")]):
        w = (tc >= lo) & (tc < hi)
        a.errorbar(tc[w & good], y[w & good], e[w & good], fmt=".", ms=3, color="C0",
                   lw=0.5, label="HE 1 s, dead-time corrected, net")
        tt = np.linspace(lo, hi, 4000)
        p = np.array(out[nm]["params"])
        a.plot(tt, model(tt, p), color="k", lw=1, label="sum of Norris pulses")
        for i in range(len(p) // 4):
            a.plot(tt, norris(tt, *p[4 * i:4 * i + 4]), lw=0.7, ls="--")
        if nm == "main":
            a.step(gc_t, g * 200, where="mid", color="C1", lw=0.7, alpha=0.8,
                   label="GECAM-C LG x200")
            for x0, x1 in OFF:
                a.axvspan(x0, x1, color="0.85", zorder=0)
            a.set_yscale("log"); a.set_ylim(2e3, 1e7)
        for bl in blocks:
            if bl[1] > lo and bl[0] < hi:
                a.hlines(bl[2], max(bl[0], lo), min(bl[1], hi), color="C3", lw=1.2)
        a.set_xlim(lo, hi); a.set_ylabel("evt/s"); a.legend(fontsize=7)
    ax[1].set_xlabel("T - T0 (s)")
    fig.tight_layout()
    fig.savefig(OUT / "pulses.png", dpi=130)


if __name__ == "__main__":
    main()
