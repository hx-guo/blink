#!/usr/bin/env python3
"""Tier 3 tooling: forward-folding spectral fit of an HE CsI/NaI count spectrum
through an OGIP response (RSP with EBOUNDS + SPECRESP MATRIX), numpy + scipy
only (no XSPEC). Poisson source with a Gaussian-error background model
(XSPEC "pgstat"), or pure C-stat when the background is Poisson.

Models (photon flux, ph cm^-2 s^-1 keV^-1):
  cpl : K (E/100)^(-alpha) exp(-E (2-alpha) / Ep)          (Ep = nu F_nu peak, alpha > 0 convention)
  band: Band function with (alpha_B, beta_B, Ep), alpha_B in the usual (negative) convention

Two uses:
  --selftest     fold a model, take the folded counts as data (no noise), refit -> must
                 recover the input parameters (validates the folding + minimiser).
  --pha/--bak/--rsp   fit real OGIP files (used here on GRB 260421B, the one burst for
                 which an official HE-CsI DRM exists on this machine, to check the
                 fitter against the SVOM/GRM result of the same burst).

For GRB 250919A / 260226A no DRM for the burst direction exists locally, so the
saturation-recovered spectra cannot be fitted yet; once the DRM arrives, write the
recovered spectrum as a PHA (scripts/talk20_spec_tier3_write_pha.py) and run this.

Run from blink/:
  .venv/bin/python scripts/talk20_spec_tier3_fold_fit.py --selftest --rsp <rsp>
  .venv/bin/python scripts/talk20_spec_tier3_fold_fit.py --pha X.pha --bak X.bak --rsp X.rsp --emin 200 --emax 900
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize


# ────────────────────────── response ──────────────────────────

def load_rsp(path):
    """Return (elo, ehi, R[n_e, n_ch], ch_emin, ch_emax); R is counts/photon per energy bin."""
    with fits.open(path) as h:
        eb = h["EBOUNDS"].data
        m = h["SPECRESP MATRIX"].data if "SPECRESP MATRIX" in h else h["MATRIX"].data
        elo, ehi = m["ENERG_LO"].astype(float), m["ENERG_HI"].astype(float)
        nch = len(eb)
        R = np.zeros((len(elo), nch))
        f_chan_min = h["SPECRESP MATRIX"].header.get("TLMIN4", 1) if "SPECRESP MATRIX" in h else 1
        for i in range(len(elo)):
            ngrp = int(np.atleast_1d(m["N_GRP"][i])[0])
            fch = np.atleast_1d(m["F_CHAN"][i]).astype(int)
            nchn = np.atleast_1d(m["N_CHAN"][i]).astype(int)
            row = np.atleast_1d(m["MATRIX"][i]).astype(float)
            pos = 0
            for g in range(ngrp):
                a = fch[g] - f_chan_min
                n = nchn[g]
                R[i, a:a + n] = row[pos:pos + n]
                pos += n
        return elo, ehi, R, eb["E_MIN"].astype(float), eb["E_MAX"].astype(float)


def load_pha(path):
    with fits.open(path) as h:
        d = h["SPECTRUM"].data; hd = h["SPECTRUM"].header
        counts = d["COUNTS"].astype(float) if "COUNTS" in d.columns.names else d["RATE"].astype(float) * hd["EXPOSURE"]
        err = d["STAT_ERR"].astype(float) if "STAT_ERR" in d.columns.names else None
        return counts, err, float(hd["EXPOSURE"]), bool(hd.get("POISSERR", err is None))


# ────────────────────────── models ──────────────────────────

def cpl(E, K, alpha, Ep):
    return K * (E / 100.0) ** (-alpha) * np.exp(-E * (2.0 - alpha) / Ep)


def band(E, K, a, b, Ep):
    Eb = (a - b) * Ep / (2.0 + a)
    lo = E < Eb
    out = np.empty_like(E)
    out[lo] = K * (E[lo] / 100.0) ** a * np.exp(-E[lo] * (2.0 + a) / Ep)
    out[~lo] = K * ((a - b) * Ep / (100.0 * (2.0 + a))) ** (a - b) * np.exp(b - a) * (E[~lo] / 100.0) ** b
    return out


MODELS = {
    "cpl": dict(f=cpl, names=["K", "alpha", "Ep"], p0=[1.0, 1.0, 200.0],
                bounds=[(1e-6, 1e4), (-1.0, 1.9), (10.0, 5000.0)]),
    "band": dict(f=band, names=["K", "alpha_B", "beta_B", "Ep"], p0=[1.0, -1.0, -2.5, 200.0],
                 bounds=[(1e-6, 1e4), (-1.9, 1.0), (-6.0, -2.01), (10.0, 5000.0)]),
}


def fold(model, pars, elo, ehi, R, tsrc):
    """Expected source counts per channel: Simpson over each energy bin."""
    Em = 0.5 * (elo + ehi)
    f = (model(elo, *pars) + 4.0 * model(Em, *pars) + model(ehi, *pars)) / 6.0 * (ehi - elo)
    return tsrc * (f @ R)


# ────────────────────────── statistics ──────────────────────────

def pgstat(S, m, B, sB, ts, tb):
    """XSPEC pgstat (Poisson source, Gaussian background), background profiled out.
    S, B: counts; sB: background sigma (counts); m: model source RATE * ts, i.e. expected
    source counts. Returns the statistic and the profiled background counts."""
    m = np.maximum(m, 1e-12)
    # work in rate units: model rate mr = m/ts, background rate f
    mr = m / ts
    sig2 = (sB / tb) ** 2
    Br = B / tb
    # derive: L = ts(mr+f) - S ln(ts(mr+f)) + (Br - f)^2/(2 sig2)
    # dL/df = ts - S/(mr+f) - (Br - f)/sig2 = 0
    #  -> f^2 + (mr - Br + ts sig2) f + (ts sig2 mr - S sig2 - Br mr) = 0
    bq = mr - Br + ts * sig2
    cq = ts * sig2 * mr - S * sig2 - Br * mr
    disc = np.maximum(bq * bq - 4.0 * cq, 0.0)
    f = 0.5 * (-bq + np.sqrt(disc))
    zero = S <= 0
    f[zero] = Br[zero] - ts * sig2[zero]
    exact = sig2 <= 0                      # zero background error: background exactly known
    f[exact] = Br[exact]
    f = np.maximum(f, -mr + 1e-12)
    tot = ts * (mr + f)
    gauss = np.zeros_like(f)
    gauss[~exact] = (Br[~exact] - f[~exact]) ** 2 / (2.0 * sig2[~exact])
    stat = 2.0 * (tot - S * np.log(tot) + gauss)
    stat = stat + 2.0 * np.where(S > 0, S * np.log(np.maximum(S, 1e-300)) - S, 0.0)
    return float(np.sum(stat)), f * tb


def cstat(S, m):
    m = np.maximum(m, 1e-12)
    return float(2.0 * np.sum(m - S + np.where(S > 0, S * np.log(S / m), 0.0)))


# ────────────────────────── fitting ──────────────────────────

def fit(model_name, S, B, sB, ts, tb, elo, ehi, R, use, p0=None, fixed=None):
    M = MODELS[model_name]
    f = M["f"]
    p0 = list(M["p0"] if p0 is None else p0)
    fixed = fixed or {}
    free = [i for i in range(len(p0)) if M["names"][i] not in fixed]

    def unpack(q):
        p = list(p0)
        for k, i in enumerate(free):
            p[i] = q[k]
        for n, v in fixed.items():
            p[M["names"].index(n)] = v
        return p

    def stat(q):
        p = unpack(q)
        for k, i in enumerate(free):
            lo, hi = M["bounds"][i]
            if not (lo <= q[k] <= hi):
                return 1e30
        m = fold(f, p, elo, ehi, R, ts)[use]
        if B is None:
            return cstat(S[use], m)
        return pgstat(S[use], m, B[use], sB[use], ts, tb)[0]

    q0 = np.array([p0[i] for i in free], float)
    # log-K start from the data normalisation
    r = minimize(stat, q0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=20000, maxfev=40000))
    r = minimize(stat, r.x, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=20000, maxfev=40000))
    best = unpack(r.x)
    # 1-sigma profile intervals (delta stat = 1) for each free parameter
    errs = {}
    for k, i in enumerate(free):
        name = M["names"][i]
        lo, hi = M["bounds"][i]

        def prof(v):
            fx = dict(fixed); fx[name] = v
            sub_free = [j for j in free if j != i]
            q_start = np.array([best[j] for j in sub_free], float)

            def st2(q):
                p = list(best)
                for kk, j in enumerate(sub_free):
                    p[j] = q[kk]
                    l2, h2 = M["bounds"][j]
                    if not (l2 <= q[kk] <= h2):
                        return 1e30
                p[i] = v
                m = fold(f, p, elo, ehi, R, ts)[use]
                return cstat(S[use], m) if B is None else pgstat(S[use], m, B[use], sB[use], ts, tb)[0]
            if len(sub_free) == 0:
                return st2(np.zeros(0))
            rr = minimize(st2, q_start, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=5000))
            return rr.fun

        def bracket(direction):
            step = 0.05 * abs(best[i]) if best[i] != 0 else 0.05
            v = best[i]
            for _ in range(60):
                v2 = v + direction * step
                if not (lo <= v2 <= hi):
                    return None
                if prof(v2) - r.fun > 1.0:
                    # bisection between v and v2
                    a, b_ = v, v2
                    for _ in range(25):
                        mid = 0.5 * (a + b_)
                        if prof(mid) - r.fun > 1.0:
                            b_ = mid
                        else:
                            a = mid
                    return 0.5 * (a + b_)
                v = v2
                step *= 1.5
            return None
        errs[name] = (bracket(-1), bracket(+1))
    return dict(model=model_name, names=M["names"], best=best, stat=float(r.fun),
                dof=int(np.sum(use)) - len(free), err1sig=errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsp", required=True)
    ap.add_argument("--pha"); ap.add_argument("--bak")
    ap.add_argument("--model", default="cpl", choices=list(MODELS))
    ap.add_argument("--emin", type=float, default=None); ap.add_argument("--emax", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--p0", type=float, nargs="*")
    ap.add_argument("--json", help="write result JSON here")
    ap.add_argument("--fix", nargs="*", default=[], help="freeze parameters, e.g. alpha=1.284")
    args = ap.parse_args()
    fixed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.fix}

    elo, ehi, R, cmin, cmax = load_rsp(args.rsp)
    use = np.ones(len(cmin), bool)
    if args.emin is not None:
        use &= cmin >= args.emin
    if args.emax is not None:
        use &= cmax <= args.emax
    use &= R.sum(axis=0) > 0

    if args.selftest:
        M = MODELS[args.model]
        truth = list(M["p0"]) if not args.p0 else list(args.p0)
        ts = 10.0
        S = fold(M["f"], truth, elo, ehi, R, ts)
        res = fit(args.model, S, None, None, ts, ts, elo, ehi, R, use,
                  p0=[v * 1.3 for v in truth])
        print("selftest truth:", dict(zip(M["names"], truth)))
        print("selftest fit  :", {n: round(v, 4) for n, v in zip(res["names"], res["best"])},
              "stat", round(res["stat"], 4), "dof", res["dof"])
        ok = all(abs(a - b) / max(abs(b), 1e-9) < 1e-3 for a, b in zip(res["best"], truth))
        print("selftest", "PASS" if ok else "FAIL")
        return

    S, _, ts, _ = load_pha(args.pha)
    if args.bak:
        B, sB, tb, bpois = load_pha(args.bak)
        if bpois or sB is None:
            sB = np.sqrt(np.maximum(B, 1.0))
    else:
        B = sB = None; tb = ts
    res = fit(args.model, S, B, sB, ts, tb, elo, ehi, R, use, p0=args.p0, fixed=fixed)
    out = {k: v for k, v in res.items()}
    out["fixed"] = fixed
    out["n_channels"] = int(use.sum())
    out["energy_range"] = [float(cmin[use].min()), float(cmax[use].max())]
    if args.model == "cpl":
        K, a, Ep = res["best"]
        out["Ec"] = Ep / (2.0 - a)
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
