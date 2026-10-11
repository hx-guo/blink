"""Referee 2 M6: can the time-resolved precursor model reproduce the published
time-integrated Ep values (243 keV ... 4 MeV)?

Deterministic (Asimov, no noise) synthetic spectra: the piecewise time-resolved
model (GBM+HE joint COMP fits, DT_HE = 0.0108, for -1...8.576 s; GBM-only COMP
for 8.576-13 and 13-20 s; GBM-only power law for 20-30 s) is integrated over
each literature interval and folded through the GBM (and HE) responses, the
real background expectation is added, and the result is refitted with the
literature's models (Band, COMP) and detector/energy selections:
  SZ    Song & Zhang 2023: 0-10 s, n8 (10-900 keV) + b0 (0.385-38 MeV) + HE
        CsI (ch 20-255, C = 1.19); Band (alpha -1.55, beta -2.02, Ep 243 keV)
  Ror   Ror+2024: 0-10 s, n7 (8-900 keV) + b1 (0.25-38 MeV); Band (Ep 1531 keV,
        beta -3.09)
  LesIa Lesage+2023 Ia: -0.003-8.576 s, n4+n8 (20-900 keV) + b1 (0.4-38 MeV);
        COMP (Ep 3980 keV; their fit includes LAT-LLE, not reproducible here)
  KW    Frederiks+2023: KW 0-24.832 s = GBM 0.36-25.19 s, emulated with
        n4+n8+b1 restricted to 20 keV-10 MeV (no KW response); CPL (Ep 970 keV)
  KW8   KW 0-8.4 s (Ep ~1.5 MeV)
Statistic: PG-stat (GBM) + chi2 (HE).  Outputs $GBM_OUT/synth/synth_fits.csv
and synth_epprofile.csv (python p221009_gbm_synth.py [CASE ...] -> _CASE files).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gdt.core.spectra.fitting import pgstat
from gdt.core.spectra.functions import Comptonized, Band, PowerLaw
import p221009_gbm_common as C
from p221009_gbm_specfit import DetData

sys.path.insert(0, "/scratchfs2/gecam/guohx/p221009/blink/scripts")
import p221009_resp_spec as H     # noqa: E402

od = C.outdir("synth")
J = pd.read_csv("/scratchfs2/gecam/guohx/p221009/gbm/runs/joint3/joint/joint_fits.csv")
J = J[(J.model == "COMP") & (J["var"] == "nominal")]
PR = pd.read_csv("/scratchfs2/gecam/guohx/p221009/gbm/runs/prec4/precursor/prec_fits.csv")
CO, BA, PLF = Comptonized(), Band(), PowerLaw()

# piecewise model: (t1, t2, function of energy)
PIECES = []
for t1, t2 in [(-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5), (4.5, 6.5), (6.5, 8.576)]:
    r = J[(J.tag == "tr") & np.isclose(J.t1, t1) & np.isclose(J.t2, t2) & (J.data == "GBM+HE")].iloc[0]
    PIECES.append((t1, t2, [r.A, min(r.Ep, 1e5), r.alpha, 100.0], "COMP"))
for t1, t2 in [(8.576, 13.0), (13.0, 20.0)]:
    r = J[(J.tag == "tr") & np.isclose(J.t1, t1) & np.isclose(J.t2, t2) & (J.data == "GBM")].iloc[0]
    PIECES.append((t1, t2, [r.A, min(r.Ep, 1e5), r.alpha, 100.0], "COMP"))
r = PR[(PR.tag == "tr") & np.isclose(PR.t1, 20.0) & (PR.model == "PL")].iloc[0]
PIECES.append((20.0, 30.0, [r.A, r["index"], 100.0], "PL"))
C_HE = 1.19
for p in PIECES:
    print("piece", p, flush=True)


def piece_fun(p):
    return (lambda par, e: CO.eval(par, e)) if p[3] == "COMP" else (lambda par, e: PLF.eval(par, e))


CASES = {
    "SZ": (0.0, 10.0, {"n8": (10.0, 900.0), "b0": (385.0, 38000.0)}, True,
           dict(Band=(243.0, -1.55, -2.02), COMP=None)),
    "Ror": (0.0, 10.0, {"n7": (8.0, 900.0), "b1": (250.0, 38000.0)}, False, dict(Band=(1531.0, -1.661, -3.09), COMP=None)),
    "LesIa": (-0.003, 8.576, {"n4": (20.0, 900.0), "n8": (20.0, 900.0), "b1": (400.0, 38000.0)}, False,
              dict(COMP=(3980.0, -1.69), Band=None)),
    "KW": (0.36, 25.19, {"n4": (20.0, 900.0), "n8": (20.0, 900.0), "b1": (400.0, 10000.0)}, False,
           dict(COMP=(970.0, -1.62), Band=None)),
    "KW8": (0.36, 8.76, {"n4": (20.0, 900.0), "n8": (20.0, 900.0), "b1": (400.0, 10000.0)}, False,
            dict(COMP=(1495.0, -1.65), Band=None)),
}
DD = {}
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
RHE = H.group_drm(z["precursor_per"].astype(float))
EG = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * np.linspace(0, 1, 9)[None, :])
T_HE, CH_HE = H.load_events()


def he_fold(fun, par):
    f = fun(par, EG.ravel()).reshape(EG.shape)
    return np.trapezoid(f, EG, axis=1) @ RHE


ONLY = sys.argv[1:] or list(CASES)
rows, prow = [], []
for case, (t1, t2, dets, use_he, lit) in CASES.items():
    if case not in ONLY:
        continue
    sets = []
    for d, er in dets.items():
        if d not in DD:
            DD[d] = DetData(d, [(-250.0, -10.0), (95.0, 118.0)], 2)
        pha, bak, rsp, mask = DD[d].pieces(t1, t2, er)
        expo = pha.exposure
        live = expo / (t2 - t1)
        src = np.zeros(pha.num_chans)
        for p in PIECES:
            ov = max(0.0, min(t2, p[1]) - max(t1, p[0]))
            if ov > 0:
                src += rsp.drm.fold_spectrum(piece_fun(p), p[2]) * ov * live
        bc = bak.rates * expo; bv = (bak.rate_uncertainty * expo) ** 2
        obs = bc + src
        sets.append((d, rsp, mask, obs, bc, bv, expo))
    if use_he:
        w = [(t1 - C.HXMT_DT, t2 - C.HXMT_DT)]
        _, hbkg, hbvar, hexpo = H.counts(T_HE, CH_HE, w)
        fl = H.flive_mean([(np.floor(w[0][0]), np.ceil(w[0][1]))])
        hsrc = np.zeros(RHE.shape[1])
        for p in PIECES:
            ov = max(0.0, min(w[0][1], p[1] - C.HXMT_DT) - max(w[0][0], p[0] - C.HXMT_DT))
            if ov > 0:
                hsrc += C_HE * he_fold(piece_fun(p), p[2]) * ov * fl
        hobs = hbkg + hsrc

    def stat(fun, par):
        s = 0.0
        for d, rsp, mask, obs, bc, bv, expo in sets:
            m = rsp.drm.fold_spectrum(fun, par)[mask]
            s += -2.0 * pgstat(obs[mask], m, expo, bc[mask], bv[mask], expo)
        if use_he:
            mu = C_HE * he_fold(fun, par) * hexpo * fl
            s += np.sum((hobs - hbkg - mu) ** 2 / (np.maximum(mu + hbkg, 1) + hbvar))
        return s

    def fit(model, ep_fixed=None, beta_fixed=None, alpha_fixed=None):
        def full(x):
            x = list(x)
            A = np.exp(x.pop(0))
            al = alpha_fixed if alpha_fixed is not None else x.pop(0)
            ep = ep_fixed if ep_fixed is not None else np.exp(x.pop(0))
            if model == "COMP":
                return [A, ep, al, 100.0]
            b = beta_fixed if beta_fixed is not None else x.pop(0)
            return [A, ep, al, b, 100.0]
        fun = CO.eval if model == "COMP" else BA.eval

        def obj(x):
            par = full(x)
            if not (-1.99 < par[2] < 2.0 and 5.0 < par[1] < 2e5):
                return 1e12
            if model == "Band" and not -10 < par[3] < -2.0001:
                return 1e12
            with np.errstate(all="ignore"):
                v = stat(fun, par)
            return v if np.isfinite(v) else 1e12
        best = None
        nb = model == "Band" and beta_fixed is None
        quick = ep_fixed is not None
        for lnA in ((np.log(0.01),) if quick else (np.log(0.02), np.log(0.005))):
            for ep in ((None,) if ep_fixed else (200.0, 1000.0, 4000.0)):
                for b in (((-2.3,) if quick else (-2.1, -3.0)) if nb else (None,)):
                    x = [lnA] + ([] if alpha_fixed is not None else [-1.5]) + ([] if ep is None else [np.log(ep)]) + \
                        ([] if b is None else [b])
                    r = minimize(obj, x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6, adaptive=True))
                    r = minimize(obj, r.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7, adaptive=True))
                    if best is None or r.fun < best.fun:
                        best = r
        return best.fun, full(best.x)

    for model in ("Band", "COMP"):
        s0, par = fit(model)
        # Ep profile (Asimov: delta = 1 gives the expected 1-sigma range)
        grid = np.geomspace(60.0, 60000.0, 25)
        prof = np.array([fit(model, ep_fixed=g)[0] for g in grid])
        for g, s in zip(grid, prof):
            prow.append(dict(case=case, model=model, Ep=g, stat=s))
        y = prof - min(s0, prof.min()); ok = np.where(y < 1.0)[0]
        lo = np.exp(np.interp(1.0, [y[ok[0]], y[ok[0] - 1]], np.log(grid[[ok[0], ok[0] - 1]]))) if len(ok) and ok[0] > 0 else np.nan
        hi = np.exp(np.interp(1.0, [y[ok[-1]], y[ok[-1] + 1]], np.log(grid[[ok[-1], ok[-1] + 1]]))) \
            if len(ok) and ok[-1] < len(grid) - 1 else np.nan
        row = dict(case=case, t1=t1, t2=t2, dets="+".join(dets) + ("+HE" if use_he else ""), model=model, stat=s0,
                   Ep=par[1], alpha=par[2], beta=par[3] if model == "Band" else np.nan, Ep_lo1=lo, Ep_hi1=hi)
        # the literature shape, normalisation free: delta stat
        L = lit.get(model)
        if L:
            if model == "Band":
                sl, _ = fit("Band", ep_fixed=L[0], alpha_fixed=L[1], beta_fixed=L[2])
            else:
                sl, _ = fit("COMP", ep_fixed=L[0], alpha_fixed=L[1])
            row.update(lit_Ep=L[0], dstat_lit_shape=sl - s0)
        rows.append(row)
        print(case, model, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        suf = "" if len(ONLY) == len(CASES) else "_" + "-".join(ONLY)
        pd.DataFrame(rows).to_csv(os.path.join(od, "synth_fits%s.csv" % suf), index=False, float_format="%.6g")
        pd.DataFrame(prow).to_csv(os.path.join(od, "synth_epprofile%s.csv" % suf), index=False, float_format="%.6g")
print("done")
