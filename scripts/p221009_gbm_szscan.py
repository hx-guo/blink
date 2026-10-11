"""Referee round 2 R5: robustness of the Song & Zhang (2023) CPL test
(p221009_gbm_sztest.py) to the figure read-off: Ep scaled by 0.8-1.2 and
alpha scanned over -1.3...-1.8, for HE response tilts delta = 0, +-0.1.
Reports, per interval, the HE shape-only delta chi2 (HE normalisation free)
and the delta chi2 with the normalisation from GBM n8 x C = 1.26 (the most
favourable cross-calibration), relative to our joint best model, and the
least favourable (minimum) values over the grid.  Output
$GBM_OUT/sztest/szscan.csv.

Original description of the test:
Direct test of the Song & Zhang (2023) time-resolved precursor CPL fits
(Fig. 2e, read off by the literature group, precursor_lit.md section 9.1)
against HE/CsI and GBM BGO.

Intervals (GBM T0) and CPL (Ep keV, alpha): 0-1.2 (299, -1.45), 1.2-2.4
(239, -1.51), 2.4-3.9 (178, -1.60), 3.9-6.7 (144, -1.6*), 6.7-10 (118, -1.6*)
(* alpha not given in the figure, -1.6 assumed).
For each interval:
  ours   : GBM (n4+n8 20-900 keV, b1 0.4-38 MeV) + HE COMP joint fit, C free
           (HE window = GBM - C.HXMT_DT);
  HE     : chi2 of the HE spectrum (27 groups, ch 20-255) for
           (a) the SZ shape with the HE normalisation free (shape-only test,
               independent of the HE/GBM cross-calibration),
           (b) the SZ shape normalised to GBM n8 (10-900 keV, their NaI) times
               C = 0.81 / 1.00 / 1.19 / 1.26,
           (c) our joint best model, normalisation free and at its fitted C,
           (d) a free power law and a free CPL fitted to HE alone;
  BGO    : PG-stat of b1 and of b0 (0.385-38 MeV, their BGO) for the SZ shape
           normalised to n8, and for the SZ shape with free normalisation,
           compared with a free power law (BGO alone) and with our joint model.
Optional argument: HE response tilt delta, (E/1 MeV)^delta -> sztest_d+0.10.csv.
Output $GBM_OUT/sztest/sztest.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import Comptonized, PowerLaw
import p221009_gbm_common as C
from p221009_gbm_specfit import DetData

sys.path.insert(0, "/scratchfs2/gecam/guohx/p221009/blink/scripts")
import p221009_resp_spec as H     # noqa: E402

od = C.outdir("sztest")
DELTA = 0.0
SZ = [(0.0, 1.2, 299.0, -1.45, False), (1.2, 2.4, 239.0, -1.51, False), (2.4, 3.9, 178.0, -1.60, False),
      (3.9, 6.7, 144.0, -1.60, True), (6.7, 10.0, 118.0, -1.60, True)]
BKG = ([(-250.0, -10.0), (95.0, 118.0)], 2)
ER = {"n4": (20.0, 900.0), "n8": (20.0, 900.0), "b1": (400.0, 38000.0), "b0": (385.0, 38000.0)}
DD = {d: DetData(d, *BKG) for d in ["n4", "n8", "b1", "b0"]}
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
RHE = H.group_drm(z["precursor_per"].astype(float))
EG = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * np.linspace(0, 1, 9)[None, :])
T_HE, CH_HE = H.load_events()
CO, PL = Comptonized(), PowerLaw()


def comp(A, ep, al):
    return [A, ep, al, 100.0]


def nm(f, x0, n=2):
    best = None
    for k in range(n):
        st = np.array(x0, float) + (k * 0.7) * (np.arange(len(x0)) == 0)
        r = minimize(f, st, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7, adaptive=True))
        r = minimize(f, r.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-8, fatol=1e-8, adaptive=True))
        if best is None or r.fun < best.fun:
            best = r
    return best


rows = []
EPF = [0.8, 0.9, 1.0, 1.1, 1.2]
ALS = [-1.3, -1.4, -1.5, -1.6, -1.7, -1.8]
for t1, t2, ep_sz, al_sz, al_assumed in SZ:
    fts = {}
    for key, dets in {"n8": ["n8"], "joint": ["n4", "n8", "b1"]}.items():
        er = {"n8": (10.0, 900.0)} if key == "n8" else {}
        P, B, R, M = zip(*[DD[d].pieces(t1, t2, er.get(d, ER[d])) for d in dets])
        fts[key] = SpectralFitterPgstat(list(P), list(B), list(R), channel_masks=list(M))
    w = [(t1 - C.HXMT_DT, t2 - C.HXMT_DT)]
    src, bkg, bvar, expo = H.counts(T_HE, CH_HE, w)
    fl = H.flive_mean([(np.floor(w[0][0]), np.ceil(w[0][1]))])

    def he_mu(fun, par, delta):
        f = fun(par, EG.ravel()).reshape(EG.shape)
        pf = np.trapezoid(f, EG, axis=1)
        if delta:
            pf = pf * (np.sqrt(ELO * EHI) / 1000.0) ** delta
        return pf @ RHE * expo * fl

    def chi_he(mu):
        return float(np.sum((src - bkg - mu) ** 2 / (np.maximum(mu + bkg, 1) + bvar)))

    def pg(key, fun, par):
        with np.errstate(all="ignore"):
            v = -2.0 * fts[key]._fold_model(fun, par)
        return v if np.isfinite(v) else 1e12

    for delta in (0.0, 0.1, -0.1):
        def S_joint(x):
            if not (-1.99 < x[1] < 2 and np.log(5) < x[2] < np.log(2e5) and np.log(0.3) < x[3] < np.log(3)):
                return 1e12
            par = comp(np.exp(x[0]), np.exp(x[2]), x[1])
            return pg("joint", CO.eval, par) + chi_he(np.exp(x[3]) * he_mu(CO.eval, par, delta))
        bj = None
        for ep0 in (300.0, 1500.0, 8000.0):
            r = nm(S_joint, [np.log(0.02), -1.6, np.log(ep0), np.log(1.2)])
            bj = r if bj is None or r.fun < bj.fun else bj
        pj = comp(np.exp(bj.x[0]), np.exp(bj.x[2]), bj.x[1]); Cj = np.exp(bj.x[3])
        mu_ours = he_mu(CO.eval, pj, delta)
        chi_ours = nm(lambda x: chi_he(np.exp(x[0]) * mu_ours), [np.log(Cj)], 1).fun
        chi_ours_C = chi_he(Cj * mu_ours)
        for fe in EPF:
            for al in ALS:
                ep = ep_sz * fe
                mu1 = he_mu(CO.eval, comp(1.0, ep, al), delta)
                chi_sz = nm(lambda x: chi_he(np.exp(x[0]) * mu1), [np.log(0.02)], 3).fun
                a8 = nm(lambda x: pg("n8", CO.eval, comp(np.exp(x[0]), ep, al)), [np.log(0.02)], 2)
                chi_n8 = chi_he(1.26 * np.exp(a8.x[0]) * mu1)
                rows.append(dict(t1=t1, t2=t2, delta=delta, Ep=ep, Ep_factor=fe, alpha=al, alpha_read=al_sz,
                                 alpha_assumed=al_assumed, chi2_ours=chi_ours, chi2_sz_free=chi_sz,
                                 dchi2_shape=chi_sz - chi_ours, chi2_sz_n8_C126=chi_n8,
                                 dchi2_n8_C126=chi_n8 - chi_ours_C, pg_n8_sz=a8.fun,
                                 he_pred_over_meas_C126=float(1.26 * np.exp(a8.x[0]) * mu1.sum() / max(src.sum() - bkg.sum(), 1))))
        g = pd.DataFrame([r for r in rows if r["t1"] == t1 and r["delta"] == delta])
        print(t1, t2, "delta", delta, "min dchi2 shape %.1f at Ep %.0f alpha %.1f; min dchi2 n8xC1.26 %.1f" % (
            g.dchi2_shape.min(), g.Ep[g.dchi2_shape.idxmin()], g.alpha[g.dchi2_shape.idxmin()], g.dchi2_n8_C126.min()),
            flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(od, "szscan.csv"), index=False, float_format="%.6g")
print("done")
