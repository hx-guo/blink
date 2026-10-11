"""Direct test of the Song & Zhang (2023) time-resolved precursor CPL fits
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
# optional HE response tilt (E/1 MeV)^delta (first argument), referee-2 M2
DELTA = float(sys.argv[1].lstrip("d")) if len(sys.argv) > 1 else 0.0   # e.g. d-0.1
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
for t1, t2, ep_sz, al_sz, al_assumed in SZ:
    fts = {}
    for key, dets in {"NaI48": ["n4", "n8"], "n8": ["n8"], "b1": ["b1"], "b0": ["b0"], "joint": ["n4", "n8", "b1"]}.items():
        er = {"n8": (10.0, 900.0)} if key == "n8" else {}
        P, B, R, M = zip(*[DD[d].pieces(t1, t2, er.get(d, ER[d])) for d in dets])
        fts[key] = SpectralFitterPgstat(list(P), list(B), list(R), channel_masks=list(M))
    w = [(t1 - C.HXMT_DT, t2 - C.HXMT_DT)]
    src, bkg, bvar, expo = H.counts(T_HE, CH_HE, w)
    fl = H.flive_mean([(np.floor(w[0][0]), np.ceil(w[0][1]))])

    def he_mu(fun, par):
        f = fun(par, EG.ravel()).reshape(EG.shape)
        pf = np.trapezoid(f, EG, axis=1)
        if DELTA:
            pf = pf * (np.sqrt(ELO * EHI) / 1000.0) ** DELTA
        return pf @ RHE * expo * fl

    def chi_he(mu):
        return float(np.sum((src - bkg - mu) ** 2 / (np.maximum(mu + bkg, 1) + bvar)))

    def pg(key, fun, par):
        with np.errstate(all="ignore"):
            v = -2.0 * fts[key]._fold_model(fun, par)
        return v if np.isfinite(v) else 1e12

    # ours: joint COMP, C free
    def S_joint(x):
        if not (-1.99 < x[1] < 2 and np.log(5) < x[2] < np.log(2e5) and np.log(0.3) < x[3] < np.log(3)):
            return 1e12
        par = comp(np.exp(x[0]), np.exp(x[2]), x[1])
        return pg("joint", CO.eval, par) + chi_he(np.exp(x[3]) * he_mu(CO.eval, par))
    bj = None
    for ep0 in (300.0, 1500.0, 8000.0):
        r = nm(S_joint, [np.log(0.02), -1.6, np.log(ep0), np.log(1.2)])
        bj = r if bj is None or r.fun < bj.fun else bj
    pj = comp(np.exp(bj.x[0]), np.exp(bj.x[2]), bj.x[1]); Cj = np.exp(bj.x[3])
    mu_ours = he_mu(CO.eval, pj)
    chi_ours_C = chi_he(Cj * mu_ours)
    k_ours = nm(lambda x: chi_he(np.exp(x[0]) * mu_ours), [np.log(Cj)], 1)
    chi_ours_free = k_ours.fun

    # SZ shape
    mu_sz1 = he_mu(CO.eval, comp(1.0, ep_sz, al_sz))
    k_sz = nm(lambda x: chi_he(np.exp(x[0]) * mu_sz1), [np.log(0.02)], 3)
    chi_sz_free = k_sz.fun
    # SZ normalised to n8 (their NaI) and to n4+n8
    a_n8 = nm(lambda x: pg("n8", CO.eval, comp(np.exp(x[0]), ep_sz, al_sz)), [np.log(0.02)], 3)
    A_n8 = np.exp(a_n8.x[0])
    a_48 = nm(lambda x: pg("NaI48", CO.eval, comp(np.exp(x[0]), ep_sz, al_sz)), [np.log(0.02)], 3)
    # our model in n8 alone, for reference
    pg_n8_ours = nm(lambda x: pg("n8", CO.eval, comp(np.exp(x[0]), pj[1], pj[2])), [np.log(pj[0])], 1).fun
    # free PL / CPL on HE alone
    hp = nm(lambda x: chi_he(he_mu(PL.eval, [np.exp(x[0]), x[1], 100.0])) if -4 < x[1] < 0 else 1e12, [np.log(0.02), -2.0], 3)
    hc = None
    for ep0 in (300.0, 1500.0, 6000.0):
        r = nm(lambda x: chi_he(he_mu(CO.eval, comp(np.exp(x[0]), np.exp(x[2]), x[1])))
               if (-1.99 < x[1] < 2 and np.log(20) < x[2] < np.log(1e5)) else 1e12, [np.log(0.02), -1.5, np.log(ep0)], 2)
        hc = r if hc is None or r.fun < hc.fun else hc
    row = dict(delta=DELTA, t1=t1, t2=t2, Ep_sz=ep_sz, alpha_sz=al_sz, alpha_assumed=al_assumed, he_net=float(src.sum() - bkg.sum()),
               n_groups=len(src),
               ours_Ep=pj[1], ours_alpha=pj[2], ours_C=Cj, chi2_he_ours_atC=chi_ours_C, chi2_he_ours_free=chi_ours_free,
               chi2_he_sz_free=chi_sz_free, dchi2_he_sz_free=chi_sz_free - chi_ours_free,
               he_PL_index=hp.x[1], chi2_he_PL=hp.fun, he_CPL_Ep=np.exp(hc.x[2]), he_CPL_alpha=hc.x[1], chi2_he_CPL=hc.fun,
               A_sz_n8=A_n8, pg_n8_sz=a_n8.fun, pg_n8_ours=pg_n8_ours, A_sz_n48=np.exp(a_48.x[0]),
               he_pred_sz_over_meas=float((A_n8 * mu_sz1).sum() / max(src.sum() - bkg.sum(), 1)))
    for Cn in (0.81, 1.00, 1.19, 1.26):
        mu = Cn * A_n8 * mu_sz1
        row["chi2_he_sz_n8_C%.2f" % Cn] = chi_he(mu)
        row["dchi2_he_sz_n8_C%.2f" % Cn] = chi_he(mu) - chi_ours_C
    # BGO tests
    for b in ("b1", "b0"):
        row["pg_%s_sz_n8" % b] = pg(b, CO.eval, comp(A_n8, ep_sz, al_sz))
        row["pg_%s_ours" % b] = pg(b, CO.eval, pj)
        sf = nm(lambda x: pg(b, CO.eval, comp(np.exp(x[0]), ep_sz, al_sz)), [np.log(A_n8)], 3)
        row["pg_%s_sz_free" % b] = sf.fun
        row["A_%s_sz_free_over_n8" % b] = np.exp(sf.x[0]) / A_n8
        pf = nm(lambda x: pg(b, PL.eval, [np.exp(x[0]), x[1], 100.0]) if -5 < x[1] < 0 else 1e12, [np.log(0.01), -2.0], 3)
        row["pg_%s_PL" % b] = pf.fun; row["%s_PL_index" % b] = pf.x[1]
        row["pg_%s_null" % b] = pg(b, CO.eval, comp(1e-12, 300.0, -1.5))
        row["d_%s_sz_n8_vs_ours" % b] = row["pg_%s_sz_n8" % b] - row["pg_%s_ours" % b]
        row["d_%s_sz_free_vs_PL" % b] = sf.fun - pf.fun
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(od, "sztest%s.csv" % ("" if not DELTA else "_d%+.2f" % DELTA)),
                              index=False, float_format="%.6g")
print("done")
