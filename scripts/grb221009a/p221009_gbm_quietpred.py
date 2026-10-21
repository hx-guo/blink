"""Referee 2 M3: HE/CsI predictions for the quiescent interval (Lesage region
II, 121.219-164.228 s GBM time) from GBM-constrained models, compared with
the HE measurement +6 +- 16 (stat) +- 15 (sys) evt/s in CsI ch 20-255
(writer3.md / figs_v4.md, T0paper window 121.208-164.217 s).

Models, normalisations fitted to GBM n4+n8 (20-500 keV) + b1 (0.4-38 MeV)
with the orbital background (+/-30 and +/-15 orbits, p221009_gbm_quietspec):
  L_BBCOMP : Lesage+2023 region II, BB (kT = 19.27 keV) + COMP (alpha = -0.4,
             Ep = 5722 keV), both amplitudes free (shapes fixed);
  L_COMP   : the COMP component of L_BBCOMP alone (its fitted amplitude);
  L_COMPonly: COMP (alpha -0.4, Ep 5722 keV) alone, amplitude fitted;
  PL       : free power law (the "Gamma ~ -2 extending to MeV" case).
HE rate = C x (photon spectrum folded through the precursor-direction DRM,
ch 20-255) x live fraction, for C = 0.81, 1.00, 1.12.  The GBM-fit
uncertainty of each prediction is from the delta-PG = 1 profile of the
predicted rate.  Output $GBM_OUT/quiet/quietpred.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import PowerLaw, Comptonized, BlackBody
import p221009_gbm_common as C
import p221009_gbm_quietspec as Q

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
import p221009_resp_spec as H     # noqa: E402

T1, T2 = 121.219, 164.228
ER = {"n4": (20.0, 500.0), "n8": (20.0, 500.0), "b1": (400.0, 38000.0)}
HE_RATE, HE_STAT, HE_SYS = 6.0, 16.0, 15.0
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
RHE = H.group_drm(z["precursor_per"].astype(float)).sum(1)          # ch 20-255
FL = H.flive_mean([(np.floor(T1 - C.HXMT_DT), np.ceil(T2 - C.HXMT_DT))])
EG = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * np.linspace(0, 1, 9)[None, :])
BB, CO, PL = BlackBody(), Comptonized(), PowerLaw()


def he_rate(fun):
    f = fun(EG.ravel()).reshape(EG.shape)
    return float(np.trapezoid(f, EG, axis=1) @ RHE * FL)


MODELS = {
    # name: (parameter vector -> photon spectrum function, start, which part predicts HE)
    "L_BBCOMP": (lambda x: (lambda e: BB.eval([np.exp(x[0]), 19.27], e) + CO.eval([np.exp(x[1]), 5722.0, -0.4, 100.0], e)),
                 [np.log(1e-4), np.log(1e-3)]),
    "L_COMPonly": (lambda x: (lambda e: CO.eval([np.exp(x[0]), 5722.0, -0.4, 100.0], e)), [np.log(1e-3)]),
    "PL": (lambda x: (lambda e: PL.eval([np.exp(x[0]), x[1], 100.0], e)), [np.log(1e-2), -2.0]),
}

rows = []
for bkg, ks in [("orb30", (-30, 30)), ("orb15", (-15, 15))]:
    GD = [Q.OrbData(d, ks) for d in ER]
    P_, B_, R_, M_ = zip(*[g.pieces(T1, T2, ER[g.det]) for g in GD])
    FT = SpectralFitterPgstat(list(P_), list(B_), list(R_), channel_masks=list(M_))

    for name, (mk, x0) in MODELS.items():
        def S(x):
            if name == "PL" and not -4 < x[1] < 0:
                return 1e12
            fun = mk(x)
            with np.errstate(all="ignore"):
                v = -2.0 * FT._fold_model(lambda p, e: fun(e), [0.0])
            return v if np.isfinite(v) else 1e12
        best = None
        for j in range(4):
            st = np.array(x0, float) + (j - 1.5) * 1.0 * (np.arange(len(x0)) == 0)
            r = minimize(S, st, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=20000))
            r = minimize(S, r.x, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=20000))
            if best is None or r.fun < best.fun:
                best = r
        x = best.x
        preds = {name: he_rate(mk(x))}
        if name == "L_BBCOMP":
            preds["L_COMP(part)"] = he_rate(lambda e: CO.eval([np.exp(x[1]), 5722.0, -0.4, 100.0], e))
        # profile of the predicted HE rate: scan a scale on the predicting part
        for key, p in preds.items():
            def pred_of(xx):
                if key == "L_COMP(part)":
                    return he_rate(lambda e: CO.eval([np.exp(xx[1]), 5722.0, -0.4, 100.0], e))
                return he_rate(mk(xx))
            # constrained scan: fix the predicted rate via a penalty and refit
            lo = hi = np.nan
            grid = p * np.array([0.0, 0.25, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]) if p > 0 else np.array([0.0])
            prof = []
            for g in grid:
                def Sc(xx, g=g):
                    pr = pred_of(xx)
                    return S(xx) + ((pr - g) / max(0.002 * p, 1e-3)) ** 2
                rr = minimize(Sc, x, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=20000))
                prof.append(S(rr.x))
            prof = np.array(prof) - best.fun
            ok = np.where(prof < 1.0)[0]
            if len(ok) and p > 0:
                i0, i1 = ok[0], ok[-1]
                lo = np.interp(1.0, [prof[i0], prof[i0 - 1]], [grid[i0], grid[i0 - 1]]) if i0 > 0 else 0.0
                hi = np.interp(1.0, [prof[i1], prof[i1 + 1]], [grid[i1], grid[i1 + 1]]) if i1 < len(grid) - 1 else np.nan
            for Cn in (0.81, 1.00, 1.12):
                pr = Cn * p
                sig_he = np.hypot(HE_STAT, HE_SYS)
                sig_pred = Cn * 0.5 * (hi - lo) if np.isfinite(hi) and np.isfinite(lo) else 0.0
                rows.append(dict(bkg=bkg, model=name, component=key, pg_gbm=best.fun, params=" ".join("%.4g" % v for v in x),
                                 he_pred_rate_C1=p, he_pred_lo1=lo, he_pred_hi1=hi, C=Cn, he_pred=pr,
                                 he_meas=HE_RATE, sigma_he=sig_he, sigma_pred=sig_pred,
                                 excl_sigma=(pr - HE_RATE) / sig_he,
                                 excl_sigma_incl_gbm=(pr - HE_RATE) / np.hypot(sig_he, sig_pred),
                                 eflux_10_1000=None))
            print(bkg, name, key, "PG %.2f" % best.fun, "x", np.round(x, 4), "HE pred (C=1) %.1f [%.1f, %.1f] evt/s" % (p, lo, hi),
                  flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.outdir("quiet"), "quietpred.csv"), index=False, float_format="%.6g")
print("done")
