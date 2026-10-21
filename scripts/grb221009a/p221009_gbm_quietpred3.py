"""Referee round 2 R2/R3: quiescent-interval (Lesage region II,
121.219-164.228 s GBM time) HE predictions with
  R2  C = 0.81, 1.00, 1.12 and 1.25 (the shared precursor value);
  R3b the HE DRM response to incident photons above 3 MeV scaled by 0, 0.7,
      1.0, 1.3 (out-of-band partial deposits drive the hard-component
      prediction);
  R3a GBM BGO alone (b0 0.385-38 MeV, b1 0.4-38 MeV): the Lesage BB+COMP and
      COMP-only shapes with the normalisation from the NaI (n4+n8 20-500 keV)
      fit, compared with the BGO data (delta PG vs background only), and the
      BGO-only 2-sigma upper limit on the COMP amplitude.
HE measurement +23 +- 15 +- 24 evt/s (resp.md section 8), hard-source
suppression factors 1.00 / 0.93 / 0.87 applied to the prediction; the GBM-fit
uncertainty of a prediction is taken from the same relative 1-sigma width as
in p221009_gbm_quietpred.py.  Output $GBM_OUT/quiet/quietpred_v3.csv,
quiet_bgo_lesage.csv.
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
ER = {"n4": (20.0, 500.0), "n8": (20.0, 500.0), "b1": (400.0, 38000.0), "b0": (385.0, 38000.0)}
MEAS, SST, SSY = 23.0, 15.0, 24.0
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
R0 = H.group_drm(z["precursor_per"].astype(float)).sum(1)
FL = H.flive_mean([(np.floor(T1 - C.HXMT_DT), np.ceil(T2 - C.HXMT_DT))])
EG = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * np.linspace(0, 1, 9)[None, :])
BB, CO, PL = BlackBody(), Comptonized(), PowerLaw()
QP = pd.read_csv("/scratchfs2/gecam/guohx/p221009/gbm/runs/qpred1/quiet/quietpred.csv")


def he_rate(fun, s3):
    resp = R0 * np.where(ELO >= 3000.0, s3, 1.0)
    f = fun(EG.ravel()).reshape(EG.shape)
    return float(np.trapezoid(f, EG, axis=1) @ resp * FL)


def nm(S, x0):
    best = None
    for j in range(4):
        st = np.array(x0, float) + (j - 1.5) * (np.arange(len(x0)) == 0)
        r = minimize(S, st, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=20000))
        r = minimize(S, r.x, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-9, maxiter=20000))
        best = r if best is None or r.fun < best.fun else best
    return best


def bbcomp(x):
    return lambda e: BB.eval([np.exp(x[0]), 19.27], e) + CO.eval([np.exp(x[1]), 5722.0, -0.4, 100.0], e)


def componly(x):
    return lambda e: CO.eval([np.exp(x[0]), 5722.0, -0.4, 100.0], e)


def plaw(x):
    return lambda e: PL.eval([np.exp(x[0]), x[1], 100.0], e)


MODELS = {"L_BBCOMP": (bbcomp, [np.log(1e-4), np.log(1e-3)]), "L_COMPonly": (componly, [np.log(1e-3)]),
          "PL": (plaw, [np.log(1e-2), -2.0])}
rows, brows = [], []
for bkg, ks in [("orb30", (-30, 30)), ("orb15", (-15, 15))]:
    GD = {d: Q.OrbData(d, ks) for d in ER}
    FT = {}
    for key, dets in {"main": ["n4", "n8", "b1"], "nai": ["n4", "n8"], "b1": ["b1"], "b0": ["b0"]}.items():
        P_, B_, R_, M_ = zip(*[GD[d].pieces(T1, T2, ER[d]) for d in dets])
        FT[key] = SpectralFitterPgstat(list(P_), list(B_), list(R_), channel_masks=list(M_))

    def S(key, fun):
        with np.errstate(all="ignore"):
            v = -2.0 * FT[key]._fold_model(lambda p, e: fun(e), [0.0])
        return v if np.isfinite(v) else 1e12

    for name, (mk, x0) in MODELS.items():
        bounds = (lambda x: -4 < x[1] < 0) if name == "PL" else (lambda x: True)
        x = nm(lambda x: S("main", mk(x)) if bounds(x) else 1e12, x0).x
        q = QP[(QP.bkg == bkg) & (QP.model == name) & (QP.component == name) & (QP.C == 1.0)].iloc[0]
        relw = 0.5 * (q.he_pred_hi1 - q.he_pred_lo1) / q.he_pred_rate_C1 if np.isfinite(q.he_pred_hi1) else 0.0
        for s3 in (1.0, 0.7, 1.3, 0.0):
            p1 = he_rate(mk(x), s3)
            for Cn in (0.81, 1.00, 1.12, 1.25):
                for f in (1.0, 0.93, 0.87):
                    pred = Cn * p1 * f
                    sh = np.hypot(SST, SSY)
                    rows.append(dict(bkg=bkg, model=name, drm_gt3MeV_scale=s3, C=Cn, suppression=f,
                                     he_pred=pred, he_pred_C1_noSupp=p1, he_meas=MEAS,
                                     excl_sigma=(pred - MEAS) / sh,
                                     excl_sigma_incl_gbm=(pred - MEAS) / np.hypot(sh, relw * pred)))
        print(bkg, name, "x", np.round(x, 4), "HE pred (C=1, s3=1,0.7,1.3,0):",
              [round(he_rate(mk(x), s), 1) for s in (1.0, 0.7, 1.3, 0.0)], flush=True)
        # R3a: BGO alone, Lesage shapes normalised to NaI
        if name != "PL":
            xn = nm(lambda xx: S("nai", mk(xx)), x0).x
            for b in ("b1", "b0"):
                null = S(b, lambda e: 0.0 * e)
                s_nai = S(b, mk(xn)); s_main = S(b, mk(x))
                # BGO-only fit of the COMP amplitude (COMP-only shape) and its 2-sigma upper limit
                xb = nm(lambda xx: S(b, componly(xx)), [np.log(1e-4)])
                grid = xb.x[0] + np.linspace(-3, 4, 141)
                prof = np.array([S(b, componly([g])) for g in grid])
                ok = grid[prof - min(prof.min(), xb.fun) < 4.0]
                A_ul2 = np.exp(ok.max()) if len(ok) else np.nan
                A_comp_main = np.exp(x[1]) if name == "L_BBCOMP" else np.exp(x[0])
                A_comp_nai = np.exp(xn[1]) if name == "L_BBCOMP" else np.exp(xn[0])
                brows.append(dict(bkg=bkg, model=name, bgo=b, pg_null=null, pg_shape_norm_nai=s_nai,
                                  pg_shape_norm_main=s_main, d_nai_vs_null=s_nai - null, d_main_vs_null=s_main - null,
                                  A_comp_from_nai=A_comp_nai, A_comp_from_main=A_comp_main,
                                  A_comp_bgo_best=np.exp(xb.x[0]), A_comp_bgo_ul2=A_ul2,
                                  ratio_nai_over_ul2=A_comp_nai / A_ul2, ratio_main_over_ul2=A_comp_main / A_ul2,
                                  he_pred_C1_at_bgo_ul2=he_rate(componly([np.log(A_ul2)]), 1.0)))
                print(brows[-1], flush=True)
    C.write_csv(pd.DataFrame(rows), "quiet", "quietpred_v3.csv")
    C.write_csv(pd.DataFrame(brows), "quiet", "quiet_bgo_lesage.csv")
print("done")
