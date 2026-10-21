"""HE/CsI count-rate predictions in the quiescent interval II (GBM T0
121.219-164.228 s) for the models drawn in Fig Q(c), for the count-space
panel Fig Q(d) (referee 2 round 2, R3).

    run_gbm.sh -argu RUN p221009_v4_qfold      (gdt venv)

Same folding as p221009_gbm_quietpred3.py (he_rate: precursor-direction DRM,
CsI ch 20-255, times the mean live fraction): GBM n4+n8+b1 orb30 PL, COMP,
Band (gbm/runs/qspec1 quietspec_fits.csv, the curves of qcurves_II.csv) and
the Lesage II BB+COMP with the amplitudes of quietpred.csv (orb30, orb15).
DRM rows above 3 MeV incident scaled by s3 = 1, 0.7, 1.3, 0.  C = 1, no
suppression factor (applied in the figure).  The PL and Lesage rows of
quietpred_v3.csv are the ones quoted in gbm.md 9; this file adds COMP and
Band and cross-checks the Lesage numbers.
Output $GBM_OUT/v4/qfold_II.csv.
"""
import sys
import numpy as np
import pandas as pd
from gdt.core.spectra.functions import Band, BlackBody, Comptonized, PowerLaw
import p221009_gbm_common as C

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
import p221009_resp_spec as H     # noqa: E402

T1, T2 = 121.219, 164.228
QS = "/scratchfs2/gecam/guohx/p221009/gbm/runs/qspec1/quiet/quietspec_fits.csv"
QP = "/scratchfs2/gecam/guohx/p221009/gbm/runs/qpred1/quiet/quietpred.csv"
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
R0 = H.group_drm(z["precursor_per"].astype(float)).sum(1)
FL = H.flive_mean([(np.floor(T1 - C.HXMT_DT), np.ceil(T2 - C.HXMT_DT))])
EG = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * np.linspace(0, 1, 9)[None, :])


def he_rate(fun, s3):
    resp = R0 * np.where(ELO >= 3000.0, s3, 1.0)
    f = fun(EG.ravel()).reshape(EG.shape)
    return float(np.trapezoid(f, EG, axis=1) @ resp * FL)


S = pd.read_csv(QS)
S = S[(S.window == "II") & (S.bkg == "orb30") & (S.dets == "n4+n8+b1")].set_index("model")
M = {}
s = S.loc["PL"]
M["gbm_PL_orb30"] = lambda e, s=s: PowerLaw().eval([s.A, s["index"], 100.0], e)
s = S.loc["COMP"]
M["gbm_COMP_orb30"] = lambda e, s=s: Comptonized().eval([s.A, s.Epeak, s["index"], 100.0], e)
s = S.loc["Band"]
M["gbm_Band_orb30"] = lambda e, s=s: Band().eval([s.A, s.Epeak, s.alpha, s.beta, 100.0], e)
Q = pd.read_csv(QP)
ref = {}
for bkg in ("orb30", "orb15"):
    q = Q[(Q.bkg == bkg) & (Q.model == "L_BBCOMP") & (Q.component == "L_BBCOMP")].iloc[0]
    lb, lc = (float(v) for v in q.params.split())
    M[f"lesage_BBCOMP_{bkg}"] = lambda e, lb=lb, lc=lc: (BlackBody().eval([np.exp(lb), 19.27], e)
                                                         + Comptonized().eval([np.exp(lc), 5722.0, -0.4, 100.0], e))
    ref[f"lesage_BBCOMP_{bkg}"] = q.he_pred_rate_C1
rows = []
for name, fun in M.items():
    r = {"model": name, "flive": FL}
    for s3 in (1.0, 0.7, 1.3, 0.0):
        r[f"he_pred_C1_s3_{s3:g}"] = he_rate(fun, s3)
    r["quietpred_C1_ref"] = ref.get(name, np.nan)
    rows.append(r)
    print(r, flush=True)
C.write_csv(pd.DataFrame(rows), "v4", "qfold_II.csv")
print("done")
