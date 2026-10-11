"""Referee round 2 R1: consistency of GBM and HE in the precursor sub-intervals
(parameter goodness of fit, Maltoni & Schwetz 2003).

For each 1-2 s interval: COMP fits to GBM alone (n4+n8 20-900 keV, b1
0.4-38 MeV), to HE alone (27 groups, normalisation free) and jointly with the
HE/GBM normalisation fixed to the shared value C = 1.25 (section 8.2).
chi2_PG = S_joint - S_GBM - S_HE with 3 degrees of freedom (A, alpha, Ep
shared); also the Ep profiles of the three fits.  Arguments: interval indices
(default all) -> pgof_<i>.csv.  Output
$GBM_OUT/pgof/pgof.csv and pgof_profiles.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import Comptonized
import p221009_gbm_common as C
from p221009_gbm_specfit import DetData

sys.path.insert(0, "/scratchfs2/gecam/guohx/p221009/blink/scripts")
import p221009_resp_spec as H     # noqa: E402

od = C.outdir("pgof")
CSH = 1.25
INT = [(-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5), (4.5, 6.5), (6.5, 8.576)]
ER = {"n4": (20.0, 900.0), "n8": (20.0, 900.0), "b1": (400.0, 38000.0)}
DD = {d: DetData(d, [(-250.0, -10.0), (95.0, 118.0)], 2) for d in ER}
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
RHE = H.group_drm(z["precursor_per"].astype(float))
EG = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * np.linspace(0, 1, 9)[None, :])
T_HE, CH_HE = H.load_events()
CO = Comptonized()
EPG = np.geomspace(60.0, 60000.0, 36)


def nm(f, x0):
    r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6, adaptive=True))
    return minimize(f, r.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7, adaptive=True))


SEL = [int(a) for a in sys.argv[1:]] or list(range(len(INT)))
SUF = "" if len(SEL) == len(INT) else "_" + "-".join(map(str, SEL))
rows, prow = [], []
for t1, t2 in [INT[i] for i in SEL]:
    P, B, R, M = zip(*[DD[d].pieces(t1, t2, ER[d]) for d in ER])
    ft = SpectralFitterPgstat(list(P), list(B), list(R), channel_masks=list(M))
    w = [(t1 - C.HXMT_DT, t2 - C.HXMT_DT)]
    src, bkg, bvar, expo = H.counts(T_HE, CH_HE, w)
    fl = H.flive_mean([(np.floor(w[0][0]), np.ceil(w[0][1]))])

    def SG(par):
        with np.errstate(all="ignore"):
            v = -2.0 * ft._fold_model(CO.eval, par)
        return v if np.isfinite(v) else 1e12

    def SH(par, scale=1.0):
        f = CO.eval(par, EG.ravel()).reshape(EG.shape)
        mu = scale * np.trapezoid(f, EG, axis=1) @ RHE * expo * fl
        v = float(np.sum((src - bkg - mu) ** 2 / (np.maximum(mu + bkg, 1) + bvar)))
        return v if np.isfinite(v) else 1e12

    def mk(which):
        def S(x, ep=None):
            al = x[1]; e = ep if ep is not None else np.exp(x[2])
            if not (-1.99 < al < 2 and 5 < e < 2e5):
                return 1e12
            par = [np.exp(x[0]), e, al, 100.0]
            if which == "G":
                return SG(par)
            if which == "H":
                return SH(par)
            return SG(par) + SH(par, CSH)
        return S

    res = {}
    for which in ("G", "H", "J"):
        S = mk(which)
        best = None
        for lnA in (np.log(0.02), np.log(0.005)):
            for ep in (300.0, 1500.0, 8000.0):
                for al in (-1.6, -1.0):
                    r = nm(S, [lnA, al, np.log(ep)])
                    best = r if best is None or r.fun < best.fun else best
        prof = []
        for ep in EPG:
            r = nm(lambda x: S(x, ep), best.x[:2])
            r2 = nm(lambda x: S(x, ep), [best.x[0], -1.3])
            prof.append(min(r.fun, r2.fun))
            prow.append(dict(t1=t1, t2=t2, fit=which, Ep=ep, stat=prof[-1]))
        prof = np.array(prof); smin = min(best.fun, prof.min())
        y = prof - smin; ok = np.where(y < 1)[0]
        lo = np.exp(np.interp(1, [y[ok[0]], y[ok[0] - 1]], np.log(EPG[[ok[0], ok[0] - 1]]))) if len(ok) and ok[0] > 0 else np.nan
        hi = np.exp(np.interp(1, [y[ok[-1]], y[ok[-1] + 1]], np.log(EPG[[ok[-1], ok[-1] + 1]]))) \
            if len(ok) and ok[-1] < len(EPG) - 1 else np.nan
        res[which] = dict(S=smin, Ep=np.exp(best.x[2]) if best.fun <= prof.min() else EPG[np.argmin(prof)],
                          alpha=best.x[1], lo=lo, hi=hi, prof=prof)
    pg = res["J"]["S"] - res["G"]["S"] - res["H"]["S"]
    p = chi2.sf(pg, 3)
    epJ = res["J"]["Ep"]; epG = res["G"]["Ep"]
    row = dict(t1=t1, t2=t2, C_shared=CSH, he_net=float(src.sum() - bkg.sum()))
    for k in ("G", "H", "J"):
        row.update({"%s_S" % k: res[k]["S"], "%s_Ep" % k: res[k]["Ep"], "%s_alpha" % k: res[k]["alpha"],
                    "%s_Ep_lo1" % k: res[k]["lo"], "%s_Ep_hi1" % k: res[k]["hi"]})
    row.update(chi2_PG=pg, p_PG=p, sigma_PG=norm.isf(p / 2) if p > 0 else np.inf,
               dS_GBM_at_jointEp=np.interp(np.log(epJ), np.log(EPG), res["G"]["prof"]) - res["G"]["S"],
               dS_HE_at_GBMEp=np.interp(np.log(epG), np.log(EPG), res["H"]["prof"]) - res["H"]["S"])
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(od, "pgof%s.csv" % SUF), index=False, float_format="%.6g")
    pd.DataFrame(prow).to_csv(os.path.join(od, "pgof_profiles%s.csv" % SUF), index=False, float_format="%.6g")
print("done")
