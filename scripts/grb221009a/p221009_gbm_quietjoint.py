"""Quiescent-interval emission: GBM (n4+n8 NaI 20-500 keV, b1 0.4-38 MeV) +
HE/CsI (0.17-3.1 MeV, 27 channel groups) joint fit, to quantify the spectral
cutoff that the HE non-detection requires.

    python p221009_gbm_quietjoint.py GBMBKG HEBKG WINDOW
GBMBKG: orb30 | orb15 (scaled orbital CSPEC, p221009_gbm_quietspec.OrbData) |
        reg (per energy group regression of the burst-day pre-burst TTE rate
        on the +/-15 and +/-30 orbit rates: B = a*R15 + b*R30) |
        poly2 (TTE polynomial, -400...-10 and 95...118 s).
HEBKG:  proxy (resp default: Cnt_LargeEvt-linear, calibrated pre-burst +
        Earth occultation) | proxypre (calibrated pre-burst only) |
        tloc (local linear from T0-101...-6 s; not validated this far out).
WINDOW: II (121.219-164.228 s) | wide (115-172 s), GBM time; HE = GBM - C.HXMT_DT.
Statistic: PG(GBM) + chi2(HE) + ((C - 1.1)/0.2)^2, C = HE/GBM normalisation.
Models PL, COMP (CPL), Band.  Ep profile (COMP) and beta profile (Band), GBM
alone and joint -> one-sided 95% (delta 2.71) and 99.7% (delta 9) limits;
delta stat PL - COMP.  Output $GBM_OUT/quiet/quietjoint_<args>.csv (+ _prof).
"""
import functools
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import PowerLaw, Comptonized, Band
from gdt.core.background.primitives import BackgroundSpectrum
import p221009_gbm_common as C
import p221009_gbm_specfit as S
import p221009_gbm_quietspec as Q

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
import p221009_resp_spec as H     # noqa: E402

GBMBKG, HEBKG, WIN = sys.argv[1:4]
HEMODE = sys.argv[4] if len(sys.argv) > 4 else "spec"   # spec: 27 groups | total: one ch 20-255 bin + sys
T1, T2 = {"II": (121.219, 164.228), "wide": (115.0, 172.0)}[WIN]
od = C.outdir("quiet")
TAG = "%s_%s_%s" % (GBMBKG, HEBKG, WIN) + ("" if HEMODE == "spec" else "_" + HEMODE)
ER = {"n4": (20.0, 500.0), "n8": (20.0, 500.0), "b1": (400.0, 38000.0)}
Q.cspec = functools.lru_cache(maxsize=None)(Q.cspec)
PRE_WIN = [(-400.0, -10.0), (95.0, 118.0)]


class RegData(S.DetData):
    """Background = a_g * R15 + b_g * R30 per energy group g, with R15/R30 the
    mean +/-15 / +/-30 orbit CSPEC rates; (a, b) fitted to the burst-day TTE
    rate on a 4.096 s grid over PRE_WIN."""

    def __init__(self, det):
        self.det = det
        self.tte = C.open_tte(det, (-400.0, 800.0))
        t = np.asarray(self.tte.data.times); ch = np.asarray(self.tte.data.channels)
        edges = np.concatenate([np.arange(a, b - 4.095, 4.096) for a, b in PRE_WIN])
        y, r15, r30, w = [], [], [], []
        for e0 in edges:
            e1 = e0 + 4.096
            m = (t >= e0) & (t < e1)
            y.append(np.bincount(ch[m], minlength=128)[:128] / 4.096)
            c15, x15 = Q.orb_counts(det, (-15, 15), e0, e1)
            c30, x30 = Q.orb_counts(det, (-30, 30), e0, e1)
            r15.append(c15 / x15); r30.append(c30 / x30)
        y, r15, r30 = map(np.array, (y, r15, r30))
        lo = np.asarray(self.tte.ebounds.low_edges())
        gedges = Q.GROUP_EDGES[det[0]]
        self.grp = np.clip(np.searchsorted(gedges, lo, side="right") - 1, 0, len(gedges) - 2)
        self.coef = {}
        for g in np.unique(self.grp):
            m = self.grp == g
            Y = y[:, m].sum(1); A = np.stack([r15[:, m].sum(1), r30[:, m].sum(1)], 1)
            wt = 4.096 / np.maximum(Y, 1e-3)
            cov = np.linalg.inv(A.T @ (A * wt[:, None]))
            p = cov @ (A.T @ (Y * wt))
            self.coef[g] = (p, cov)
        print(det, "reg coefs", {int(g): np.round(v[0], 3) for g, v in self.coef.items()}, flush=True)

    def pieces(self, t1, t2, erange=None):
        erange = erange or C.erange_for(self.det)
        pha = self.tte.to_pha(time_ranges=[(t1, t2)])
        lo, hi = C.chan_range(pha.ebounds, *erange)
        mask = np.zeros(pha.num_chans, bool); mask[lo:hi + 1] = True
        c15, x15 = Q.orb_counts(self.det, (-15, 15), t1, t2)
        c30, x30 = Q.orb_counts(self.det, (-30, 30), t1, t2)
        r15, r30 = c15 / x15, c30 / x30
        rate = np.zeros(128); var = np.zeros(128)
        for g, (p, cov) in self.coef.items():
            m = self.grp == g
            rate[m] = p[0] * r15[m] + p[1] * r30[m]
            # parameter error spread over the group in proportion to the rate,
            # plus orbital counting noise
            Rg = np.array([r15[m].sum(), r30[m].sum()])
            frac = rate[m] / max(rate[m].sum(), 1e-12)
            var[m] = (frac * np.sqrt(Rg @ cov @ Rg)) ** 2 + p[0] ** 2 * c15[m] / x15 ** 2 + p[1] ** 2 * c30[m] / x30 ** 2
        e = pha.ebounds
        bak = BackgroundSpectrum(rate, np.sqrt(var), np.asarray(e.low_edges()), np.asarray(e.high_edges()),
                                 np.full(128, pha.exposure))
        return pha, bak, C.response_for(self.det, t1, t2), mask


if GBMBKG in ("orb30", "orb15"):
    GD = [Q.OrbData(d, (-30, 30) if GBMBKG == "orb30" else (-15, 15)) for d in ER]
elif GBMBKG == "reg":
    GD = [RegData(d) for d in ER]
else:
    GD = [S.DetData(d, PRE_WIN, 2) for d in ER]
P_, B_, R_, M_ = zip(*[g.pieces(T1, T2, ER[g.det]) for g in GD])
FT = SpectralFitterPgstat(list(P_), list(B_), list(R_), channel_masks=list(M_))

# HE
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
RHE = H.group_drm(z["precursor_per"].astype(float))
T_HE, CH_HE = H.load_events()
w = [(T1 - C.HXMT_DT, T2 - C.HXMT_DT)]
if HEBKG == "proxypre":
    H.PROXY_CAL = [(-1020, -10)]
SRC, BKG, BVAR, EXPO = H.counts(T_HE, CH_HE, w, mode="tloc" if HEBKG == "tloc" else "proxy")
FL = H.flive_mean([(np.floor(w[0][0]), np.ceil(w[0][1]))])
if HEMODE == "total":
    # the proxy background is unbiased in total counts but not in spectral shape
    # (resp section 3.1): use the summed 0.17-3.1 MeV counts only, with the
    # quiet-period validation residual rms (166 counts per 10 s, resp section
    # 4.4) added as a fully time-correlated systematic
    SYS = 166.0 * EXPO / 10.0
    SRC, BKG, BVAR = np.array([SRC.sum()]), np.array([BKG.sum()]), np.array([BVAR.sum() + SYS ** 2])
    RHE = RHE.sum(1, keepdims=True)
print("HE net %.0f +- %.0f (bkg %.0f) in %.1f s, flive %.3f" % (SRC.sum() - BKG.sum(), np.sqrt(BKG.sum() + BVAR.sum()),
                                                            BKG.sum(), EXPO, FL), flush=True)
U = np.linspace(0, 1, 9)
EGRID = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * U[None, :])

FUN = {"PL": PowerLaw(), "COMP": Comptonized(), "Band": Band()}


def gdt_params(model, q):
    A = np.exp(q[0])
    if model == "PL":
        return [A, q[1], 100.0]
    if model == "COMP":
        return [A, np.exp(q[2]), q[1], 100.0]
    return [A, np.exp(q[2]), q[1], q[3], 100.0]


def he_counts(model, q):
    f = FUN[model].eval(gdt_params(model, q), EGRID.ravel()).reshape(EGRID.shape)
    return np.trapezoid(f, EGRID, axis=1) @ RHE * EXPO * FL


def stat(model, q, lnC, use_he):
    if model != "PL" and not (-1.99 < q[1] < 2.0 and np.log(2.0) < q[2] < np.log(1e5)):
        return 1e12
    if model == "PL" and not -4.0 < q[1] < 0.0:
        return 1e12
    if model == "Band" and not -10.0 < q[3] < -2.0001:
        return 1e12
    with np.errstate(all="ignore"):
        s = -2.0 * FT._fold_model(FUN[model].eval, gdt_params(model, q))
        if use_he:
            mu = np.exp(lnC) * he_counts(model, q)
            s += np.sum((SRC - BKG - mu) ** 2 / (np.maximum(mu + BKG, 1) + BVAR))
            s += ((np.exp(lnC) - 1.1) / 0.2) ** 2
    return s if np.isfinite(s) else 1e12


NP = {"PL": 2, "COMP": 3, "Band": 4}


def fit(model, use_he, fix=None):
    """fix = (index, value) of a fixed shape parameter (in q space)."""
    def unpack(x):
        q = list(x[:NP[model] - (1 if fix else 0)])
        if fix:
            q.insert(fix[0], fix[1])
        lnC = x[-1] if use_he else 0.0
        return np.array(q), lnC

    def obj(x):
        q, lnC = unpack(x)
        if use_he and not np.log(0.3) < lnC < np.log(3.0):
            return 1e12
        return stat(model, q, lnC, use_he)

    starts = []
    for lnA in (np.log(0.01), np.log(0.001)):
        for sh in {"PL": [[-2.0]], "COMP": [[-1.5, np.log(50.0)], [-1.0, np.log(300.0)], [-1.8, np.log(2000.0)]],
                   "Band": [[-1.0, np.log(40.0), -2.3], [-1.5, np.log(150.0), -3.0], [-0.5, np.log(30.0), -2.1]]}[model]:
            q = [lnA] + sh
            if fix:
                q.pop(fix[0])
            starts.append(q + ([np.log(1.1)] if use_he else []))
    best = None
    for x0 in starts:
        r = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6, adaptive=True))
        r = minimize(obj, r.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7, adaptive=True))
        if best is None or r.fun < best.fun:
            best = r
    q, lnC = unpack(best.x)
    return best.fun, q, lnC


def upper_limit(grid, prof, best, d):
    """Largest grid value with prof - best < d, interpolated (NaN if open)."""
    y = np.asarray(prof) - best
    ok = np.where(y < d)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[-1]
    if i == len(grid) - 1:
        return np.nan
    return float(np.interp(d, [y[i], y[i + 1]], [grid[i], grid[i + 1]]))


rows, prows = [], []
fits = {}
for use_he in (False, True):
    for model in ["PL", "COMP", "Band"]:
        s0, q, lnC = fit(model, use_he)
        fits[(model, use_he)] = s0
        mu = he_counts(model, q) * (np.exp(lnC) if use_he else 1.1)
        chi_he = float(np.sum((SRC - BKG - mu) ** 2 / (np.maximum(mu + BKG, 1) + BVAR)))
        p = gdt_params(model, q)
        rows.append(dict(cfg=TAG, gbm_bkg=GBMBKG, he_bkg=HEBKG, window=WIN, model=model,
                         data="GBM+HE" if use_he else "GBM", stat=s0,
                         pg_gbm=-2.0 * FT._fold_model(FUN[model].eval, p), chi2_he=chi_he,
                         index=q[1], Ep=np.exp(q[2]) if model != "PL" else np.nan,
                         beta=q[3] if model == "Band" else np.nan, C=np.exp(lnC) if use_he else np.nan,
                         he_mode=HEMODE, he_pred_counts=float(mu.sum()), he_net=float(SRC.sum() - BKG.sum()),
                         he_sigma=float(np.sqrt(BKG.sum() + BVAR.sum())) if HEMODE == "spec" else float(np.sqrt(BKG.sum() + BVAR[0])),
                         eflux_10_1000=FUN[model].integrate(p, (10, 1000), energy=True),
                         eflux_200_3000=FUN[model].integrate(p, (200, 3000), energy=True),
                         pflux_200_3000=FUN[model].integrate(p, (200, 3000))))
        print(TAG, model, rows[-1]["data"], "stat %.2f" % s0, "idx %.3f" % q[1], "Ep", rows[-1]["Ep"],
              "beta", rows[-1]["beta"], "C", rows[-1]["C"], "HE pred %.0f vs net %.0f +- %.0f" % (
                  mu.sum(), SRC.sum() - BKG.sum(), rows[-1]["he_sigma"]), flush=True)
    # profiles
    EPG = np.geomspace(8.0, 20000.0, 30)
    pr = []
    for ep in EPG:
        s, _, _ = fit("COMP", use_he, fix=(2, np.log(ep)))
        pr.append(s); prows.append(dict(cfg=TAG, data="GBM+HE" if use_he else "GBM", par="Ep_COMP", value=ep, stat=s))
    b0 = min(fits[("COMP", use_he)], min(pr))
    BG = np.linspace(-6.0, -2.02, 25)
    pb = []
    for b in BG:
        s, _, _ = fit("Band", use_he, fix=(3, b))
        pb.append(s); prows.append(dict(cfg=TAG, data="GBM+HE" if use_he else "GBM", par="beta_Band", value=b, stat=s))
    bb = min(fits[("Band", use_he)], min(pb))
    for r in rows[-3:]:
        r.update(Ep_ul95=np.exp(upper_limit(np.log(EPG), pr, b0, 2.71)),
                 Ep_ul997=np.exp(upper_limit(np.log(EPG), pr, b0, 9.0)),
                 beta_ul95=upper_limit(BG, pb, bb, 2.71), beta_ul997=upper_limit(BG, pb, bb, 9.0),
                 d_PL_COMP=fits[("PL", use_he)] - fits[("COMP", use_he)],
                 d_PL_Band=fits[("PL", use_he)] - fits[("Band", use_he)],
                 d_beta2=pb[-1] - bb)
    pd.DataFrame(rows).to_csv(os.path.join(od, "quietjoint_%s.csv" % TAG), index=False, float_format="%.6g")
    pd.DataFrame(prows).to_csv(os.path.join(od, "quietjoint_%s_prof.csv" % TAG), index=False, float_format="%.6g")
print("done")
