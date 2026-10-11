"""HE/CsI (low gain) + Fermi-GBM (n4+n8 NaI, b1 BGO) joint time-resolved
spectroscopy of the GRB 221009A precursor.

GBM: TTE, polynomial background (as p221009_gbm_precursor.py), PG-stat.
HE:  CsI events (pw 76-255), unwrapped channels 20-255 summed over 18
     detectors in 27 groups, local linear background on [w0-100, w0-5] s, raw
     channel DRM for the precursor direction (resp group, p221009_resp_spec /
     resp/out/drm_221009a.npz), chi^2 with variance mu + B + var(B).
Joint statistic = PG-stat(GBM) + chi^2(HE); same photon model (COMP or Band,
pivot 100 keV) with a free HE/GBM normalisation C.  HE windows are the GBM
windows shifted by -0.06 s (HXMT T0 = GBM T0 + 0.06 s).
For each interval: best fits GBM-only and joint, and Ep likelihood profiles
for both, giving the 1-sigma / 2-sigma Ep ranges and the gain from adding HE.
    python p221009_gbm_joint.py [all | i,j,... | merge] [VARIANT] [DT_HE]
(indices into INTERVALS -> joint_fits_i-j.csv; 'merge' concatenates and plots).
Outputs $GBM_OUT/joint/joint_fits.csv, joint_epprofile.csv, joint_ep.png.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import Comptonized, Band
import p221009_gbm_common as C
from p221009_gbm_specfit import DetData

sys.path.insert(0, "/scratchfs2/gecam/guohx/p221009/blink/scripts")
import p221009_resp_spec as H     # noqa: E402

od = C.outdir("joint")
# HE time = GBM time - DT_HE.  The HE event cache is referenced to 13:17:00.000
# UTC, so the correct value is 0.0108 s (GBM trigger 13:16:59.9875 plus the
# 0.0017 s Fermi-HXMT light-travel difference; figs_v4.md section 6); runs
# joint2 used 0.06.  Third argument overrides (default 0.0108).
DT_HE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0108
BKG = ([(-250.0, -10.0), (95.0, 118.0)], 2)
VAR = sys.argv[2] if len(sys.argv) > 2 else "nominal"
# variants can be combined with "~", e.g. C1.25~g0.95 (shared-C systematics)
PARTS = VAR.split("~")


def part(prefix, exclude=None):
    """The variant part starting with prefix (and not with exclude), or None."""
    for p in PARTS:
        if p.startswith(prefix) and not (exclude and p.startswith(exclude)):
            return p
    return None


GD = {d: DetData(d, *BKG) for d in (["n7", "n8", "b1"] if "n7" in PARTS else ["n4", "n8", "b1"])}
z = np.load("/scratchfs2/gecam/guohx/p221009/resp/out/drm_221009a.npz")
ELO, EHI = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
# systematics variant (resp conventions), 2nd argument: nominal | prior / prior11
# (Gaussian prior C ~ N(1.0, 0.2) / N(1.1, 0.2)) | g0.95 / g1.05 (channel gain) |
# b_lo / b_hi (EC offset DRM variants) | n7 (GBM n7 instead of n4); a variant
# other than nominal fits COMP only
_per = z["prec_b_lo" if "b_lo" in PARTS else "prec_b_hi" if "b_hi" in PARTS else "precursor_per"].astype(float)
if part("g"):
    _per = H.gain_shift(_per, float(part("g")[1:]))
RHE = H.group_drm(_per)
PRIOR = 1.0 if "prior" in PARTS else 1.1 if "prior11" in PARTS else None
MODELS = ["COMP", "Band"] if VAR == "nominal" else ["COMP"]
# referee-2 variants: C<val> (HE/GBM normalisation fixed, for the shared-C
# profile), tilt<val> (HE response tilted by (E/1 MeV)^delta, fixed),
# tiltp<sigma> (delta free with prior N(0, sigma)), tlin2 (HE background:
# quadratic in time on T0-400...-5 s), alphaprof (alpha profiles and the
# Axelsson-alpha fixed fits instead of the Ep profile)
CFIX = float(part("C")[1:]) if part("C") else None
DELTA_FIX = float(part("tilt", "tiltp")[4:]) if part("tilt", "tiltp") else 0.0
DELTA_PRIOR = float(part("tiltp")[5:]) if part("tiltp") else None
HE_BKG_MODE = ("tlin", 2) if "tlin2" in PARTS else ("tloc", 1)
# hech<lo>-<hi>: only HE channel groups within unwrapped ch [lo, hi) (e.g.
# hech20-53 ~0.2-0.6 MeV, hech53-256 ~0.6-3.1 MeV)
_hc = part("hech")
GMASK = (H.GROUP[:-1] >= int(_hc[4:].split("-")[0])) & (H.GROUP[1:] <= int(_hc[4:].split("-")[1])) \
    if _hc else np.ones(H.GROUP.size - 1, bool)
T_HE, CH_HE = H.load_events()
EMID = np.sqrt(ELO * EHI)
AXEL_ALPHA = {"axel1": -1.36, "axel2": -1.60, "axel3": -1.73, "axel4": -1.77, "axel5": -1.88}

INTERVALS = [("axel1", -0.69, 0.23), ("axel2", 0.23, 1.00), ("axel3", 1.00, 2.72), ("axel4", 2.72, 7.00),
             ("axel5", 7.00, 19.95),
             ("tr", -1.0, 0.0), ("tr", 0.0, 1.0), ("tr", 1.0, 2.0), ("tr", 2.0, 3.0), ("tr", 3.0, 4.5),
             ("tr", 4.5, 6.5), ("tr", 6.5, 8.576), ("tr", 8.576, 13.0), ("tr", 13.0, 20.0),
             ("Ia", -0.003, 8.576), ("HEwin", -0.84, 10.06)]
AXEL = {"axel1": (15000, 3000, 3000), "axel2": (7000, 2000, 2000), "axel3": (6000, 2000, 2000),
        "axel4": (1300, 1200, 600), "axel5": (330, 40, 20)}
EP_GRID = np.geomspace(60.0, 60000.0, 36)


class Joint:
    def __init__(self, t1, t2, model):
        self.model = model
        P, B, R, M = zip(*[GD[d].pieces(t1, t2) for d in GD])
        self.ft = SpectralFitterPgstat(list(P), list(B), list(R), channel_masks=list(M))
        self.f = Comptonized() if model == "COMP" else Band()
        self.ft._function = self.f        # for ft.dof
        w = [(t1 - DT_HE, t2 - DT_HE)]
        self.src, self.bkg, self.bvar, self.expo = H.counts(T_HE, CH_HE, w, bkg_order=HE_BKG_MODE[1],
                                                            mode=HE_BKG_MODE[0])
        self.src, self.bkg, self.bvar = self.src[GMASK], self.bkg[GMASK], self.bvar[GMASK]
        self.fl = H.flive_mean([(np.floor(w[0][0]), np.ceil(w[0][1]))])
        self.he_net = float(self.src.sum() - self.bkg.sum())

    def gbm_full(self, q):
        # q = [lnA, alpha, lnEp, (beta)] -> gdt order
        A, al, ep = np.exp(q[0]), q[1], np.exp(q[2])
        return [A, ep, al, 100.0] if self.model == "COMP" else [A, ep, al, q[3], 100.0]

    def he_model(self, e, q):
        return H.cpl(e, q[0], q[1], q[2]) if self.model == "COMP" else H.band(e, q[0], q[1], q[2], q[3])

    def stat(self, q, lnC, use_he=True, delta=DELTA_FIX):
        al = q[1]
        if not -1.99 < al < 2.0 or not np.log(5.0) < q[2] < np.log(2e5):
            return 1e12
        if self.model == "Band" and not -10 < q[3] < -2.0001:
            return 1e12
        with np.errstate(all="ignore"):
            s = -2.0 * self.ft._fold_model(self.f.eval, self.gbm_full(q))
            if use_he:
                mu = np.exp(lnC) * self._hefold(q, delta)
                var = np.maximum(mu + self.bkg, 1) + self.bvar
                s += np.sum((self.src - self.bkg - mu) ** 2 / var)
                if PRIOR:
                    s += ((np.exp(lnC) - PRIOR) / 0.2) ** 2
        return s if np.isfinite(s) else 1e12

    def _hefold(self, q, delta=0.0):
        u = np.linspace(0, 1, 9)
        e = np.exp(np.log(ELO)[:, None] + (np.log(EHI) - np.log(ELO))[:, None] * u[None, :])
        f = self.he_model(e, q)
        pf = np.trapezoid(f, e, axis=1)
        if delta:
            pf = pf * (EMID / 1000.0) ** delta
        return (pf @ RHE * self.expo * self.fl)[GMASK]

    def fit(self, use_he=True, ep_fixed=None, start=None, alpha_fixed=None):
        """Minimise over [lnA, (alpha), (lnEp), (beta), (lnC), (delta)]."""
        nb = 1 if self.model == "Band" else 0
        fitC = use_he and CFIX is None
        fitD = use_he and DELTA_PRIOR is not None

        def unpack(x):
            x = list(x)
            q = [x.pop(0)]
            q.append(alpha_fixed if alpha_fixed is not None else x.pop(0))
            q.append(np.log(ep_fixed) if ep_fixed else x.pop(0))
            if nb:
                q.append(x.pop(0))
            lnC = (x.pop(0) if fitC else np.log(CFIX)) if use_he else 0.0
            d = x.pop(0) if fitD else DELTA_FIX
            return np.array(q), lnC, d

        def obj(x):
            q, lnC, d = unpack(x)
            if use_he and not np.log(0.3) < lnC < np.log(3.0):
                return 1e12
            s = self.stat(q, lnC, use_he, d)
            if fitD:
                s += (d / DELTA_PRIOR) ** 2
            return s

        starts = list(start) if start else []
        for lnA in (() if start else (np.log(0.02), np.log(0.005))):
            for al in ((None,) if alpha_fixed is not None else (-1.6, -1.0)):
                for ep in ((None,) if ep_fixed else (300.0, 1500.0, 8000.0)):
                    x = [lnA] + ([] if al is None else [al]) + ([] if ep_fixed else [np.log(ep)]) + \
                        ([-2.6] if nb else []) + ([0.0] if fitC else []) + ([0.0] if fitD else [])
                    starts.append(x)
        best = None
        for x0 in starts:
            r = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-5, fatol=1e-5, adaptive=True))
            r = minimize(obj, r.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6, adaptive=True))
            if best is None or r.fun < best.fun:
                best = r
        q, lnC, d = unpack(best.x)
        self.last_delta = d
        return best.fun, q, lnC, best.x


def interval_range(grid, prof, best, d):
    """Ep range where profile - best < d (log interpolation); NaN if open."""
    y = prof - best
    ok = y < d
    if not ok.any():
        return np.nan, np.nan
    i0, i1 = np.where(ok)[0][[0, -1]]
    lo = np.nan if i0 == 0 else np.exp(np.interp(d, [y[i0], y[i0 - 1]], np.log(grid[[i0, i0 - 1]])))
    hi = np.nan if i1 == len(grid) - 1 else np.exp(np.interp(d, [y[i1], y[i1 + 1]], np.log(grid[[i1, i1 + 1]])))
    return lo, hi


MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
SEL = range(len(INTERVALS)) if MODE in ("all", "merge") else [int(i) for i in MODE.split(",")]
SUF = ("" if MODE == "all" else "_" + MODE.replace(",", "-")) + ("" if VAR == "nominal" else "_" + VAR)
rows, prow = [], []
for i in (SEL if MODE != "merge" else []):
    tag, t1, t2 = INTERVALS[i]
    for model in MODELS:
        J = Joint(t1, t2, model)
        res = {}
        he_only = CFIX is not None or part("tilt") is not None or "tlin2" in PARTS
        for use_he in ((True,) if he_only else (False, True)):
            s0, q, lnC, x = J.fit(use_he)
            delta = J.last_delta
            extra = {}
            if "alphaprof" in PARTS:
                AG = np.arange(-2.0, -0.79, 0.05)
                xa = [x[0]] + list(x[2:])                 # drop alpha
                pa = np.array([J.fit(use_he, alpha_fixed=a, start=[xa])[0] for a in AG])
                ba = min(s0, pa.min())
                y = pa - ba
                ok = np.where(y < 1.0)[0]
                lo = np.interp(1.0, [y[ok[0]], y[ok[0] - 1]], AG[[ok[0], ok[0] - 1]]) if len(ok) and ok[0] > 0 else np.nan
                hi = np.interp(1.0, [y[ok[-1]], y[ok[-1] + 1]], AG[[ok[-1], ok[-1] + 1]]) \
                    if len(ok) and ok[-1] < len(AG) - 1 else np.nan
                extra.update(alpha_lo1=lo, alpha_hi1=hi)
                for a, s_ in zip(AG, pa):
                    prow.append(dict(tag=tag, t1=t1, t2=t2, model=model, he=use_he, par="alpha", value=a, stat=s_))
                if tag in AXEL_ALPHA:
                    sa, qa, _, _ = J.fit(use_he, alpha_fixed=AXEL_ALPHA[tag], start=[xa])
                    extra.update(alpha_axel=AXEL_ALPHA[tag], dstat_alpha_axel=sa - s0, Ep_at_alpha_axel=np.exp(qa[2]))
                prof = np.full(len(EP_GRID), np.nan)
                s0 = min(s0, pa.min())
            else:
                prof = np.full(len(EP_GRID), np.inf)
                xfree = list(x[:2]) + list(x[3:])          # drop lnEp
                j0 = int(np.argmin(abs(np.log(EP_GRID) - q[2])))
                for order in (range(j0, len(EP_GRID)), range(j0 - 1, -1, -1)):
                    warm = xfree
                    for j in order:
                        s, _, _, xx = J.fit(use_he, ep_fixed=EP_GRID[j], start=[warm, xfree])
                        prof[j] = min(s, 1e12); warm = list(xx)
                for ep, s in zip(EP_GRID, prof):
                    prow.append(dict(tag=tag, t1=t1, t2=t2, model=model, he=use_he, Ep=ep, stat=s))
            best = min(s0, np.nanmin(prof)) if np.isfinite(prof).any() else s0
            lo1, hi1 = interval_range(EP_GRID, prof, best, 1.0) if np.isfinite(prof).any() else (np.nan, np.nan)
            lo2, hi2 = interval_range(EP_GRID, prof, best, 4.0) if np.isfinite(prof).any() else (np.nan, np.nan)
            mu_he = np.exp(lnC) * J._hefold(q, delta) if use_he else J._hefold(q)
            chi_he = float(np.sum((J.src - J.bkg - mu_he) ** 2 / (np.maximum(mu_he + J.bkg, 1) + J.bvar)))
            r = dict(tag=tag, t1=t1, t2=t2, model=model, data="GBM+HE" if use_he else "GBM", var=VAR, dt_he=DT_HE,
                     stat=s0, pg_gbm=-2.0 * J.ft._fold_model(J.f.eval, J.gbm_full(q)),
                     chi2_he=chi_he, n_he_groups=len(J.src), dof_gbm=J.ft.dof,
                     A=np.exp(q[0]), alpha=q[1], Ep=np.exp(q[2]), beta=q[3] if model == "Band" else np.nan,
                     C_he=np.exp(lnC) if use_he else np.nan,
                     Ep_lo1=lo1, Ep_hi1=hi1, Ep_lo2=lo2, Ep_hi2=hi2, he_net=J.he_net,
                     eflux_10_1000=J.f.integrate(J.gbm_full(q), (10, 1000), energy=True),
                     eflux_200_3000=J.f.integrate(J.gbm_full(q), (200, 3000), energy=True),
                     delta=delta if use_he else np.nan, **extra)
            res[use_he] = r
            rows.append(r)
            print(tag, t1, t2, model, r["data"], "stat %.1f" % s0, "Ep %.0f [%.0f, %.0f] 2s [%.0f, %.0f]" % (
                r["Ep"], lo1, hi1, lo2, hi2), "alpha %.2f" % q[1], "C %.2f" % r["C_he"], "chi2_he %.1f" % chi_he,
                flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(od, "joint_fits%s.csv" % SUF), index=False, float_format="%.6g")
        pd.DataFrame(prow).to_csv(os.path.join(od, "joint_epprofile%s.csv" % SUF), index=False, float_format="%.6g")
if MODE not in ("all", "merge"):
    sys.exit(0)
if MODE == "merge":
    import glob
    for stem in ("joint_fits", "joint_epprofile"):
        fs = sorted(glob.glob(os.path.join(od, stem + "_*.csv")))
        pd.concat([pd.read_csv(f) for f in fs]).to_csv(os.path.join(od, stem + ".csv"), index=False, float_format="%.6g")
    A = pd.read_csv(os.path.join(od, "joint_fits.csv"))
    A["var"] = A["var"].fillna("nominal") if "var" in A else "nominal"
    A.to_csv(os.path.join(od, "joint_fits.csv"), index=False, float_format="%.6g")
    rows = A[A["var"] == "nominal"].to_dict("records")

# ---------------------------------------------------------------- figure
R = pd.DataFrame(rows)
fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
for a, sel, title in [(ax[0], R.tag == "tr", "1-2 s bins"), (ax[1], R.tag.str.startswith("axel"), "Axelsson+2025 bins")]:
    for (data, col, off) in [("GBM", "0.5", -0.1), ("GBM+HE", "k", 0.1)]:
        g = R[sel & (R.model == "COMP") & (R.data == data)]
        xm = 0.5 * (g.t1 + g.t2) + off; xe = 0.5 * (g.t2 - g.t1)
        lo = np.where(np.isfinite(g.Ep_lo1), g.Ep_lo1, 60.0); hi = np.where(np.isfinite(g.Ep_hi1), g.Ep_hi1, 6e4)
        a.errorbar(xm, g.Ep, xerr=xe, yerr=[np.clip(g.Ep - lo, 0, None), np.clip(hi - g.Ep, 0, None)],
                   fmt="o", ms=3, color=col, label="COMP " + data, capsize=2)
    if title.startswith("Axel"):
        for k, (ep, eh, el) in AXEL.items():
            t1, t2 = [x for x in INTERVALS if x[0] == k][0][1:]
            a.errorbar(0.5 * (t1 + t2) + 0.25, ep, xerr=0.5 * (t2 - t1), yerr=[[el], [eh]], fmt="s", ms=3,
                       color="C3", mfc="none", label="Axelsson+25 (GBM+LLE+LAT)" if k == "axel1" else None)
    a.axhspan(170, 3100, color="C0", alpha=0.08, label="HE/CsI LG band")
    a.set_yscale("log"); a.set_ylim(60, 6e4); a.set_ylabel("Ep (keV)"); a.set_title(title, fontsize=9)
    a.legend(fontsize=7)
ax[1].set_xlabel("t - T0(GBM) (s)")
fig.tight_layout(); fig.savefig(os.path.join(od, "joint_ep.png"), dpi=140)
print("done")
