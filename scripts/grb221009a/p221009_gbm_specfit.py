"""Forward-folding spectral fits of GBM TTE with gdt-core (PG-stat: Poisson
data, Gaussian polynomial background), shared by p221009_gbm_precursor.py and
p221009_gbm_quiet.py."""
import copy
import numpy as np
from scipy.optimize import minimize, brentq, OptimizeResult
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import Comptonized, Band, PowerLaw, BlackBody
import p221009_gbm_common as C


class DetData:
    """TTE + 1.024 s PHAII + polynomial background for one detector."""

    def __init__(self, det, bkg_ranges, order, trange=(-400.0, 800.0)):
        self.det = det
        self.tte = C.open_tte(det, trange)
        self.ph = C.phaii(self.tte, 1.024, trange)
        self.bf = C.poly_bkg(self.ph, bkg_ranges, order)
        self.bkg_ranges, self.order = bkg_ranges, order

    def pieces(self, t1, t2, erange=None):
        erange = erange or C.erange_for(self.det)
        # full spectrum + explicit channel mask (gdt energy_slice fails when a
        # channel has no events)
        pha = self.tte.to_pha(time_ranges=[(t1, t2)])
        lo, hi = C.chan_range(pha.ebounds, *erange)
        mask = np.zeros(pha.num_chans, bool); mask[lo:hi + 1] = True
        n = max(int(np.ceil((t2 - t1) / 0.064)), 1)
        e = np.linspace(t1, t2, n + 1)
        bak = self.bf.interpolate_bins(e[:-1], e[1:]).integrate_time(e[0], e[-1])
        rsp = C.response_for(self.det, t1, t2)
        return pha, bak, rsp, mask


def make_model(name):
    if name == "PL":
        f = PowerLaw(); f.default_values = [0.05, -1.6, 100.0]
    elif name == "COMP":
        f = Comptonized(); f.default_values = [0.05, 1000.0, -1.2, 100.0]
        f.min_values = [1e-10, 5.0, -1.99, 0.01]
    elif name == "Band":
        f = Band(); f.default_values = [0.05, 1000.0, -1.0, -2.5, 100.0]
        f.min_values = [1e-10, 5.0, -1.99, -10.0, 0.01]
    elif name == "BB+COMP":
        f = BlackBody() + Comptonized()
        f.default_values = [1e-3, 20.0, 0.01, 1000.0, -1.0, 100.0]
        lo = list(f.min_values); lo[4] = -1.99; f.min_values = lo
    else:
        raise ValueError(name)
    return f


STARTS = {"PL": [None], "COMP": [200.0, 1000.0, 4000.0, 15000.0],
          "Band": [200.0, 1000.0, 4000.0], "BB+COMP": [500.0, 3000.0, 8000.0]}


LOGP = ("A", "Epeak", "kT")


class _Obj:
    """-log L (= PG-stat / 2, gdt convention) of the free parameters, with log
    transform of positive scale parameters and box bounds by penalty."""

    def __init__(self, ft, f):
        self.ft, self.f = ft, f
        names = [n for n, _, _ in f.param_list]
        self.idx = [i for i in range(f.nparams) if f.free[i]]
        self.names = [names[i] for i in self.idx]
        self.lo = np.array([f.min_values[i] for i in self.idx], float)
        self.hi = np.array([f.max_values[i] for i in self.idx], float)
        self.log = np.array([n.split(": ")[-1] in LOGP for n in self.names])

    def to_p(self, u):
        return np.where(self.log, np.exp(np.clip(u, -700, 700)), u)

    def to_u(self, p):
        return np.where(self.log, np.log(np.maximum(p, 1e-300)), p)

    def __call__(self, u):
        p = self.to_p(u)
        if np.any(p < self.lo) or np.any(p > self.hi):
            return 1e30
        with np.errstate(all="ignore"):
            v = -self.ft._fold_model(self.f.fit_eval, p)
        return v if np.isfinite(v) else 1e30


def _nm(obj, u0):
    best = None
    for _ in range(3):           # restarts from the previous optimum
        r = minimize(obj, u0, method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-5, maxiter=40000, maxfev=40000, adaptive=True))
        if best is not None and best.fun - r.fun < 1e-4:
            best = r if r.fun < best.fun else best
            break
        best, u0 = r, r.x
    return best


def _profile(ft, f, k, pbest, stat0, factor=1.25, nmax=40):
    """1-sigma (delta PG-stat = 1, i.e. delta -logL = 0.5) profile-likelihood
    interval of free parameter k; other free parameters re-minimised."""
    def prof(val):
        g = copy.deepcopy(f)
        dv = list(g.default_values); dv[k] = val; g.default_values = dv
        fr = list(g.free); fr[k] = False; g.free = fr
        o = _Obj(ft, g)
        start = [pbest[j] for j, i in enumerate(_Obj(ft, f).idx) if i != k]
        r = _nm(o, o.to_u(np.array(start)))
        return r.fun - stat0 - 0.5
    j = _Obj(ft, f).idx.index(k)
    x0 = pbest[j]
    lo_b, hi_b = f.min_values[k], f.max_values[k]
    logp = x0 > 0 and f.param_list[k][0].split(": ")[-1] in LOGP
    out = []
    for sgn in (-1, 1):
        prev = x0
        x = x0
        lim = np.nan
        for _ in range(nmax):
            x = x * factor ** sgn if logp else x + sgn * max(abs(x0) * 0.03, 0.02) * (1.5 ** _)
            if x <= lo_b or x >= hi_b:
                break
            if prof(x) > 0:
                lim = brentq(prof, min(prev, x), max(prev, x), xtol=1e-4 * abs(x0) + 1e-6)
                break
            prev = x
        out.append(abs(lim - x0) if np.isfinite(lim) else np.nan)
    return out


def fit(datas, t1, t2, model, eranges=None, profile=("Epeak", "index", "alpha", "beta", "kT")):
    """Fit one model jointly to the detectors in datas over [t1, t2]: gdt TNC
    from several Epeak starts, polished by Nelder-Mead; keep the best.  Errors:
    profile likelihood (delta PG-stat = 1) for the shape parameters, Hessian for
    the rest.  'stat' is PG-stat (= 2 x gdt's -logL).  Returns a flat dict."""
    eranges = eranges or {}
    P, B, R, M = zip(*[d.pieces(t1, t2, eranges.get(d.det)) for d in datas])
    best = None
    for ep in STARTS[model]:
        f = make_model(model)
        if ep is not None:
            dv = list(f.default_values)
            k = [n for n, _, _ in f.param_list].index("Epeak") if model != "BB+COMP" else 3
            dv[k] = ep
            f.default_values = dv
        ft = SpectralFitterPgstat(list(P), list(B), list(R), channel_masks=list(M), method="TNC")
        try:
            ft.fit(f, options=dict(maxfun=4000))
            u0 = ft.parameters
        except Exception as exc:     # noqa: BLE001
            print("gdt fit failed", model, ep, exc, flush=True)
            u0 = np.array(f.default_values, float)[np.array(f.free)]
        obj = _Obj(ft, f)
        cands = [np.array(f.default_values, float)[np.array(f.free)]]
        if ft._res is not None and np.all(np.isfinite(ft.parameters)):
            cands.append(np.array(ft.parameters, float))
        for c in cands:
            r = _nm(obj, obj.to_u(np.clip(c, obj.lo + 1e-12, obj.hi - 1e-12)))
            if best is None or r.fun < best[2].fun:
                best = (ft, f, r)
    ft, f, r = best
    obj = _Obj(ft, f)
    par = obj.to_p(r.x)
    ft._res = OptimizeResult(x=par, fun=r.fun, success=bool(r.success), message=r.message)
    names = [n for n, _, _ in f.param_list]
    free = np.array(f.free)
    try:
        se = np.asarray(ft.symmetric_errors)
    except Exception:                # noqa: BLE001
        se = np.full(len(par), np.nan)
    full = np.array(f.default_values, float)
    full[free] = par
    out = dict(t1=t1, t2=t2, model=model, dets="+".join(d.det for d in datas),
               stat=2 * r.fun, dof=ft.dof, success=bool(r.success))
    j = 0
    for k, (n, fr) in enumerate(zip(names, free)):
        if not fr:
            continue
        out[n] = par[j]
        if n.split(": ")[-1] in profile:
            lo, hi = _profile(ft, f, k, par, r.fun)
        else:
            lo = hi = se[j]
        out[n + "_elo"] = lo; out[n + "_ehi"] = hi
        j += 1
    for lo, hi in [(10, 1000), (8, 40000), (200, 3000), (1000, 10000)]:
        out["eflux_%g_%g" % (lo, hi)] = f.integrate(full, (lo, hi), energy=True)  # erg cm-2 s-1
    out["pflux_200_3000"] = f.integrate(full, (200, 3000))
    # net counts per detector in the fitted channels
    for p, b, d, m in zip(P, B, datas, M):
        out["net_" + d.det] = float(p.data.counts[m].sum() - (b.rates[m] * p.exposure).sum())
        out["bkg_" + d.det] = float((b.rates[m] * p.exposure).sum())
    out["_fitter"] = ft
    return out
