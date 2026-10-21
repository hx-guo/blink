"""Forward-folding CsI spectral fits with the low-gain DRM (GRB 221009A).

Uses the 1K hour cache (observed events only, no gap-fillers), CsI events
(76 <= pw <= 255), unwrapped raw channels summed over the 18 detectors, a
per-channel-group polynomial background from pre-burst windows, and the
raw-channel DRM from p221009_resp_drm.py.  Only intervals without FIFO loss
are used (1K == 1B there; eband.md section 15); the source spectrum is divided
by the mean box live fraction.

    python p221009_resp_spec.py [OUTDIR]

For each interval: CPL and Band fits, an Ep profile (CPL, alpha free), and
the chi2 at Ep fixed to 240 keV and 4 MeV.  The precursor fit is repeated with
the DRM variants (FWHM from orbit, EC offset +-1 sigma, theta+5 deg) and with a
+-3% channel-gain shift to give the calibration systematics."""
import os
import sys

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz
from scipy.optimize import minimize

import matplotlib.pyplot as plt

BASE = "/scratchfs2/gecam/guohx/p221009"
CACHE = f"{BASE}/cache/he1k_hour.npz"
TS = f"{BASE}/resp/in/ts_1s.csv"
GROUP = np.array([20, 22, 24, 26, 28, 30, 33, 36, 40, 44, 48, 53, 58, 64, 70, 77, 85, 94,
                  104, 115, 127, 140, 155, 172, 190, 210, 232, 256])   # ch>=256 (wrapped, >3.1 MeV) dropped: upper-threshold roll-off not in the DRM
INTERVALS = {"precursor": [(-1, 9)],
             "rise": [(180, 216)],
             "mid": [(288, 330), (330, 440)],
             "flare_pre": [(440, 503)]}
BKG_WIN = [(-400, -5)]
EP_GRID = np.geomspace(50, 30000, 70)


def load_events():
    z = np.load(CACHE)
    t, ch, pw = z["t"], z["ch"].astype(int), z["pw"].astype(int)
    ch = np.where(ch < 20, ch + 256, ch)
    s = (pw >= 76) & (pw <= 255) & (ch >= GROUP[0]) & (ch < GROUP[-1])
    return t[s], ch[s]


def flive_mean(windows):
    import csv
    fl = {}
    with open(TS) as f:
        for r in csv.DictReader(f):
            fl.setdefault(float(r["t"]), []).append(float(r["flive"]))
    vals = [np.mean(v) for k, v in fl.items() if any(a <= k < b for a, b in windows)]
    return float(np.mean(vals)) if vals else 1.0


_PROXY = {}


def largeevt():
    """1-s Cnt_LargeEvt (18 det) keyed by T0-relative second (p221009_timing_bkg)."""
    if not _PROXY:
        cwd = os.getcwd()
        os.chdir(f"{BASE}/blink")
        try:
            from p221009_timing_bkg import eng_counter
            tl, lg = eng_counter("LargeEvt")
        finally:
            os.chdir(cwd)
        _PROXY.update(zip(tl.astype(int), lg))
    return _PROXY


PROXY_CAL = [(-1020, -10), (1900, 2580)]      # pre-burst + Earth-occulted (source down from T0+1884 s)


def counts_proxy(t, ch, windows):
    """As counts(), but each group's background is linear in Cnt_LargeEvt,
    calibrated on PROXY_CAL; in the source window the proxy is interpolated
    to the sub-second window edges by weighting whole seconds by overlap."""
    gi = np.digitize(ch, GROUP) - 1
    ng = GROUP.size - 1
    lm = largeevt()
    edges = np.arange(-1020, 2581, 1.0)
    x = np.array([lm.get(int(k), np.nan) for k in edges[:-1]])
    h = np.zeros((ng, edges.size - 1))
    sel = (t >= edges[0]) & (t < edges[-1])
    np.add.at(h, (gi[sel], np.floor(t[sel] - edges[0]).astype(int)), 1)
    cal = np.zeros(edges.size - 1, bool)
    for a, b in PROXY_CAL:
        cal |= (edges[:-1] >= a) & (edges[:-1] < b)
    cal &= np.isfinite(x) & (h.sum(0) > 0)
    src = np.zeros(ng)
    bkg = np.zeros(ng)
    bvar = np.zeros(ng)
    expo = 0.0
    wts = np.zeros(edges.size - 1)
    for a, b in windows:
        s = (t >= a) & (t < b)
        src += np.bincount(gi[s], minlength=ng)
        expo += b - a
        lo, hi = edges[:-1], edges[1:]
        wts += np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
    V = np.column_stack([x[cal], np.ones(cal.sum())])
    xw = np.array([np.nansum(wts * x), wts.sum()])
    for g in range(ng):
        y = h[g, cal]
        coef = np.linalg.lstsq(V, y, rcond=None)[0]
        cov = np.linalg.inv(V.T @ V) * max(np.var(y - V @ coef), 1.0)
        bkg[g] = xw @ coef
        bvar[g] = xw @ cov @ xw
    return src, bkg, bvar, expo


def counts(t, ch, windows, bkg_order=1, mode="tloc"):
    """Observed counts per channel group and background estimate (+var).
    mode 'tloc' (default): linear in time on [w0-100, w0-5] s, w0 = start of
    the first window or T0-1 s, whichever is earlier -- the only method unbiased in total and in spectral
    shape on pre-burst null windows (p221009_resp_bkgval.py); use only where
    that range is burst-free (precursor, quiet period);
    mode 'proxy': Cnt_LargeEvt-linear (counts_proxy), for the bright
    intervals where S/B >> 1; mode 'tlin': polynomial in time on BKG_WIN."""
    if mode == "proxy":
        return counts_proxy(t, ch, windows)
    bkg_win = BKG_WIN
    if mode == "tloc":
        w0 = min(np.floor(min(a for a, _ in windows)), -1.0)   # never later than the precursor onset
        bkg_win = [(w0 - 100, w0 - 5)]
    gi = np.digitize(ch, GROUP) - 1
    ng = GROUP.size - 1
    src = np.zeros(ng)
    expo = 0.0
    for a, b in windows:
        s = (t >= a) & (t < b)
        src += np.bincount(gi[s], minlength=ng)
        expo += b - a
    # background: polynomial in time per group from 1-s bins
    tb = np.concatenate([np.arange(a, b) for a, b in bkg_win]) + 0.5
    edges = np.arange(bkg_win[0][0], bkg_win[-1][1] + 1)
    h = np.zeros((ng, edges.size - 1))
    sel = (t >= edges[0]) & (t < edges[-1])
    np.add.at(h, (gi[sel], np.floor(t[sel] - edges[0]).astype(int)), 1)
    keep = np.isin(edges[:-1] + 0.5, tb)
    x = (edges[:-1] + 0.5)[keep]
    bkg = np.zeros(ng)
    bvar = np.zeros(ng)
    for g in range(ng):
        y = h[g, keep]
        V = np.vander(x, bkg_order + 1)
        cov = np.linalg.inv(V.T @ V / np.maximum(y.mean(), 1))
        coef = np.linalg.lstsq(V, y, rcond=None)[0]
        for a, b in windows:
            tt = np.arange(a, b, 0.01) + 0.005           # exact for non-integer windows
            Vt = np.vander(tt, bkg_order + 1) * 0.01
            bkg[g] += (Vt @ coef).sum()
            J = Vt.sum(0)
            bvar[g] += J @ cov @ J
    return src, bkg, bvar, expo


def group_drm(per):
    m = per.sum(0) if per.ndim == 3 else per
    return np.array([m[:, GROUP[i]:GROUP[i + 1]].sum(1) for i in range(GROUP.size - 1)]).T


def gain_shift(per, f):
    """Response for channels stretched by factor f (E = k*ch*f + b)."""
    m = per.sum(0) if per.ndim == 3 else per
    c = np.arange(m.shape[1] + 1) - 0.5
    cum = np.concatenate([np.zeros((m.shape[0], 1)), np.cumsum(m, 1)], 1)
    out = np.array([np.diff(np.interp(c * f, c, row)) for row in cum])
    return out


def photon_flux(model, p, elo, ehi, nsub=8):
    u = np.linspace(0, 1, nsub + 1)
    e = np.exp(np.log(elo)[:, None] + (np.log(ehi) - np.log(elo))[:, None] * u[None, :])
    f = model(e, *p)
    return _trapz(f, e, axis=1)          # ph/cm2/s per incident bin


def cpl(e, lnA, alpha, lnEp):
    ep = np.exp(lnEp)
    return np.exp(lnA) * (e / 100.0) ** alpha * np.exp(-(2 + alpha) * e / ep)


def band(e, lnA, alpha, lnEp, beta):
    ep = np.exp(lnEp)
    e0 = ep / (2 + alpha)
    eb = (alpha - beta) * e0
    llo = alpha * np.log(e / 100.0) - e / e0
    lhi = (alpha - beta) * np.log(max(eb, 1e-3) / 100.0) + (beta - alpha) + beta * np.log(e / 100.0)
    return np.exp(lnA + np.where(e < eb, llo, lhi))


def pl(e, lnA, gamma):
    return np.exp(lnA) * (e / 100.0) ** gamma


def fit(model, p0, R, elo, ehi, src, bkg, bvar, expo, fl, fixed=None, bounds=None, sys_frac=0.0):
    fixed = fixed or {}
    free = [i for i in range(len(p0)) if i not in fixed]

    def full(q):
        p = np.array(p0, float)
        p[free] = q
        for i, v in fixed.items():
            p[i] = v
        return p

    def chi2(q):
        p = full(q)
        if bounds:
            for i, (lo, hi) in bounds.items():
                if not lo <= p[i] <= hi:
                    return 1e12
        mu = photon_flux(model, p, elo, ehi) @ R * expo * fl
        var = np.maximum(mu + bkg, 1) + bvar + (sys_frac * mu) ** 2
        return np.sum((src - bkg - mu) ** 2 / var)

    best = None
    for jit in (0.0, 0.5, -0.5):
        q0 = np.array([p0[i] for i in free], float) + jit * (np.arange(len(free)) == 0)
        r = minimize(chi2, q0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-5, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return full(best.x), best.fun


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    t, ch = load_events()
    names = {"precursor": "precursor", "rise": "rise", "mid": "mid", "flare_pre": "flare"}
    rows = []
    fig, axs = plt.subplots(2, len(INTERVALS), figsize=(4.5 * len(INTERVALS), 7))
    for j, (iname, wins) in enumerate(INTERVALS.items()):
        R = group_drm(z[f"{names[iname]}_per"].astype(float))
        src, bkg, bvar, expo = counts(t, ch, wins)
        fl = flive_mean(wins)
        net = src - bkg
        sn = net.sum() / np.sqrt(src.sum() + bvar.sum())
        p_c, c2_c = fit(cpl, [np.log(max(net.sum(), 1) / expo / 3e3), -0.8, np.log(1000)], R, elo, ehi,
                        src, bkg, bvar, expo, fl, bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))})
        p_b, c2_b = fit(band, [p_c[0], p_c[1], p_c[2], -2.5], R, elo, ehi, src, bkg, bvar, expo, fl,
                        bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5)), 3: (-5, -1.6)})
        prof = []
        for ep in EP_GRID:
            pp, c2 = fit(cpl, [p_c[0], p_c[1], np.log(ep)], R, elo, ehi, src, bkg, bvar, expo, fl,
                         fixed={2: np.log(ep)}, bounds={1: (-2.5, 1.5)})
            prof.append((ep, c2, pp[1]))
        prof = np.array(prof)
        c240 = np.interp(np.log(240), np.log(prof[:, 0]), prof[:, 1])
        c4m = np.interp(np.log(4000), np.log(prof[:, 0]), prof[:, 1])
        ok = prof[:, 1] <= prof[:, 1].min() + 2.71
        ep_lo, ep_hi = prof[ok, 0].min(), prof[ok, 0].max()
        dof = GROUP.size - 1 - 3
        row = dict(interval=iname, windows=wins, expo=expo, flive=round(fl, 3), net=round(net.sum()),
                   snr=round(sn, 1), cpl_alpha=round(p_c[1], 3), cpl_ep=round(np.exp(p_c[2]), 1),
                   cpl_chi2=round(c2_c, 1), dof=dof, ep90=(round(ep_lo), round(ep_hi)),
                   band_alpha=round(p_b[1], 3), band_ep=round(np.exp(p_b[2]), 1), band_beta=round(p_b[3], 2),
                   band_chi2=round(c2_b, 1), dchi2_ep240=round(c240 - c2_c, 1), dchi2_ep4000=round(c4m - c2_c, 1))
        rows.append(row)
        print(row, flush=True)
        np.savetxt(f"{out}/spec_{iname}_counts.txt",
                   np.column_stack([GROUP[:-1], GROUP[1:], src, bkg, bvar,
                                    photon_flux(cpl, p_c, elo, ehi) @ R * expo * fl]),
                   fmt="%d %d %.1f %.2f %.2f %.2f", header="ch_lo ch_hi src bkg bkg_var cpl_model")
        np.savetxt(f"{out}/spec_{iname}_epprofile.txt", prof, fmt="%.2f %.3f %.3f", header="Ep chi2 alpha")
        # plot
        cm = 0.5 * (GROUP[:-1] + GROUP[1:])
        w = np.diff(GROUP)
        a = axs[0, j]
        a.errorbar(cm, net / w / expo, yerr=np.sqrt(src + bvar) / w / expo, fmt="o", ms=3, label="net")
        for lab, p in (("CPL best", p_c),):
            a.step(cm, photon_flux(cpl, p, elo, ehi) @ R * fl / w, where="mid", label=lab)
        for ep, ls in ((240, "--"), (4000, ":")):
            k = np.argmin(np.abs(prof[:, 0] - ep))
            pp, _ = fit(cpl, [p_c[0], prof[k, 2], np.log(ep)], R, elo, ehi, src, bkg, bvar, expo, fl,
                        fixed={2: np.log(ep)}, bounds={1: (-2.5, 1.5)})
            a.step(cm, photon_flux(cpl, pp, elo, ehi) @ R * fl / w, where="mid", ls=ls, label=f"Ep={ep} fixed")
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"{iname} {wins}")
        a.set_xlabel("raw channel (unwrapped)"); a.legend(fontsize=7)
        b = axs[1, j]
        b.plot(prof[:, 0], prof[:, 1] - prof[:, 1].min()); b.set_xscale("log")
        b.axhline(2.71, c="k", lw=0.5); b.axvline(240, ls="--", c="gray"); b.axvline(4000, ls=":", c="gray")
        b.set_ylim(0, 50); b.set_xlabel("Ep (keV), CPL alpha free"); b.set_ylabel("dchi2")
    fig.tight_layout(); fig.savefig(f"{out}/spec_fits.png", dpi=110)
    # precursor systematics
    wins = INTERVALS["precursor"]
    src, bkg, bvar, expo = counts(t, ch, wins)
    fl = flive_mean(wins)
    variants = {"nominal": z["precursor_per"], "fwhm_orbit": z["prec_fwhm_orbit"], "b_lo": z["prec_b_lo"],
                "b_hi": z["prec_b_hi"], "theta+5": z["prec_theta5"],
                "gain+3%": gain_shift(z["precursor_per"].astype(float), 1.03),
                "gain-3%": gain_shift(z["precursor_per"].astype(float), 0.97)}
    src2, bkg2, bvar2, _ = counts(t, ch, wins, mode="tlin")
    for vname, per in variants.items():
        R = group_drm(np.asarray(per, float))
        for bname, (s_, b_, v_) in {"bkg1": (src, bkg, bvar), "bkg2": (src2, bkg2, bvar2)}.items():
            if bname == "bkg2" and vname != "nominal":
                continue
            p_c, c2 = fit(cpl, [-3, -0.8, np.log(1000)], R, elo, ehi, s_, b_, v_, expo, fl,
                          bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))})
            c = {}
            for ep in (240, 4000):
                _, c[ep] = fit(cpl, [p_c[0], p_c[1], np.log(ep)], R, elo, ehi, s_, b_, v_, expo, fl,
                               fixed={2: np.log(ep)}, bounds={1: (-2.5, 1.5)})
            print(f"precursor syst {vname:10s} {bname}: alpha={p_c[1]:.2f} Ep={np.exp(p_c[2]):.0f} "
                  f"chi2={c2:.1f} d240={c[240] - c2:.1f} d4000={c[4000] - c2:.1f}", flush=True)


if __name__ == "__main__":
    main()
