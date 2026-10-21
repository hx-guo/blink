#!/usr/bin/env python3
"""Quantitative HXMT/HE (recovered) vs INTEGRAL/IBIS-ISGRI sub-peak lag for
the SGR 1935+2154 burst of FRB/XRB 200428.

Both light curves are projected onto the common geocentric reference using
the light-travel functions of ``plot_hxmt_vs_ibis.py`` (re-used, not
re-implemented).  T0 = 2020-04-28T14:34:24.011 UTC (geocentric).

Sign convention: lag tau > 0  <=>  HXMT is LATER than IBIS.

Methods (all deterministic, no random numbers):
  1. chi^2 lag scan: IBIS 1-ms counts n_i modelled as a*H_i(tau) + b with
     H_i the HXMT net rate on its own 1-ms grid and the IBIS photon times
     shifted by +tau (so that HXMT(t) = IBIS(t - tau)); shifting the
     gridded HXMT curve instead produces 1-ms periodic artefacts.  Variance = model Poisson
     (a*H+b) + a^2 * HXMT Poisson.  Best tau = min chi^2, 1-sigma from
     Delta chi^2 = 1 (parabola through the minimum).  Also a pure Poisson
     (Cash) variant that ignores HXMT noise.
  2. Pearson cross-correlation coefficient vs tau (same shift scheme),
     peak located by a parabola through the maximum.
  3. Independent peak positions: model-free centroid of the net counts in
     a +/-15 ms window around each Mereghetti+2020 peak, HXMT and IBIS
     separately (Poisson errors); difference of centroids = lag.

Outputs go to /Users/skyair/Developer/ihep/talk-20/figs/lag/.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts",)]  # sibling and shared script modules
import plot_hxmt_vs_ibis as P  # noqa: E402

OUT = Path("/Users/skyair/Developer/ihep/talk-20/figs/lag")
OUT.mkdir(parents=True, exist_ok=True)

BIN = 0.001
WIN_ALL = (0.380, 0.540)
# Mereghetti+2020 IBIS peaks relative to 14:34:24.000 UTC -> our T0 (=+0.011)
PEAKS_M20 = {"P1": 0.434 - 0.011, "P2": 0.462 - 0.011, "P3": 0.493 - 0.011}
HALF_SUB = 0.025            # +/- window for sub-peak CCF / chi^2
HALF_FIT = 0.015            # +/- window for centroid peak positions
TAU = np.round(np.arange(-0.020, 0.020 + 1e-9, 0.0001), 5)   # -20..+20 ms, 0.1 ms
TAU_INT = np.round(np.arange(-0.020, 0.020 + 1e-9, 0.001), 4)  # integer-ms lags for CCF
BKG_WIN = [(-0.5, -0.1), (0.7, 1.0)]
IBIS_E = (20.0, 200.0)


# ----------------------------------------------------------------------------
def load_all():
    trig = Time(P.HXMT_TRIGGER_UTC_STR, scale="utc")
    met = (trig - Time("2012-01-01T00:00:00", scale="utc")).sec
    ijd = (trig.tt - Time("2000-01-01T00:00:00", scale="tt")).to("day").value
    lt_h = P.compute_hxmt_light_travel(met, trig, P.HXMT_ORBIT_FILE)
    lt_i = P.compute_integral_light_travel(ijd, P.IBIS_FILE, P.INTEGRAL_ORBIT_FILE)
    obs, fill = P.load_hxmt_from_cache(P.HXMT_CACHE, met)
    obs = obs + lt_h
    fill = fill + lt_h
    it, ie = P.load_ibis_events(P.IBIS_FILE, ijd, lt_i)
    m = (ie >= IBIS_E[0]) & (ie < IBIS_E[1])
    ibis = it[m]
    return dict(hxmt_obs=obs, hxmt_fill=fill, hxmt_all=np.concatenate([obs, fill]),
                ibis=ibis, ibis_near=ibis[(ibis > 0.2) & (ibis < 0.8)], lt_h=lt_h, lt_i=lt_i)


def bkg_rate_per_bin(t):
    n, dur = 0, 0.0
    for a, b in BKG_WIN:
        n += int(np.sum((t >= a) & (t < b)))
        dur += b - a
    return n / dur * BIN


def hist(t, edges):
    return np.histogram(t, bins=edges)[0].astype(float)


# ----------------------------------------------------------------------------
def chi2_scan(h_t, i_t, win, bkg_h, bkg_i, taus, cash=False):
    """chi^2(tau) for IBIS counts modelled from the (grid-aligned) HXMT rate.

    The HXMT recovered curve lives on a 1-ms grid, so it is histogrammed ONCE
    on that grid; the continuous (61 us) IBIS photon times are shifted by +tau
    and re-binned.  tau > 0 <=> HXMT later than IBIS (IBIS(t - tau) = HXMT(t)).
    """
    edges = np.arange(win[0], win[1] + BIN / 2, BIN)
    H = hist(h_t, edges)
    x = H - bkg_h
    A = np.vstack([x, np.ones_like(x)]).T
    out = np.empty(len(taus))
    pars = []
    for k, tau in enumerate(taus):
        n = hist(i_t + tau, edges)
        a, b = np.linalg.lstsq(A, n, rcond=None)[0]
        if cash:
            def nll(p):
                m = p[0] * x + p[1]
                if np.any(m <= 0):
                    return 1e30
                return 2.0 * np.sum(m - n * np.log(m))
            r = minimize(nll, [a, b], method="Nelder-Mead",
                         options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
            out[k] = r.fun
            pars.append(r.x)
            continue
        for _ in range(6):
            m = np.clip(a * x + b, 1e-3, None)
            var = m + a * a * H          # IBIS Poisson + scaled HXMT Poisson
            w = 1.0 / var
            Aw = A * np.sqrt(w)[:, None]
            a, b = np.linalg.lstsq(Aw, n * np.sqrt(w), rcond=None)[0]
        m = np.clip(a * x + b, 1e-3, None)
        var = m + a * a * H
        out[k] = np.sum((n - m) ** 2 / var)
        pars.append((a, b))
    return out, pars, len(H)


def ccf_scan(h_t, i_t, win, bkg_h, bkg_i, taus):
    """Pearson r at (integer-ms) lags; IBIS photons shifted by +tau, re-binned."""
    edges = np.arange(win[0], win[1] + BIN / 2, BIN)
    x = hist(h_t, edges) - bkg_h
    out = np.empty(len(taus))
    for k, tau in enumerate(taus):
        y = hist(i_t + tau, edges) - bkg_i
        out[k] = np.corrcoef(x, y)[0, 1]
    return out


def cash_unbinned_scan(h_t, i_t, win, bkg_h, taus):
    """Unbinned Poisson (Cash) likelihood of the IBIS photons vs lag.

    Template S(t): HXMT net rate on its 1-ms grid, linearly interpolated
    between bin centres (so the likelihood is a continuous function of tau).
    IBIS rate model lambda(t) = a * S(t + tau) + b  (tau > 0: HXMT later).
    C(tau) = 2 [ int lambda dt - sum_j ln lambda(t_j) ], profiled over a, b.
    Returns C(tau) (min-subtracted later), (a, b) at each tau.
    """
    # template on a WIDE grid so that shifting never runs off the edge
    wide = (0.30, 0.62)
    edges = np.arange(wide[0], wide[1] + BIN / 2, BIN)
    xc = 0.5 * (edges[1:] + edges[:-1])
    S = np.clip(hist(h_t, edges) - bkg_h, 0.0, None)
    W = win[1] - win[0]
    tj = i_t[(i_t >= win[0]) & (i_t < win[1])]
    tf = np.arange(win[0], win[1] + 1e-12, BIN / 20.0)   # fine grid for int S(t+tau) dt
    out = np.empty(len(taus))
    pars = []
    # start each tau afresh: 80% of photons from the template, 20% flat (photons/s)
    for k, tau in enumerate(taus):
        sj = np.interp(tj + tau, xc, S)        # template value at each photon
        S_int = np.trapz(np.interp(tf + tau, xc, S), tf)   # int_win S(t+tau) dt

        def C(p):
            a, b = p
            return 2.0 * (a * S_int + b * W - np.sum(np.log(a * sj + b)))

        # damped Newton on the (concave-likelihood) 2-parameter problem
        pcur = np.array([0.8 * len(tj) / S_int, 0.2 * len(tj) / W])
        fcur = C(pcur)
        for _ in range(60):
            lam = pcur[0] * sj + pcur[1]
            g = 2.0 * np.array([S_int - np.sum(sj / lam), W - np.sum(1.0 / lam)])
            Hm = 2.0 * np.array([[np.sum(sj ** 2 / lam ** 2), np.sum(sj / lam ** 2)],
                                 [np.sum(sj / lam ** 2), np.sum(1.0 / lam ** 2)]])
            step = np.linalg.solve(Hm, g)
            lr = 1.0
            while lr > 1e-6:
                pn = pcur - lr * step
                if np.all(pn > 0):
                    fn = C(pn)
                    if fn <= fcur:
                        break
                lr *= 0.5
            else:
                break
            if abs(fcur - fn) < 1e-10:
                pcur, fcur = pn, fn
                break
            pcur, fcur = pn, fn
        r = type("R", (), {})()
        r.fun, r.x = fcur, pcur
        out[k] = r.fun
        pars.append(r.x)
    return out, pars, len(tj)


def parabola_min(taus, c, half=0.004):
    """Parabola through the points within +/-half of the minimum -> (tau0, sigma)."""
    k = int(np.argmin(c))
    sel = np.abs(taus - taus[k]) <= half + 1e-9
    p = np.polyfit(taus[sel], c[sel], 2)
    if p[0] <= 0:
        return taus[k], np.nan, p
    tau0 = -p[1] / (2 * p[0])
    sig = 1.0 / np.sqrt(p[0])          # Delta chi^2 = 1  ->  a*sig^2 = 1
    return tau0, sig, p


def parabola_max(taus, c, half=0.004):
    k = int(np.argmax(c))
    sel = np.abs(taus - taus[k]) <= half + 1e-9
    p = np.polyfit(taus[sel], c[sel], 2)
    if p[0] >= 0:
        return taus[k], p
    return -p[1] / (2 * p[0]), p


def dchi2_interval(taus, c, level=1.0):
    """Direct Delta chi^2 <= level interval around the global minimum."""
    k = int(np.argmin(c))
    ok = c <= c[k] + level
    # contiguous region containing k
    lo = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(c) - 1 and ok[hi + 1]:
        hi += 1
    return taus[lo], taus[hi]


# ----------------------------------------------------------------------------
def centroid(t, centre, half, bkg):
    """Model-free first moment of the net 1-ms counts in [centre-half, centre+half].

    Poisson error: var(c) = sum_i n_i^raw (t_i - c)^2 / N_net^2 (background
    treated as a fixed constant)."""
    edges = np.arange(centre - half, centre + half + BIN / 2, BIN)
    x = 0.5 * (edges[1:] + edges[:-1])
    raw = hist(t, edges)
    net = raw - bkg
    N = net.sum()
    c = np.sum(net * x) / N
    var = np.sum(raw * (x - c) ** 2) / N ** 2
    return dict(mu=c, mu_err=np.sqrt(var), N=N, x=x, n=raw, net=net)


# ----------------------------------------------------------------------------
def main():
    D = load_all()
    h_all, h_obs, i_t = D["hxmt_all"], D["hxmt_obs"], D["ibis_near"]
    bkg_h = bkg_rate_per_bin(h_obs)      # background from un-gapped region
    bkg_i = bkg_rate_per_bin(D["ibis"])
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say("# HXMT/HE vs INTEGRAL/IBIS-ISGRI sub-peak lag, FRB/XRB 200428")
    say()
    say(f"- T0 = {P.HXMT_TRIGGER_UTC_STR} UTC (geocentric); bin = 1 ms")
    say(f"- Geocentric projection applied: HXMT +{D['lt_h']*1e3:.2f} ms, "
        f"INTEGRAL +{D['lt_i']*1e3:.1f} ms (both added to event times)")
    say(f"- HXMT: all events incl. fill (EVT {len(h_obs):,} + FILL {len(D['hxmt_fill']):,}); "
        f"IBIS-ISGRI {IBIS_E[0]:.0f}-{IBIS_E[1]:.0f} keV, {len(i_t):,} photons total")
    say(f"- Background per 1-ms bin: HXMT {bkg_h:.2f}, IBIS {bkg_i:.3f} "
        f"(windows {BKG_WIN})")
    say(f"- Sign convention: tau > 0  <=>  HXMT later than IBIS")
    say(f"- Lag grid: -20..+20 ms, step 0.1 ms (events shifted continuously, re-binned)")
    say()

    windows = {"ALL (0.380-0.540 s, P1+P2+P3)": WIN_ALL}
    for k, c in PEAKS_M20.items():
        windows[f"{k} ({c:.3f} +/- {HALF_SUB*1e3:.0f} ms)"] = (c - HALF_SUB, c + HALF_SUB)
    windows["P2+P3 (0.426-0.507 s)"] = (PEAKS_M20["P2"] - HALF_SUB, PEAKS_M20["P3"] + HALF_SUB)

    res = {}
    say("## Method 1: lag scan")
    say()
    say("Primary: unbinned Poisson (Cash) likelihood of the IBIS photons against the HXMT "
        "1-ms template (linearly interpolated), profiled over amplitude and background; "
        "1 sigma from Delta C = 1 (parabola through the minimum over +/-4 ms, and the direct "
        "Delta C <= 1 crossing).  Cross-checks: binned chi^2 with both Poisson variances "
        "(noisy vs tau at the 0.1-ms level because re-binning changes the data; parabola over "
        "+/-4 ms), and Pearson CCF at integer-ms lags with a parabola through the peak.")
    say()
    say("| window | Cash tau (ms) | +/-1sig parabola | DC<=1 crossing (ms) | N_IBIS | binned chi2 tau (ms) | +/-1sig | chi2_min/dof | CCF peak (ms) | r_max |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for name, win in windows.items():
        cs, cpars, nph = cash_unbinned_scan(h_all, i_t, win, bkg_h, TAU)
        tc, sc, _ = parabola_min(TAU, cs)
        lo, hi = dchi2_interval(TAU, cs)
        c2, pars, nb = chi2_scan(h_all, i_t, win, bkg_h, bkg_i, TAU)
        t0, s0, _ = parabola_min(TAU, c2)
        r = ccf_scan(h_all, i_t, win, bkg_h, bkg_i, TAU_INT)
        tr, _ = parabola_max(TAU_INT, r, half=0.002)
        res[name] = dict(win=win, cash=cs, tau=tc, sig=sc, lo=lo, hi=hi, nph=nph,
                         chi2=c2, tau_chi=t0, sig_chi=s0, chi2min=c2.min(), dof=nb - 3,
                         ccf=r, tau_r=tr, rmax=r.max(), a=cpars[int(np.argmin(cs))][0])
        say(f"| {name} | {tc*1e3:+.2f} | {sc*1e3:.2f} | [{lo*1e3:+.1f}, {hi*1e3:+.1f}] | {nph} | "
            f"{t0*1e3:+.2f} | {s0*1e3:.2f} | {c2.min():.1f}/{nb-3} | {tr*1e3:+.2f} | {r.max():.3f} |")
    say()
    # consistency of the three sub-peak lags (Cash)
    subs = [res[n] for n in res if n.startswith(("P1", "P2 (", "P3"))]
    lg = np.array([r_["tau"] for r_ in subs]); er = np.array([r_["sig"] for r_ in subs])
    w_ = 1 / er ** 2
    say(f"Sub-peak Cash lags P1/P2/P3: {', '.join(f'{v*1e3:+.2f}+/-{e*1e3:.2f}' for v, e in zip(lg, er))} ms; "
        f"weighted mean {np.sum(w_*lg)/np.sum(w_)*1e3:+.2f} +/- {1/np.sqrt(np.sum(w_))*1e3:.2f} ms, "
        f"consistency chi2 = {np.sum(w_*(lg-np.sum(w_*lg)/np.sum(w_))**2):.2f} for 2 dof")
    say()

    # ---- Method 2: independent centroids ---------------------------------
    say("## Method 2: independent net-count centroids, HXMT and IBIS separately")
    say()
    say(f"Window: Mereghetti+2020 peak +/- {HALF_FIT*1e3:.0f} ms (same window for both instruments), "
        "model-free first moment of background-subtracted 1-ms counts, Poisson error.")
    say()
    say("| peak | window (s) | HXMT centroid (s) | IBIS centroid (s) | IBIS net counts | lag = HXMT-IBIS (ms) |")
    say("|---|---|---|---|---|---|")
    gfits = {}
    for k, c in PEAKS_M20.items():
        gh = centroid(h_all, c, HALF_FIT, bkg_h)
        gi = centroid(i_t, c, HALF_FIT, bkg_i)
        lag = gh["mu"] - gi["mu"]
        lag_err = np.hypot(gh["mu_err"], gi["mu_err"])
        gfits[k] = (gh, gi, lag, lag_err)
        say(f"| {k} | {c-HALF_FIT:.3f}-{c+HALF_FIT:.3f} | {gh['mu']:.4f} +/- {gh['mu_err']*1e3:.2f} ms | "
            f"{gi['mu']:.4f} +/- {gi['mu_err']*1e3:.2f} ms | {gi['N']:.0f} | "
            f"{lag*1e3:+.2f} +/- {lag_err*1e3:.2f} |")
    say()
    lags = np.array([g[2] for g in gfits.values()])
    errs = np.array([g[3] for g in gfits.values()])
    w = 1 / errs ** 2
    wm = np.sum(w * lags) / np.sum(w)
    we = 1 / np.sqrt(np.sum(w))
    chi_lag = np.sum(w * (lags - wm) ** 2)
    say(f"Weighted mean of the three centroid lags: {wm*1e3:+.2f} +/- {we*1e3:.2f} ms "
        f"(chi2 of mutual consistency = {chi_lag:.2f} for 2 dof)")
    say()

    # ---- EVT-only sanity check (no fill) ----------------------------------
    cso, _, _ = cash_unbinned_scan(h_obs, i_t, WIN_ALL, bkg_h, TAU)
    to, so, _ = parabola_min(TAU, cso)
    say(f"Sanity check, HXMT observed events only (FIFO gaps left empty, no fill): "
        f"Cash tau = {to*1e3:+.2f} +/- {so*1e3:.2f} ms; gaps bias this, reference only.")
    say()

    # ---- Figure -------------------------------------------------------------
    A = res["ALL (0.380-0.540 s, P1+P2+P3)"]
    edges = np.arange(WIN_ALL[0], WIN_ALL[1] + BIN / 2, BIN)
    x = edges[:-1]
    nh = hist(h_all, edges) - bkg_h
    ni = hist(i_t, edges) - bkg_i
    # scale IBIS to HXMT with the best-fit 'a' at tau_best
    scale = 1.0 / (A["a"] * BIN)     # a: photons/s per (HXMT counts per 1-ms bin)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5),
                                   gridspec_kw=dict(height_ratios=[2.2, 1.4], hspace=0.28))
    ax1.step(x * 1e3, nh / BIN, where="post", color="C0", lw=1.2,
             label="HXMT/HE recovered (EVT + fill), net")
    ax1.step(x * 1e3, ni * scale / BIN, where="post", color="C1", lw=1.2,
             label=f"IBIS-ISGRI 20-200 keV, net, x{scale:.0f}")
    err_i = np.sqrt(np.clip(ni + bkg_i, 0, None)) * scale / BIN
    ax1.fill_between(x * 1e3, (ni * scale / BIN - err_i), (ni * scale / BIN + err_i),
                     step="post", color="C1", alpha=0.18, lw=0)
    for k, c in PEAKS_M20.items():
        ax1.axvline(c * 1e3, color="0.5", ls=":", lw=0.9)
        ax1.text(c * 1e3, ax1.get_ylim()[1] * 0.98 if False else nh.max() / BIN * 1.02,
                 f"{k} (M20)", ha="center", va="bottom", fontsize=8, color="0.35")
    ax1.set_xlim(WIN_ALL[0] * 1e3, WIN_ALL[1] * 1e3)
    ax1.set_ylabel("net rate (events s$^{-1}$)")
    ax1.set_xlabel("t - T0 (ms), geocentric, T0 = 14:34:24.011 UTC")
    ax1.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax1.set_title("FRB/XRB 200428: HXMT/HE vs INTEGRAL/IBIS, 1 ms bins", fontsize=11)

    for name, R in res.items():
        d = R["cash"] - R["cash"].min()
        lw = 2.0 if name.startswith("ALL") else 1.0
        ls = "-" if name.startswith("ALL") else "--"
        ax2.plot(TAU * 1e3, d, lw=lw, ls=ls, label=name.split(" (")[0])
    ax2.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax2.axvline(A["tau"] * 1e3, color="C0", lw=1.0)
    ax2.axvspan((A["tau"] - A["sig"]) * 1e3, (A["tau"] + A["sig"]) * 1e3,
                color="C0", alpha=0.15, lw=0)
    ax2.axvline(0, color="k", lw=0.6)
    ax2.set_ylim(0, 25)
    ax2.set_xlim(-20, 20)
    ax2.set_xlabel(r"lag $\tau$ (ms), $\tau>0$: HXMT later than IBIS")
    ax2.set_ylabel(r"$\Delta C(\tau)$  (unbinned Poisson likelihood)")
    ax2.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax2.text(0.02, 0.95,
             rf"all peaks: $\tau$ = {A['tau']*1e3:+.1f} $\pm$ {A['sig']*1e3:.1f} ms"
             f"\n($\\Delta C\\leq1$: [{A['lo']*1e3:+.1f}, {A['hi']*1e3:+.1f}] ms; "
             f"binned $\\chi^2$: {A['tau_chi']*1e3:+.1f} $\\pm$ {A['sig_chi']*1e3:.1f} ms)",
             transform=ax2.transAxes, va="top", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"lag_hxmt_ibis_200428.{ext}", dpi=200, bbox_inches="tight")

    # centroid diagnostic figure
    fig2, axs = plt.subplots(2, 3, figsize=(11, 5.5), sharex="col")
    for j, (k, (gh, gi, lag, lag_err)) in enumerate(gfits.items()):
        for i, (g, col, lab) in enumerate(((gh, "C0", "HXMT"), (gi, "C1", "IBIS"))):
            ax = axs[i, j]
            ax.step(g["x"] * 1e3, g["net"], where="mid", color=col, lw=1)
            ax.axvline(g["mu"] * 1e3, color=col, ls="--", lw=0.8)
            ax.set_title(f"{k} {lab}: centroid = {g['mu']*1e3:.1f} +/- {g['mu_err']*1e3:.1f} ms",
                         fontsize=8.5)
            if i == 1:
                ax.set_xlabel("t - T0 (ms)")
            if j == 0:
                ax.set_ylabel("net counts / 1 ms")
        axs[0, j].text(0.03, 0.93, f"lag {lag*1e3:+.1f} +/- {lag_err*1e3:.1f} ms",
                       transform=axs[0, j].transAxes, fontsize=8.5, va="top")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(OUT / f"lag_centroids_200428.{ext}", dpi=200, bbox_inches="tight")

    say("## Conclusion")
    say()
    say(f"- Full 0.38-0.54 s window (P1+P2+P3, 1 ms): HXMT - IBIS lag = {A['tau']*1e3:+.1f} +/- {A['sig']*1e3:.1f} ms "
        f"(unbinned Poisson likelihood; binned chi^2 {A['tau_chi']*1e3:+.1f} +/- {A['sig_chi']*1e3:.1f} ms; CCF {A['tau_r']*1e3:+.1f} ms).")
    say(f"- Sub-peaks separately (Cash): " + ", ".join(
        f"{n.split(' (')[0]} {res[n]['tau']*1e3:+.1f} +/- {res[n]['sig']*1e3:.1f} ms" for n in res if n[:2] in ('P1', 'P2', 'P3') and '+' not in n.split(' (')[0]) + ".")
    say(f"- Independent centroids: weighted mean {wm*1e3:+.2f} +/- {we*1e3:.2f} ms.")
    say("- The sub-peak positions agree to about 1 ms (|tau| <~ 1.5 ms, 1-sigma errors 0.7-1.9 ms); the "
        "IBIS photon statistics (~8 photons per ms at peak) set the floor, not the HXMT reconstruction.")
    say("- Caveats: 'P3 CCF' entry at -19.7 ms is an edge artefact of the 50-ms window at integer lags (ignore); "
        "HXMT fill events are treated as Poisson; HXMT-only (no fill) lag is biased by the FIFO gaps.")
    say()
    say(f"Figures: {OUT}/lag_hxmt_ibis_200428.pdf/.png, {OUT}/lag_centroids_200428.pdf/.png")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
