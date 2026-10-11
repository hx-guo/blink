"""(A) Independent HE test of the Lesage+2023 (GBM) low-frequency QPO candidate:
period 12.34 (+2.12/-1.67) s in T0+277-508 s (ln B = 2.21).

Light curves (1 s bins):
  v3   : v3 dead-time-corrected recovered net rate (B(t) of freeze_v3 removed),
         277-508 s, i.e. exactly the GBM interval (277-286 and 503-508 s
         contain gap-filled events, at 1 s these are validated in freeze_v3)
  obs  : observed events only, 288-503 s (all real seconds, no fill),
         dedup stream, minus 0.911 B_obs(t); also in three CsI channel bands
Method (Vaughan 2010): periodogram I_j (rms^2/Hz, mean-normalised), red-noise
model S(f) = N f^-alpha + C fitted by Whittle maximum likelihood; statistics
  T_R   = max_j 2 I_j / S_j over all j (blind search)
  T_cand= max over the candidate band 0.0697-0.0947 Hz (a-priori frequency)
  T_K   = sum over K adjacent bins in the candidate band (broad QPO)
calibrated with a parametric bootstrap: model parameters drawn from the
Whittle-fit covariance, periodograms I = S chi2_2/2, refit, recompute
(Vaughan 2010 posterior-predictive style). numpy seed 20221009, 4000 draws.
Sensitivity: injected Lorentzian QPO (FWHM = f0/Q) added to the fitted
model in expectation; amplitude for which T_K reaches p = 0.0027 (3 sigma,
a-priori band) -> fractional rms of the source.
Writes out/qpo12_*.csv, out/qpo12_summary.txt, out/fig_qpo12.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from p221009_sci_lib import *

LOG = open(f"{OUT}/qpo12_summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


F_LO, F_HI, F0 = 1 / (12.34 + 2.12), 1 / (12.34 - 1.67), 1 / 12.34
RNG = np.random.default_rng(20221009)
NSIM = 4000

lc = pd.read_csv(LC_V3)
t_all, box, ch, pw = he_obs()
keep = dedup_cross(t_all, box)
the, chs, pws = t_all[keep], ch[keep], pw[keep]
bk = lc[((lc.t >= -50) & (lc.t < -5)) | ((lc.t >= 20) & (lc.t < 170))]


def rate1s(tt, secs):
    return (np.searchsorted(tt, secs + 1) - np.searchsorted(tt, secs)).astype(float)


k_bkg = rate1s(the, bk.t.to_numpy().astype(float)).sum() / bk.B_obs.sum()
Bobs = dict(zip(lc.t.astype(int), lc.B_obs))

curves = {}
w = (lc.t >= 277) & (lc.t < 508)
curves["v3 277-508 (GBM interval)"] = (lc.t[w].to_numpy(), lc.dt[w].to_numpy(), lc.sig_rec[w].to_numpy() if "sig_rec" in lc else None)
secs = np.arange(288.0, 503.0)
B = np.array([Bobs[int(s)] for s in secs])
r = rate1s(the, secs)
curves["obs 288-503 all ch"] = (secs, r - k_bkg * B, np.sqrt(r))
for nm, (c0, c1) in {"CsI ch30-53": (30, 53), "CsI ch53-99": (53, 99), "CsI ch99-224": (99, 224)}.items():
    m = (chs >= c0) & (chs < c1) & (pws > 75)
    tb = the[m]
    kb = rate1s(tb, bk.t.to_numpy().astype(float)).sum() / bk.B_obs.sum()
    rb = rate1s(tb, secs)
    curves[f"obs 288-503 {nm}"] = (secs, rb - kb * B, np.sqrt(rb))


def periodogram(x, dt=1.0):
    mu = x.mean()
    X = np.fft.rfft(x - mu)
    n = len(x)
    f = np.arange(len(X)) / (n * dt)
    I = 2 * dt / (n * mu ** 2) * np.abs(X) ** 2       # (rms/mean)^2 / Hz
    return f[1:-1] if n % 2 == 0 else f[1:], (I[1:-1] if n % 2 == 0 else I[1:]), mu


def model(p, f):
    lN, al, lC = p
    return np.exp(lN) * f ** -al + np.exp(lC)


def whittle(f, I, p0=None):
    def nll(p):
        S = model(p, f)
        return np.sum(np.log(S) + I / S)
    p0 = p0 if p0 is not None else [np.log(I[:3].mean() * f[1] ** 2), 2.0, np.log(I[-len(I) // 4:].mean())]
    r = optimize.minimize(nll, p0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=20000))
    eps = 1e-4
    h = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pp = np.array(r.x, float)
            def fij(di, dj):
                q = pp.copy(); q[i] += di; q[j] += dj
                return nll(q)
            h[i, j] = (fij(eps, eps) - fij(eps, -eps) - fij(-eps, eps) + fij(-eps, -eps)) / (4 * eps ** 2)
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        cov = np.diag([0.1, 0.1, 0.1])
    return r.x, cov


def stats_of(f, I, S):
    R = 2 * I / S
    cand = (f >= F_LO) & (f <= F_HI)
    T_R = R.max(); T_c = R[cand].max()
    T_K = R[cand].sum() / 2           # ~ Gamma(K) under the null
    return T_R, T_c, T_K, cand


rows = []
PLOT = {}
for nm, (tt, x, sx) in curves.items():
    f, I, mu = periodogram(x)
    p, cov = whittle(f, I)
    S = model(p, f)
    T_R, T_c, T_K, cand = stats_of(f, I, S)
    K = cand.sum()
    jbest = np.argmax(np.where(cand, 2 * I / S, -1))
    # Poisson level (rms/mean)^2/Hz for comparison with the red noise at F0
    pois = 2 * np.mean(sx ** 2) / mu ** 2 if sx is not None else np.nan
    # bootstrap
    sim = np.empty((NSIM, 3))
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(3)) if np.all(np.linalg.eigvalsh(cov) > 0) else np.diag(np.sqrt(np.abs(np.diag(cov))))
    for i in range(NSIM):
        ps = p + L @ RNG.standard_normal(3)
        Is = model(ps, f) * RNG.chisquare(2, len(f)) / 2
        # refit each simulation (Nelder-Mead from the drawn parameters)
        def nll(q):
            Sq = model(q, f); return np.sum(np.log(Sq) + Is / Sq)
        q = optimize.minimize(nll, ps, method="Nelder-Mead", options=dict(maxiter=4000)).x
        sim[i] = stats_of(f, Is, model(q, f))[:3]
    pR = (np.sum(sim[:, 0] >= T_R) + 1) / (NSIM + 1)
    pc = (np.sum(sim[:, 1] >= T_c) + 1) / (NSIM + 1)
    pK = (np.sum(sim[:, 2] >= T_K) + 1) / (NSIM + 1)
    # analytic (fixed-model) reference values
    pc_an = 1 - (1 - np.exp(-T_c / 2)) ** K
    pK_an = stats.gamma.sf(T_K, K)
    # sensitivity: Lorentzian of fractional rms r, FWHM = F0/Q; expected T_K
    S0 = model(p, F0)
    sens = {}
    for Q in (2.0, 4.0, 10.0):
        fw = F0 / Q
        Lshape = (fw / (2 * np.pi)) / ((f - F0) ** 2 + (fw / 2) ** 2)    # integrates to 1
        thrK = np.quantile(sim[:, 2], 1 - 0.0027)
        # T_K expectation with signal: sum (S + r^2 L)/S over band
        base = cand.sum()
        need = thrK - base * 1.0
        add = (Lshape[cand] / S[cand]).sum()
        sens[Q] = np.sqrt(max(need, 0) / add)
    say(f"[{nm}] N={len(x)} mean={mu:.0f}  fit: alpha={p[1]:.2f}+-{np.sqrt(cov[1,1]):.2f}, "
        f"S(f0=0.081 Hz)={S0:.3g} (rms/mean)^2/Hz, white C={np.exp(p[2]):.3g}, Poisson {pois:.3g} -> red/Poisson at f0 = {S0/pois:.0f}")
    say(f"      candidate band 0.070-0.095 Hz: K={K} bins, max 2I/S = {T_c:.2f} at {f[jbest]:.4f} Hz ({1/f[jbest]:.1f} s); "
        f"p(max) = {pc:.3f} [fixed-model {pc_an:.3f}],  band sum T_K = {T_K:.2f}, p = {pK:.3f} [fixed {pK_an:.3f}];  "
        f"blind T_R = {T_R:.2f}, p = {pR:.3f}")
    say(f"      3-sigma (a-priori band) detectable fractional rms of a Lorentzian at 12.34 s: "
        + ", ".join(f"Q={Q:g}: {100*v:.1f}%" for Q, v in sens.items()))
    rows.append(dict(curve=nm, N=len(x), mean=mu, alpha=p[1], S_f0=S0, C=np.exp(p[2]), poisson=pois,
                     K=K, Tc=T_c, f_best=f[jbest], p_max=pc, p_max_fixed=pc_an, TK=T_K, p_band=pK,
                     p_band_fixed=pK_an, TR=T_R, p_blind=pR, rms3_Q2=sens[2.0], rms3_Q4=sens[4.0], rms3_Q10=sens[10.0]))
    PLOT[nm] = (f, I, S, pois)
pd.DataFrame(rows).to_csv(f"{OUT}/qpo12_results.csv", index=False, float_format="%.5g")

fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for a, (nm, (f, I, S, pois)) in zip(axs.flat, PLOT.items()):
    a.loglog(f, I, "-", lw=0.7, drawstyle="steps-mid", label="periodogram")
    a.loglog(f, S, "r-", lw=1, label="Whittle fit N f^-a + C")
    a.axvspan(F_LO, F_HI, color="orange", alpha=0.3, label="Lesage+23 12.3 s")
    if np.isfinite(pois):
        a.axhline(pois, color="k", ls=":", lw=0.8, label="Poisson level")
    a.set_title(nm, fontsize=9); a.set_xlabel("f (Hz)", fontsize=8); a.legend(fontsize=6)
axs[0, 0].set_ylabel("(rms/mean)^2 / Hz")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_qpo12.png", dpi=130)
fig, a = plt.subplots(figsize=(9, 3.2))
for nm, (tt, x, sx) in list(curves.items())[:2]:
    a.plot(tt, x, lw=0.8, label=nm)
a.set_yscale("log"); a.set_xlabel("t - T0 (s)"); a.set_ylabel("net evt/s"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_qpo12_lc.png", dpi=130)
say("done")
