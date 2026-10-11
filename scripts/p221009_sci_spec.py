"""Shared spectral estimators for the HE variability scouting: QPO
sensitivity on averaged Leahy PDS (as p221009_sci_pds.py) and the Vaughan
(2010) red-noise test of a low-frequency candidate (as p221009_sci_qpo12.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from p221009_sci_lib import logbin, gamma_sf, gamma_isf

P3 = stats.norm.sf(3.0)


def fit_continuum(f, P, M):
    fc, Pc, K = logbin(f, P, 1.1)
    W0 = P[f > 1000].mean() if f[-1] > 1000 else P[-len(P) // 4:].mean()
    def res(p):
        W, lA, al = p
        m = W + np.exp(lA) * fc ** -al
        return (Pc - m) / (m / np.sqrt(M * K))
    A0 = max(Pc[:3].mean() - W0, 1e-3)
    r = optimize.least_squares(res, [W0, np.log(A0), 2.0], bounds=([0.5, -30, 0], [40, 30, 5]))
    return r.x[0], np.exp(r.x[1]), r.x[2]


def qpo_sens(f, P, M, I, S):
    """3 sigma (global over Q~4.5 log bins 2 Hz-Nyquist) fractional rms of a
    Lorentzian QPO; returns table and white level."""
    W, A, al = fit_continuum(f, P, M)
    model = lambda x: W + A * x ** -al
    edges = [2.0]
    while edges[-1] * 1.25 <= f[-1]:
        edges.append(edges[-1] * 1.25)
    nbins = len(edges) - 1
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        K = m.sum()
        if K == 0:
            continue
        df = K * (f[1] - f[0]); mod = model(f[m]).mean()
        x = P[m].mean() / mod
        dof = 2 * M * K
        xthr = gamma_isf(P3 / nbins, dof)
        out.append(dict(f=f[m].mean(), df=df, x=x, pglob=min(1, gamma_sf(x, dof) * nbins),
                        rms3=np.sqrt(2 * (xthr - 1) * mod * df * I / S ** 2)))
    return pd.DataFrame(out), W, al


def rms_at(T, fs):
    return [T.rms3.iloc[np.argmin(abs(T.f - x))] for x in fs]


# ---------------------------------------------------------------- Vaughan 2010
F_LO, F_HI, F0 = 1 / (12.34 + 2.12), 1 / (12.34 - 1.67), 1 / 12.34


def periodogram(x, dt=1.0):
    mu = x.mean()
    X = np.fft.rfft(x - mu)
    n = len(x)
    f = np.arange(len(X)) / (n * dt)
    I = 2 * dt / (n * mu ** 2) * np.abs(X) ** 2
    sl = slice(1, -1) if n % 2 == 0 else slice(1, None)
    return f[sl], I[sl], mu


def model(p, f):
    lN, al, lC = p
    return np.exp(np.clip(lN, -50, 50)) * f ** -np.clip(al, -1, 6) + np.exp(np.clip(lC, -50, 50))


def _nll(p, f, I):
    S = model(p, f)
    return np.sum(np.log(S) + I / S)


def whittle(f, I, p0=None):
    p0 = p0 if p0 is not None else [np.log(I[:3].mean() * f[1] ** 2), 2.0, np.log(max(I[-len(I) // 4:].mean(), 1e-12))]
    r = optimize.minimize(_nll, p0, args=(f, I), method="Nelder-Mead",
                          options=dict(xatol=1e-6, fatol=1e-8, maxiter=20000))
    eps = 1e-3
    h = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            def fij(di, dj):
                q = np.array(r.x, float); q[i] += di; q[j] += dj
                return _nll(q, f, I)
            h[i, j] = (fij(eps, eps) - fij(eps, -eps) - fij(-eps, eps) + fij(-eps, -eps)) / (4 * eps ** 2)
    try:
        cov = np.linalg.inv(h)
        if not np.all(np.linalg.eigvalsh(cov) > 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.diag([0.05, 0.05, 0.05])
    return r.x, cov


def vaughan_test(x, sx=None, nsim=2000, seed=20221009):
    """Returns dict with candidate-band statistics, bootstrap p-values and the
    3 sigma (a-priori band) detectable fractional rms of a Q=4 Lorentzian."""
    rng = np.random.default_rng(seed)
    f, I, mu = periodogram(x)
    p, cov = whittle(f, I)
    S = model(p, f)
    cand = (f >= F_LO) & (f <= F_HI)
    R = 2 * I / S
    Tc, TK, TR = R[cand].max(), R[cand].sum() / 2, R.max()
    L = np.linalg.cholesky(cov)
    sim = np.empty((nsim, 3))
    for i in range(nsim):
        ps = p + L @ rng.standard_normal(3)
        Is = model(ps, f) * rng.chisquare(2, len(f)) / 2
        q = optimize.minimize(_nll, ps, args=(f, Is), method="Nelder-Mead", options=dict(maxiter=4000)).x
        Rs = 2 * Is / model(q, f)
        sim[i] = Rs[cand].max(), Rs[cand].sum() / 2, Rs.max()
    pc = (np.sum(sim[:, 0] >= Tc) + 1) / (nsim + 1)
    pK = (np.sum(sim[:, 1] >= TK) + 1) / (nsim + 1)
    pR = (np.sum(sim[:, 2] >= TR) + 1) / (nsim + 1)
    thrK = np.quantile(sim[:, 1], 1 - 0.0027)
    fw = F0 / 4.0
    Lsh = (fw / (2 * np.pi)) / ((f - F0) ** 2 + (fw / 2) ** 2)
    rms3 = np.sqrt(max(thrK - cand.sum(), 0) / (Lsh[cand] / S[cand]).sum())
    pois = 2 * np.mean(sx ** 2) / mu ** 2 if sx is not None else np.nan
    jb = np.flatnonzero(cand)[np.argmax(R[cand])]
    return dict(N=len(x), mean=mu, alpha=p[1], S_f0=float(model(p, np.array([F0]))[0]), poisson=pois,
                Tc=Tc, f_best=f[jb], p_max=pc, TK=TK, p_band=pK, TR=TR, p_blind=pR, rms3_Q4=rms3), (f, I, S)
