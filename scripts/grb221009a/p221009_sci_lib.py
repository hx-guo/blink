"""Event loaders and timing estimators for the HE variability scouting
(p221009_sci_*.py). Deterministic: no random numbers anywhere."""
import numpy as np
from scipy import stats
from p221009_sci_common import *

GECAM_T0_MET = 55862220.0      # 2022-10-09T13:17:00 UTC in GECAM MET (as p221009_timing_common)
LTT_GECAM = 5.7e-3             # GECAM-C sees the wavefront 5.5-6.0 ms after HXMT (timing.md)
TAU_TRIG = 3.0e-6              # box-level non-paralysable busy per trigger (freeze_v3 item 6)


def he_obs(chmin=20):
    """Observed (non-filled) HE events, SEC markers dropped, ch >= chmin.
    Returns t (s rel T0), box (0..2), ch, pw, sorted by time."""
    d = np.load(EVT_REC)
    t, fill, ch, pw, box = d["t"], d["fill"], d["ch"], d["pw"], d["box"]
    sec = (ch == 0) & (pw == 0)
    k = ~fill & ~sec & (ch >= chmin)
    t, box, ch, pw = t[k], box[k], ch[k], pw[k]
    o = np.argsort(t, kind="stable")          # the cache is ordered box by box
    return t[o], box[o], ch[o], pw[o]


def dedup_cross(t, box, win=2.1e-6):
    """Mask that drops an event when an event of another box lies within
    +-win before it (cross-box coincidences, ~5% of events per box)."""
    keep = np.ones(len(t), bool)
    for b in range(3):
        m = box == b
        for b2 in range(b):                    # compare with earlier boxes only
            tb2 = t[box == b2]
            k = np.searchsorted(tb2, t[m])
            d0 = np.abs(t[m] - tb2[np.clip(k - 1, 0, len(tb2) - 1)])
            d1 = np.abs(tb2[np.clip(k, 0, len(tb2) - 1)] - t[m])
            keep[np.flatnonzero(m)[np.minimum(d0, d1) <= win]] = False
    return keep


def he_fill_frac(lo, hi):
    d = np.load(EVT_REC)
    w = (d["t"] >= lo) & (d["t"] < hi)
    return float(d["fill"][w].mean())


def gecam_events(lo=-300.0, hi=700.0):
    """All 12 GRDs of GECAM-C, times shifted to HXMT frame (constant 5.7 ms).
    Returns dict(t, det, gain, pi, emid) sorted by time; emid in keV."""
    from astropy.io import fits
    f = fits.open(GECAM_EVT, memmap=True)
    T, D, G, P = [], [], [], []
    for i in range(1, 13):
        ev = f[f"EVENTS{i:02d}"].data
        tr = ev["TIME"] - GECAM_T0_MET - LTT_GECAM
        m = (tr > lo) & (tr < hi)
        T.append(tr[m]); D.append(np.full(m.sum(), i, np.uint8))
        G.append(ev["GAIN_TYPE"][m].astype(np.uint8)); P.append(ev["PI"][m].astype(np.int16))
    t = np.concatenate(T); o = np.argsort(t, kind="stable")
    eb = f["EBOUNDS"].data
    emid = np.sqrt(eb["E_MIN"] * eb["E_MAX"])
    pi = np.concatenate(P)[o]
    return dict(t=t[o], det=np.concatenate(D)[o], gain=np.concatenate(G)[o], pi=pi,
                emid=emid[pi])


def gecam_gti():
    from astropy.io import fits
    g = fits.open(GECAM_EVT, memmap=True)["GTI"].data
    return np.c_[g["START"], g["STOP"]] - GECAM_T0_MET


def leahy(t, lo, hi, seglen, dt):
    """Leahy PDS averaged over contiguous segments of length seglen in [lo, hi).
    Returns f, P (mean), M (segments), N (mean counts/segment), per-seg counts."""
    nb = int(round(seglen / dt))
    M = int(np.floor((hi - lo) / seglen + 1e-9))
    acc = np.zeros(nb // 2)
    ns = []
    for s in range(M):
        a = lo + s * seglen
        i0, i1 = np.searchsorted(t, [a, a + seglen])
        idx = ((t[i0:i1] - a) / dt).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < nb)]
        n = np.bincount(idx, minlength=nb).astype(float)
        N = n.sum()
        ns.append(N)
        if N == 0:
            continue
        F = np.fft.rfft(n)[1:nb // 2 + 1]
        acc += 2.0 * np.abs(F[:nb // 2]) ** 2 / N
    f = np.arange(1, nb // 2 + 1) / seglen
    return f, acc / M, M, float(np.mean(ns)), np.array(ns)


def logbin(f, P, fac=1.1, fmin=None):
    """Logarithmic rebinning (geometric factor fac). Returns fc, Pm, K (bins)."""
    fmin = f[0] if fmin is None else fmin
    edges = [fmin]
    while edges[-1] < f[-1]:
        edges.append(max(edges[-1] * fac, edges[-1] + (f[1] - f[0])))
    edges = np.array(edges)
    k = np.digitize(f, edges) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    K = np.bincount(k[ok], minlength=len(edges) - 1)
    S = np.bincount(k[ok], P[ok], minlength=len(edges) - 1)
    Sf = np.bincount(k[ok], f[ok], minlength=len(edges) - 1)
    m = K > 0
    return Sf[m] / K[m], S[m] / K[m], K[m]


def pair_g(tlist, lo, hi, seglen=1.0, dt=2.0 ** -20, maxlag=0.5, cross=None):
    """Pair-correlation g(u) = conditional rate at lag u / mean rate, from
    binned counts (bin dt) via FFT autocorrelation in segments of seglen.
    tlist: sorted event times; cross: second stream for the cross-correlation
    (then lags are t2 - t1, both signs). Returns lag (s, bin centres k*dt), g, pair counts, expected pairs."""
    nb = int(round(seglen / dt))
    L = 2 * nb
    K = int(round(maxlag / dt))
    num = np.zeros(2 * K + 1)
    den = np.zeros(2 * K + 1)
    M = int(np.floor((hi - lo) / seglen + 1e-9))
    lag_k = np.arange(-K, K + 1)
    for s in range(M):
        a = lo + s * seglen
        def binned(tt):
            i0, i1 = np.searchsorted(tt, [a, a + seglen])
            idx = ((tt[i0:i1] - a) / dt).astype(np.int64)
            idx = idx[(idx >= 0) & (idx < nb)]
            return np.bincount(idx, minlength=nb).astype(float)
        n1 = binned(tlist)
        n2 = n1 if cross is None else binned(cross)
        F1 = np.fft.rfft(n1, L); F2 = np.fft.rfft(n2, L)
        c = np.fft.irfft(np.conj(F1) * F2, L)       # c[k] = sum_i n1[i] n2[i+k]
        c = np.round(c)                              # integer pair counts
        cc = np.r_[c[L - K:], c[:K + 1]]             # lags -K..K
        num += cc
        den += (nb - np.abs(lag_k)) * (n1.sum() / nb) * (n2.sum() / nb)
    g = num / np.where(den > 0, den, np.nan)
    return lag_k * dt, g, num, den


def lograte(lag, num, den, hi=0.5, fac=1.25):
    """Positive lags rebinned: 2-us wide bins centred on the 2-us time quantum
    up to 41 us, then logarithmic. Returns rows (centre, g, sigma) with sigma
    from Poisson statistics of the pair counts."""
    edges = list(np.arange(1e-6, 41.1e-6, 2e-6))
    while edges[-1] < hi:
        edges.append(edges[-1] * fac)
    out = []
    for e, e2 in zip(edges[:-1], edges[1:]):
        m = (lag >= e) & (lag < e2)
        if m.any() and num[m].sum() > 0:
            gm = num[m].sum() / den[m].sum()
            out.append((np.sqrt(e * e2), gm, gm / np.sqrt(num[m].sum())))
    return np.array(out)


def gamma_sf(x, dof):
    """P(chi2_dof/dof > x)"""
    return stats.chi2.sf(x * dof, dof)


def gamma_isf(p, dof):
    return stats.chi2.isf(p, dof) / dof
