"""Steps 3-4 of the HE variability scouting: PDS, QPO sensitivity and Haar
variability scales in the real-event intervals, HE against GECAM-C.

HE stream: observed events, ch >= 20, SEC removed, cross-box coincidences
removed (dedup_cross; ~5% of events are the second hit of one photon in
another box, which raises the white level of the summed stream to ~2.27).

PDS: 1 s segments, 2^-12 s bins (1-2048 Hz), Leahy, averaged; continuum
W + A f^-alpha fitted to the log-binned powers. QPO search: P/model in
log bins of width 0.22 f (Q ~ 4.5) and at 1 Hz resolution; probabilities
from chi2_{2MK}/(2MK). Sensitivity: 3 sigma (global over the bins searched)
threshold, converted to the fractional rms of a Lorentzian whose FWHM equals
the bin width (in-band power doubled), rms^2 = 2 (x-1) model df I/S^2,
with I the total and S the source rate of the stream.

Haar: S(D) = var(d) - eta <N_l+N_r>, d = N_r - N_l, var taken within
blocks of max(2 s, 16 D) (local linear detrending); eta = W/2 with W the
white level fitted to the same interval (dead time and residual
coincidences); sigma_S = sqrt(2/n) var(d) (+) the eta error.

GECAM-C: events only at T0+330-678 s (HG all, LG 0.4-3 MeV, 12 GRDs);
elsewhere BTIME (12 GRDs, 50 ms; HG ch0-5, LG ch9-11 = 0.38-3.2 MeV).
Writes out/pds_*.csv, out/haar_*.csv, out/pds_summary.txt, figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from astropy.io import fits
from p221009_sci_lib import *

LOG = open(f"{OUT}/pds_summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


SEGS = {
    "precursor": (-1.0, 9.0),
    "precursor_wide": (-5.0, 15.0),
    "P0rise": (176.0, 186.0),
    "decay193": (193.0, 215.0),
    "tail288": (288.0, 330.0),
    "tail330": (330.0, 440.0),
    "flare_rise": (440.0, 503.0),
    "flare_decay": (530.0, 600.0),
}
SEG_LEN, DT = 1.0, 2.0 ** -12
P3 = stats.norm.sf(3.0)                 # one-sided 3 sigma
lc = pd.read_csv(LC_V3)
seg1 = pd.read_csv(f"{OUT}/segments_1s.csv")
BRIGHT = seg1[(seg1.real) & (seg1.t >= 170) & (seg1.he_net_dt > 5000)].t.to_numpy()

t_all, box, ch, pw = he_obs()
keep = dedup_cross(t_all, box)
the = t_all[keep]
say(f"HE stream: {len(the)} events after removing {(~keep).sum()} cross-box coincidences")

# background of this stream: k * B_obs(t), k from the source-free intervals
def stream_rate_1s(tt, secs):
    secs = np.asarray(secs, float)
    return (np.searchsorted(tt, secs + 1) - np.searchsorted(tt, secs)).astype(float)

bk = lc[((lc.t >= -50) & (lc.t < -5)) | ((lc.t >= 20) & (lc.t < 170))]
k_bkg = stream_rate_1s(the, bk.t.to_numpy()).sum() / bk.B_obs.sum()
say(f"background scale: stream bkg rate / B_obs = {k_bkg:.4f}")
Bsec = dict(zip(lc.t.astype(int), k_bkg * lc.B_obs))


def he_IS(secs):
    I = stream_rate_1s(the, secs).mean()
    B = np.mean([Bsec[int(s)] for s in secs])
    return I, I - B, B


# GECAM-C events and BTIME
ge = np.load(f"{OUT}/gecam_c_evt.npz")
gsel = {"HG": ge["t"][ge["gain"] == 0],
        "LG0.4-3": ge["t"][(ge["gain"] == 1) & (ge["emid"] >= 400) & (ge["emid"] < 3000)]}
gB = {k: ((v >= -150) & (v < -68)).sum() / 82.0 for k, v in gsel.items()}
say("GECAM-C pre-burst rates (-150..-68 s): " + ", ".join(f"{k} {v:.0f}/s" for k, v in gB.items()))
bt = fits.open(GECAM_BTIME)
bt_t = bt["SPECTRUM01"].data["STARTTIME"] - GECAM_T0_MET - LTT_GECAM
bt_c = {"HG": 0, "LG0.4-3": 0}
for i in range(1, 13):
    c = bt[f"SPECTRUM{i:02d}"].data["COUNTS"]
    bt_c["HG"] = bt_c["HG"] + c[:, 0:6].sum(1)
    bt_c["LG0.4-3"] = bt_c["LG0.4-3"] + c[:, 9:12].sum(1)


# GECAM-C source rate: the GECAM-C background is not stationary here (the
# pre-burst GTI is in a high-background zone; HG BTIME 170-330 s is far from
# Poisson), so S_G is taken from a per-second regression on the HE source
# rate in a window around the interval: G(t) = poly(t) + S_HE(t)/R.
REG_WIN = {"evt": [(330.0, 678.0)], "bt": [(-30.0, 30.0), (165.0, 216.0), (280.0, 330.0), (330.0, 678.0)]}
REG = {}


def he_S_1s(secs):
    secs = np.asarray(secs, float)
    return stream_rate_1s(the, secs) - np.array([Bsec[int(x)] for x in secs])


def gecam_rate_1s(kind, gn, secs):
    secs = np.asarray(secs, float)
    if kind == "evt":
        tg = gsel[gn]
        return (np.searchsorted(tg, secs + 1) - np.searchsorted(tg, secs)).astype(float)
    i0 = np.searchsorted(bt_t, secs - 1e-6)
    return np.array([bt_c[gn][i:i + 20].sum() for i in i0], float)


def gecam_S(key, lo, hi):
    kind, gn = key
    win = [w for w in REG_WIN[kind] if w[0] <= lo and hi <= w[1]][0]
    if (key, win) not in REG:
        secs = np.arange(win[0], win[1], 1.0)
        real = seg1.set_index("t").real.reindex(secs.astype(int)).fillna(False).to_numpy()
        hs = he_S_1s(secs); g = gecam_rate_1s(kind, gn, secs)
        x = (secs - secs.mean()) / (win[1] - win[0])
        X = np.c_[np.ones_like(x), x, x ** 2, hs]
        c, *_ = np.linalg.lstsq(X[real], g[real], rcond=None)
        r = g[real] - X[real] @ c
        REG[(key, win)] = c
        say(f"[REG] GECAM-C {kind} {gn} {win}: 1/R = {c[3]:.5f} (R = {1/c[3]:.1f}), bkg poly {np.round(c[:3], 1)}, "
            f"rms resid {r.std():.1f} (Poisson {np.sqrt(g[real].mean()):.1f}), n = {real.sum()}")
    c = REG[(key, win)]
    return float(c[3] * he_S_1s(np.arange(lo, hi - 0.5, 1.0)).mean())


def fit_continuum(f, P, M):
    fc, Pc, K = logbin(f, P, 1.1)
    W0 = P[f > 1000].mean() if f[-1] > 1000 else P[-len(P) // 4:].mean()
    def res(p):
        W, lA, al = p
        m = W + np.exp(lA) * fc ** -al
        return (Pc - m) / (m / np.sqrt(M * K))
    A0 = max(Pc[:3].mean() - W0, 1e-3)
    r = optimize.least_squares(res, [W0, np.log(A0), 2.0], bounds=([0.5, -30, 0], [4, 30, 5]))
    W, lA, al = r.x
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J); eW = np.sqrt(cov[0, 0])
    except np.linalg.LinAlgError:
        eW = np.nan
    return W, np.exp(lA), al, eW


def qpo(f, P, M, I, S, label):
    """Search + sensitivity. Returns per-log-bin table and summary dict."""
    W, A, al, eW = fit_continuum(f, P, M)
    model = lambda x: W + A * x ** -al
    out = []
    edges = [2.0]
    while edges[-1] * 1.25 <= f[-1]:
        edges.append(edges[-1] * 1.25)
    edges = np.array(edges)
    nbins = len(edges) - 1
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        K = m.sum()
        if K == 0:
            continue
        fm = f[m].mean(); df = K * (f[1] - f[0])
        mod = model(f[m]).mean()
        x = P[m].mean() / mod
        dof = 2 * M * K
        p1 = gamma_sf(x, dof)
        xthr = gamma_isf(P3 / nbins, dof)
        rms3 = np.sqrt(2 * (xthr - 1) * mod * df * I / S ** 2)
        xul = max(x - 1, 0) + stats.norm.isf(0.05) / np.sqrt(M * K)
        rmsul = np.sqrt(2 * xul * mod * df * I / S ** 2)
        out.append(dict(seg=label, f=fm, df=df, K=K, M=M, x=x, p1=p1, pglob=min(1, p1 * nbins),
                        rms3=rms3, rms_ul95=rmsul, model=mod))
    T = pd.DataFrame(out)
    # 1 Hz resolution search above 4 Hz
    m = f >= 4
    x1 = P[m] / model(f[m])
    p1 = gamma_sf(x1, 2 * M)
    j = np.argmin(p1)
    summ = dict(seg=label, W=W, eW=eW, A=A, alpha=al, I=I, S=S, M=M,
                min_pglob_logbin=T.pglob.min(), f_min_logbin=T.f[T.pglob.idxmin()],
                min_pglob_1Hz=min(1, p1[j] * m.sum()), f_min_1Hz=f[m][j])
    return T, summ


def rms_at(T, fs):
    return [T.rms3.iloc[np.argmin(abs(T.f - x))] for x in fs]


def haar_curve(tt, lo, hi, eta, eeta, scales):
    rows = []
    tt = tt[np.searchsorted(tt, lo):np.searchsorted(tt, hi)]
    for D in scales:
        blk = max(2.0, 16 * D)
        nblk = int(np.floor((hi - lo) / blk + 1e-9))
        if nblk < 1:
            continue
        npair = int(np.floor(blk / (2 * D) + 1e-9))
        ds, ns = [], []
        for bi in range(nblk):
            a = lo + bi * blk
            e = a + D * np.arange(2 * npair + 1)
            n = np.histogram(tt, e)[0].astype(float)
            d = n[1::2] - n[0::2]
            ds.append(d - d.mean()); ns.append(n[0::2] + n[1::2])
        d = np.concatenate(ds); nn = np.concatenate(ns)
        nd = len(d) - nblk
        v = (d ** 2).sum() / nd
        S = v - eta * nn.mean()
        sig = np.hypot(np.sqrt(2.0 / nd) * v, np.nan_to_num(eeta) * nn.mean())
        rows.append(dict(D=D, S=S, sig=sig, snr=S / sig, mean_N=nn.mean(), var=v, npair=nd))
    return pd.DataFrame(rows)


SCALES = 0.001 * 2.0 ** np.arange(0, 11)       # 1 ms .. 1.024 s
FS = (3, 10, 30, 100, 300, 1000)
all_T, summ_rows, haar_rows, gT = [], [], [], {}
PDSPLOT = {}

for seg, (lo, hi) in list(SEGS.items()) + [("stack_bright", (None, None))]:
    if seg == "stack_bright":
        secs = BRIGHT
        acc = None
        for s in secs:
            f, P, M, N, _ = leahy(the, s, s + 1, SEG_LEN, DT)
            acc = P if acc is None else acc + P
        P = acc / len(secs); M = len(secs)
        I, S, B = he_IS(secs)
        ff = 0.0
    else:
        secs = np.arange(lo, hi, 1.0)
        f, P, M, N, _ = leahy(the, lo, hi, SEG_LEN, DT)
        I, S, B = he_IS(secs)
        ff = he_fill_frac(lo, hi)
    T, sm = qpo(f, P, M, I, S, seg)
    sm.update(inst="HE", fill=ff, B=B)
    all_T.append(T.assign(inst="HE")); summ_rows.append(sm)
    PDSPLOT[(seg, "HE")] = (f, P, M, I, S, sm)
    say(f"[HE] {seg:14s} M={M:3d} I={I:7.0f} S={S:7.0f} B={B:6.0f} fill={ff:.3f}  W={sm['W']:.4f}+-{sm['eW']:.4f} "
        f"alpha={sm['alpha']:.2f}  min p_glob {sm['min_pglob_logbin']:.3f} @ {sm['f_min_logbin']:.1f} Hz (Q~4.5), "
        f"{sm['min_pglob_1Hz']:.3f} @ {sm['f_min_1Hz']:.0f} Hz (1 Hz)  rms3sig[{','.join(map(str, FS))} Hz] = "
        + " ".join(f"{100*x:.1f}%" for x in rms_at(T, FS)))
    if seg != "stack_bright":
        H = haar_curve(the, lo, hi, sm["W"] / 2, sm["eW"] / 2, SCALES)
        H["seg"], H["inst"] = seg, "HE"
        H["frac_rms_3sig"] = np.sqrt(3 * H.sig) / (S * H.D)
        haar_rows.append(H)
        sig = H[H.snr > 3]
        say(f"      Haar: shortest D with S/sig>3: {sig.D.min() if len(sig) else None};  snr " +
            " ".join(f"{d*1e3:g}ms:{x:.1f}" for d, x in zip(H.D, H.snr)))
    # GECAM-C
    if seg != "stack_bright" and lo >= 330 and hi <= 678:
        for gn, tg in gsel.items():
            f, P, M, N, _ = leahy(tg, lo, hi, SEG_LEN, DT)
            Ig = N / SEG_LEN; Sg = gecam_S(("evt", gn), lo, hi)
            Tg, smg = qpo(f, P, M, Ig, Sg, seg)
            smg.update(inst=f"GECAM-C {gn} evt", fill=0, B=Ig - Sg)
            all_T.append(Tg.assign(inst=f"GECAM {gn}")); summ_rows.append(smg)
            PDSPLOT[(seg, gn)] = (f, P, M, Ig, Sg, smg)
            say(f"[GC] {seg:14s} {gn:8s} I={Ig:7.0f} S={Sg:7.0f}  W={smg['W']:.4f} alpha={smg['alpha']:.2f}  "
                f"min p_glob {smg['min_pglob_logbin']:.3f} @ {smg['f_min_logbin']:.1f} Hz  rms3sig = "
                + " ".join(f"{100*x:.1f}%" for x in rms_at(Tg, FS)))
            H = haar_curve(tg, lo, hi, smg["W"] / 2, smg["eW"] / 2, SCALES)
            H["seg"], H["inst"] = seg, f"GECAM {gn}"
            H["frac_rms_3sig"] = np.sqrt(3 * H.sig) / (Sg * H.D)
            haar_rows.append(H)
            sig = H[H.snr > 3]
            say(f"      Haar: shortest D with S/sig>3: {sig.D.min() if len(sig) else None};  snr " +
                " ".join(f"{d*1e3:g}ms:{x:.1f}" for d, x in zip(H.D, H.snr)))
    # BTIME (<= 10 Hz) for every interval, against HE binned the same way
    if seg != "stack_bright":
        L = 6.4
        nseg = int((hi - lo) // L)
        if nseg >= 1:
            for gn in ("HG", "LG0.4-3"):
                acc = 0; Ns = []
                for s in range(nseg):
                    a = lo + s * L
                    i0 = np.searchsorted(bt_t, a - 1e-6)
                    c = bt_c[gn][i0:i0 + 128].astype(float)
                    F = np.fft.rfft(c)[1:65]
                    acc = acc + 2 * np.abs(F) ** 2 / c.sum(); Ns.append(c.sum())
                Pb = acc / nseg; fb = np.arange(1, 65) / L
                Ib = np.mean(Ns) / L
                Sb = gecam_S(("bt", gn), lo, lo + nseg * L)
                # white level from 5-10 Hz, threshold over log bins 0.3-10 Hz
                W = Pb[fb > 5].mean()
                m = (fb >= 2) & (fb < 8)
                K = m.sum(); dof = 2 * nseg * K
                xthr = gamma_isf(P3, dof)
                rms3 = np.sqrt(2 * (xthr - 1) * W * K / L * Ib / Sb ** 2) if Sb > 0 else np.nan
                # HE in the same 2-8 Hz band, same segmentation
                fh, Ph, Mh, Nh, _ = leahy(the, lo, lo + nseg * L, L, 0.05)
                Ih, Sh, _ = he_IS(np.arange(lo, lo + nseg * L - 0.5, 1.0))
                mh = (fh >= 2) & (fh < 8)
                Wh = Ph[fh > 5].mean()
                xh = gamma_isf(P3, 2 * nseg * mh.sum())
                rmsh = np.sqrt(2 * (xh - 1) * Wh * mh.sum() / L * Ih / Sh ** 2)
                say(f"[BT] {seg:14s} GECAM-C BTIME {gn:8s} I={Ib:7.0f} S={Sb:7.0f} W={W:.3f}: rms3sig(2-8 Hz, single band) "
                    f"{100*rms3:.2f}%   HE same band/segments {100*rmsh:.2f}%  (HE/GECAM {rmsh/rms3:.2f})")
                summ_rows.append(dict(seg=seg, inst=f"GECAM-C BTIME {gn}", I=Ib, S=Sb, W=W, M=nseg,
                                      rms3_2to8Hz=rms3, HE_rms3_2to8Hz=rmsh))

pd.concat(all_T).to_csv(f"{OUT}/pds_qpo_bins.csv", index=False, float_format="%.6g")
pd.DataFrame(summ_rows).to_csv(f"{OUT}/pds_summary.csv", index=False, float_format="%.6g")
Hall = pd.concat(haar_rows)
Hall.to_csv(f"{OUT}/haar_all.csv", index=False, float_format="%.6g")

# timing.md estimator on flare 440-500 for comparison: all events incl. coincidences,
# eta = 1, minus quiet-interval Haar excess scaled by the background ratio
lo, hi = 440.0, 500.0
old = haar_curve(t_all, lo, hi, 1.0, 0.0, SCALES)
q = haar_curve(t_all, 20.0, 170.0, 1.0, 0.0, SCALES)
Bq = np.mean([Bsec[s] for s in range(20, 170)]); Bf = np.mean([Bsec[s] for s in range(440, 500)])
old["S"] = old.S - Bf / Bq * q.S
old["sig"] = np.hypot(old.sig, Bf / Bq * q.sig); old["snr"] = old.S / old.sig
say("[timing.md-like] flare 440-500 (all events, eta=1, bkg-scaled): snr " +
    " ".join(f"{d*1e3:g}ms:{x:.1f}" for d, x in zip(old.D, old.snr)))
new = Hall[(Hall.seg == "flare_rise") & (Hall.inst == "HE")]
old.assign(seg="flare440-500 timing-like").to_csv(f"{OUT}/haar_timinglike.csv", index=False, float_format="%.6g")

# ------------------------------------------------------------ figures
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
for a, seg in zip(axs.flat, list(SEGS) + ["stack_bright"]):
    for key, c in (("HE", "C0"), ("HG", "C1"), ("LG0.4-3", "C2")):
        if (seg, key) not in PDSPLOT:
            continue
        f, P, M, I, S, sm = PDSPLOT[(seg, key)]
        fc, Pc, K = logbin(f, P, 1.15)
        nrm = I / S ** 2
        a.loglog(fc, np.clip(Pc - sm["W"], 1e-9, None) * nrm, ".", ms=3, color=c, label=f"{key} (M={M})")
        T = [x for x in all_T if (x.seg.iloc[0] == seg and x.inst.iloc[0] == ("HE" if key == "HE" else f"GECAM {key}"))][0]
        a.loglog(T.f, T.rms3 ** 2 / (2 * T.df), "-", lw=0.8, color=c)
    a.set_title(seg, fontsize=9); a.set_xlabel("f (Hz)", fontsize=8)
    a.legend(fontsize=6)
axs[0, 0].set_ylabel("(P - W) I/S^2  [(rms/mean)^2/Hz]\nline: 3σ QPO threshold", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pds_segments.png", dpi=120)

fig, axs = plt.subplots(2, 4, figsize=(15, 7))
for a, seg in zip(axs.flat, SEGS):
    for inst, c in (("HE", "C0"), ("GECAM HG", "C1"), ("GECAM LG0.4-3", "C2")):
        h = Hall[(Hall.seg == seg) & (Hall.inst == inst)]
        if len(h) == 0:
            continue
        a.errorbar(h.D * 1e3, h.snr, fmt="o-", ms=3, color=c, label=inst)
    if seg == "flare_rise":
        a.plot(old.D * 1e3, old.snr, "x--", color="k", lw=0.8, label="timing.md estimator (440-500)")
    a.axhline(3, color="k", lw=0.5, ls=":"); a.axhline(0, color="k", lw=0.5)
    a.set_xscale("log"); a.set_title(seg, fontsize=9); a.set_xlabel("Haar scale D (ms)", fontsize=8)
    a.legend(fontsize=6)
axs[0, 0].set_ylabel("S(D)/σ")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_haar.png", dpi=120)

# precursor light curve
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
e = np.arange(-5, 15.0001, 0.016)
n = np.histogram(the, e)[0] / 0.016
ax[0].step(e[:-1], n, where="post", lw=0.6, label="HE (dedup, ch>=20), 16 ms")
bb = np.mean([Bsec[s] for s in range(-5, 15)])
ax[0].axhline(bb, color="k", lw=0.5, ls=":")
ax[0].set_ylabel("evt/s"); ax[0].legend(fontsize=7)
m = (bt_t >= -5) & (bt_t < 15)
for gn, c in (("HG", "C1"), ("LG0.4-3", "C2")):
    ax[1].step(bt_t[m], bt_c[gn][m] / 0.05, where="post", lw=0.6, color=c, label=f"GECAM-C BTIME {gn}, 50 ms")
ax[1].set_ylabel("cts/s"); ax[1].set_xlabel("t - T0 (s)"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_precursor.png", dpi=130)
say("done")
