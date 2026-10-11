"""Controls for the HE variability scouting, third round:
  [1] Fermi-GBM continuous TTE (2022-10-09 13z, HEASARC daily, 12 NaI + 2 BGO):
      QPO 3 sigma sensitivity in the same real-event intervals as HE
      (p221009_sci_pds.py), NaI 8-900 keV (4 brightest), BGO 0.2-3 MeV and
      0.2-40 MeV (b0+b1); GBM bad-time intervals 219-277 and 508-514 s avoided
  [2] Lesage+2023 12.3 s candidate: Vaughan (2010) test on GBM (277-508 s,
      the interval of Lesage+2023), GECAM-C (events, 330-508 s) and HE on the
      same sub-intervals, plus HE 288-440 s (before the flare onset)
Source rates of GBM / GECAM-C: per-second regression G = quadratic + S_HE/R on
HE real seconds (as p221009_sci_pds.gecam_S). Light travel between the
spacecraft (< 20 ms) is irrelevant here and not corrected.
Writes out/gbm_*.csv, out/gbm_summary.txt, out/fig_gbm.png
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from p221009_sci_lib import *
from p221009_sci_spec import *

LOG = open(f"{OUT}/gbm_summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


GBM_DIR = f"{BASE}/sci/gbm"
BTI = [(219.0, 277.0), (508.0, 514.0)]
lc = pd.read_csv(LC_V3)
seg1 = pd.read_csv(f"{OUT}/segments_1s.csv")
real = dict(zip(seg1.t.astype(int), seg1.real))

# ---------------------------------------------------------------- HE stream
t_all, box, ch, pw = he_obs()
keep = dedup_cross(t_all, box)
the = t_all[keep]


def rate1s(tt, secs):
    secs = np.asarray(secs, float)
    return (np.searchsorted(tt, secs + 1) - np.searchsorted(tt, secs)).astype(float)


bk = lc[((lc.t >= -50) & (lc.t < -5)) | ((lc.t >= 20) & (lc.t < 170))]
k_bkg = rate1s(the, bk.t.to_numpy()).sum() / bk.B_obs.sum()
Bsec = dict(zip(lc.t.astype(int), k_bkg * lc.B_obs))
he_S = lambda secs: rate1s(the, secs) - np.array([Bsec[int(s)] for s in secs])

# ---------------------------------------------------------------- GBM events
files = sorted(glob.glob(f"{GBM_DIR}/glg_tte_*_221009_13z_v00.fit.gz"))
G = {}
for fn in files:
    det = fn.split("glg_tte_")[1][:2]
    h = fits.open(fn)
    hd = h["EVENTS"].header
    mjdref = hd["MJDREFI"] + hd["MJDREFF"]
    t0 = (Time("2022-10-09T13:17:00", scale="utc").tt.mjd - mjdref) * 86400.0
    ev = h["EVENTS"].data
    tt = ev["TIME"] - t0
    m = (tt > -120) & (tt < 700)
    eb = h["EBOUNDS"].data
    emid = np.sqrt(eb["E_MIN"] * eb["E_MAX"])
    G[det] = dict(t=tt[m], e=emid[ev["PHA"][m]])
    say(f"GBM {det}: {m.sum()} events in -120..700 s; range {tt.min():.1f}..{tt.max():.1f}")


def netcounts(det, lo=440, hi=503):
    t, e = G[det]["t"], G[det]["e"]
    sel = (e >= 8) & (e < 900)
    src = ((t >= lo) & (t < hi) & sel).sum() / (hi - lo)
    b = ((t >= -60) & (t < -5) & sel).sum() / 55.0
    return src - b


nai = [d for d in G if d.startswith("n")]
rank = sorted(nai, key=netcounts, reverse=True)
say("NaI net rate 440-503 s (8-900 keV): " + ", ".join(f"{d} {netcounts(d):.0f}" for d in rank))
top = rank[:4]


def gsel(dets, emin, emax):
    t = np.concatenate([G[d]["t"][(G[d]["e"] >= emin) & (G[d]["e"] < emax)] for d in dets])
    return np.sort(t)


STREAMS = {f"GBM NaI {'+'.join(top)} 8-900 keV": gsel(top, 8, 900),
           "GBM BGO b0+b1 0.2-3 MeV": gsel(["b0", "b1"], 200, 3000),
           "GBM BGO b0+b1 0.2-40 MeV": gsel(["b0", "b1"], 200, 40000)}
ge = np.load(f"{OUT}/gecam_c_evt.npz")
GC = {"GECAM-C HG": np.sort(ge["t"][ge["gain"] == 0]),
      "GECAM-C LG 0.4-3 MeV": np.sort(ge["t"][(ge["gain"] == 1) & (ge["emid"] >= 400) & (ge["emid"] < 3000)])}

REG = {}


def regress(name, tt, win):
    """G(t) = quadratic + S_HE/R over HE real seconds in win, BTI excluded."""
    key = (name, win)
    if key not in REG:
        secs = np.arange(win[0], win[1], 1.0)
        ok = np.array([real.get(int(s), False) and not any(a <= s < b for a, b in BTI) for s in secs])
        g = rate1s(tt, secs); hs = he_S(secs)
        x = (secs - secs.mean()) / (win[1] - win[0])
        X = np.c_[np.ones_like(x), x, x ** 2, hs]
        c, *_ = np.linalg.lstsq(X[ok], g[ok], rcond=None)
        r = g[ok] - X[ok] @ c
        REG[key] = (c, secs.mean(), win[1] - win[0])
        say(f"[REG] {name} {win}: R = S_HE/S = {1/c[3]:.2f}, resid rms {r.std():.0f} (Poisson {np.sqrt(g[ok].mean()):.0f}), n={ok.sum()}")
    return REG[key]


def src_rate(name, tt, win, secs):
    c, xm, xs = regress(name, tt, win)
    return c[3] * he_S(secs)


def bkg_rate(name, tt, win, secs):
    c, xm, xs = regress(name, tt, win)
    x = (secs - xm) / xs
    return c[0] + c[1] * x + c[2] * x ** 2


WINS = [(165.0, 216.0), (280.0, 600.0)]
win_of = lambda lo, hi: [w for w in WINS if w[0] <= lo and hi <= w[1]][0]

# ------------------------------------------------------------- [1] QPO sens
SEGS = {"P0rise": (176.0, 186.0), "decay193": (193.0, 215.0), "tail288": (288.0, 330.0),
        "tail330": (330.0, 440.0), "flare_rise": (440.0, 503.0), "flare_decay": (530.0, 600.0)}
BRIGHT = seg1[(seg1.real) & (seg1.t >= 170) & (seg1.he_net_dt > 5000)].t.to_numpy().astype(float)
FS = (3, 10, 30, 100, 300, 1000)
rows = []
for seg, (lo, hi) in list(SEGS.items()) + [("stack_bright", (None, None))]:
    for nm, tt in list(STREAMS.items()) + [("HE (this run)", the)]:
        if seg == "stack_bright":
            acc = 0
            for s in BRIGHT:
                f, P, M, N, _ = leahy(tt, s, s + 1, 1.0, 2.0 ** -12)
                acc = acc + P
            P = acc / len(BRIGHT); M = len(BRIGHT); secs = BRIGHT
        else:
            f, P, M, N, _ = leahy(tt, lo, hi, 1.0, 2.0 ** -12)
            secs = np.arange(lo, hi - 0.5, 1.0)
        I = rate1s(tt, secs).mean()
        if nm.startswith("HE"):
            S = he_S(secs).mean()
        else:
            S = np.mean(np.concatenate([src_rate(nm, tt, win_of(s, s + 1), np.array([s])) for s in secs]))
        T, W, al = qpo_sens(f, P, M, I, S)
        r = rms_at(T, FS)
        rows.append(dict(seg=seg, stream=nm, M=M, I=I, S=S, W=W, alpha=al, min_pglob=T.pglob.min(),
                         **{f"rms3_{x}Hz": v for x, v in zip(FS, r)}))
        say(f"[1] {seg:12s} {nm:34s} M={M:3d} I={I:8.0f} S={S:8.0f} W={W:.3f} minp={T.pglob.min():.2f}  rms3 " +
            " ".join(f"{100*v:.1f}%" for v in r))
Q = pd.DataFrame(rows)
Q.to_csv(f"{OUT}/gbm_qpo_sens.csv", index=False, float_format="%.5g")

# ------------------------------------------------------------- [2] 12.3 s
tests = []
v3 = lc.set_index("t")


def he_v3(lo, hi):
    s = np.arange(lo, hi)
    return v3.dt.reindex(s).to_numpy(), v3.sig_rec.reindex(s).to_numpy()


def he_obs_lc(lo, hi):
    s = np.arange(lo, hi, 1.0)
    r = rate1s(the, s)
    return r - np.array([Bsec[int(x)] for x in s]), np.sqrt(r)


def other_lc(nm, tt, lo, hi):
    s = np.arange(lo, hi, 1.0)
    r = rate1s(tt, s)
    win = (330.0, 600.0) if nm.startswith("GECAM") else (280.0, 600.0)
    return r - bkg_rate(nm, tt, win, s), np.sqrt(r)


cases = [("HE v3", 277, 508, lambda: he_v3(277, 508)),
         ("HE v3", 330, 508, lambda: he_v3(330, 508)),
         ("HE obs", 288, 440, lambda: he_obs_lc(288, 440)),
         ("HE obs", 330, 440, lambda: he_obs_lc(330, 440))]
for nm, tt in STREAMS.items():
    cases.append((nm, 280, 508, lambda nm=nm, tt=tt: other_lc(nm, tt, 280, 508)))
    cases.append((nm, 330, 508, lambda nm=nm, tt=tt: other_lc(nm, tt, 330, 508)))
    cases.append((nm, 288, 440, lambda nm=nm, tt=tt: other_lc(nm, tt, 288, 440)))
for nm, tt in GC.items():
    cases.append((nm, 330, 508, lambda nm=nm, tt=tt: other_lc(nm, tt, 330, 508)))
    cases.append((nm, 330, 440, lambda nm=nm, tt=tt: other_lc(nm, tt, 330, 440)))
PL = {}
for nm, lo, hi, fn in cases:
    x, sx = fn()
    if np.any(~np.isfinite(x)) or x.mean() <= 0:
        say(f"[2] {nm} {lo}-{hi}: skipped (non-finite or non-positive mean)")
        continue
    res, spec = vaughan_test(x, sx)
    res.update(curve=nm, lo=lo, hi=hi)
    tests.append(res); PL[(nm, lo, hi)] = spec
    say(f"[2] {nm:34s} {lo}-{hi}: mean {res['mean']:8.0f} alpha {res['alpha']:.2f} red/Poisson(f0) {res['S_f0']/res['poisson']:6.0f}  "
        f"max 2I/S {res['Tc']:.2f} @ {1/res['f_best']:.1f} s p={res['p_max']:.3f}; band p={res['p_band']:.3f}; blind p={res['p_blind']:.3f}; "
        f"3sig rms (Q=4) {100*res['rms3_Q4']:.1f}%")
pd.DataFrame(tests).to_csv(f"{OUT}/gbm_qpo12.csv", index=False, float_format="%.5g")

# ------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
a = axs[0]
for nm, c in zip(["HE (this run)"] + list(STREAMS), ["C0", "C1", "C2", "C3"]):
    d = Q[(Q.stream == nm) & (Q.seg == "stack_bright")]
    if len(d):
        a.loglog(FS, [100 * d[f"rms3_{x}Hz"].iloc[0] for x in FS], "o-", color=c, label=nm)
a.set_xlabel("f (Hz)"); a.set_ylabel("3σ QPO rms (%), Q≈4.5"); a.set_title("191 bright real seconds stacked"); a.legend(fontsize=7)
a = axs[1]
for (nm, lo, hi), (f, I, S) in PL.items():
    if (lo, hi) == (330, 508):
        a.loglog(f, I / S, "-", lw=0.7, label=f"{nm}")
a.axvspan(F_LO, F_HI, color="orange", alpha=0.3)
a.set_xlabel("f (Hz)"); a.set_ylabel("periodogram / red-noise model"); a.set_title("330-508 s, 12.3 s band shaded"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gbm.png", dpi=130)
say("done")
