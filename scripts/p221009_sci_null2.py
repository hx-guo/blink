"""Step 2b: rerun of the summed-stream parts of p221009_sci_null.py.

p221009_sci_null.py [2]-[4] ran on an event list that was ordered box by
box, so its PDS saw one box only (the per-box results there remain valid:
g(u), the packet test and GECAM-C). Here, on the time-sorted list:
  [2] Leahy PDS of the 3-box sum, of each box, and of the sum with cross-box
      coincidences removed (|dt| <= 2.1 us), background intervals
  [3] 2^-20 s PDS of the quiet interval (sum and box A)
  [4] narrow-line search on the sum, Z^2_4 at 1 Hz and 1/1.048576 s
  [7] Haar noise ratio var(d)/<N> vs scale (detrended), sum / dedup / box,
      i.e. what timing.md called the 2-8 ms instrumental Haar excess
Writes out/null2_*.csv, out/null2_summary.txt, out/fig_null.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from p221009_sci_lib import *

BKG = {"pre": (-50.0, -5.0), "quiet": (20.0, 170.0), "post": (620.0, 750.0)}
LOG = open(f"{OUT}/null2_summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


t, box, ch, pw = he_obs()
keep = dedup_cross(t, box)
say(f"HE observed (ch>=20, no SEC, time-sorted): {len(t)}; cross-box coincidences removed: {(~keep).sum()} "
    f"({(~keep).mean()*100:.2f}%)")
streams = {"sum": t, "dedup": t[keep], "boxA": t[box == 0], "boxB": t[box == 1], "boxC": t[box == 2]}
tb = [t[box == b] for b in range(3)]


def pred_deadtime(lo, hi):
    lam = np.array([((x >= lo) & (x < hi)).sum() / (hi - lo) for x in tb])
    return 2 - 4 * TAU_TRIG * (lam ** 2).sum() / lam.sum(), lam


rows = []
PDS = {}
for seg, (lo, hi) in BKG.items():
    pdt, lam = pred_deadtime(lo, hi)
    for nm, tt in streams.items():
        f, P, M, N, _ = leahy(tt, lo, hi, 1.0, 2.0 ** -13)
        fc, Pc, K = logbin(f, P, 1.15)
        PDS[(seg, nm)] = (fc, Pc)
        lf = (f > 2) & (f < 30); mf = (f > 60) & (f < 500); hf = (f > 1000)
        e = P[hf].std() / np.sqrt(hf.sum())
        say(f"[2] {seg:6s} {nm:6s} rate {N:7.0f}/s  Leahy 2-30 Hz {P[lf].mean():.4f}  60-500 Hz {P[mf].mean():.4f}  "
            f"1-4 kHz {P[hf].mean():.4f} (+-{e:.4f})")
        for a, b, c in zip(fc, Pc, K):
            rows.append(dict(seg=seg, stream=nm, f=a, P=b, nbin=c, M=M))
    say(f"      dead-time-only prediction (per-box, tau=3us): {pdt:.4f}; box rates {np.round(lam)}")
pd.DataFrame(rows).to_csv(f"{OUT}/null2_pds.csv", index=False, float_format="%.6g")


def renewal_thinned(f, lam_rec, tau, p_rec):
    lam_t = lam_rec / p_rec
    r0 = lam_t / (1 - lam_t * tau)
    w = 2 * np.pi * f
    phi = np.exp(1j * w * tau) * r0 / (r0 - 1j * w)
    return 2 * (1 + p_rec * (np.real((1 + phi) / (1 - phi)) - 1))


lo, hi = BKG["quiet"]
pdt, lam = pred_deadtime(lo, hi)
US = {}
urow = []
for nm in ("sum", "boxA"):
    fu, Pu, Mu, Nu, _ = leahy(streams[nm], lo, hi, 1.0, 2.0 ** -20)
    fuc, Puc, Ku = logbin(fu, Pu, 1.2, fmin=10.0)
    l = lam[0] if nm == "boxA" else None
    Pren = renewal_thinned(fuc, lam[0], TAU_TRIG, 1 / 1.5) if nm == "boxA" else \
        sum(x * renewal_thinned(fuc, x, TAU_TRIG, 1 / 1.5) for x in lam) / lam.sum()
    US[nm] = (fuc, Puc, Pren)
    for a, b, c, d in zip(fuc, Puc, Ku, Pren):
        urow.append(dict(stream=nm, f=a, P=b, nbin=c, P_renewal_nocoinc=d))
    for fb in (100, 1e3, 1e4, 5e4, 1e5, 1.7e5, 2.5e5, 3.5e5, 4.5e5):
        k = np.argmin(abs(fuc - fb))
        say(f"[3] quiet {nm:5s} 2^-20 s PDS at {fuc[k]:8.0f} Hz: {Puc[k]:.4f}  (renewal tau=3us p=2/3, no coincidences: {Pren[k]:.4f})")
pd.DataFrame(urow).to_csv(f"{OUT}/null2_pds_us.csv", index=False, float_format="%.6g")

lrow = []
for seg, (lo, hi) in BKG.items():
    sl = 16.0 if hi - lo > 100 else 4.0
    f, P, M, N, _ = leahy(t, lo, hi, sl, 2.0 ** -12)
    Pw = np.median(P) / (stats.chi2.median(2 * M) / (2 * M))
    x = P / Pw
    pr = gamma_sf(x, 2 * M)
    ntr = len(f)
    say(f"[4] {seg:6s} sum, seg {sl:.0f} s x {M}: {ntr} freqs, white level {Pw:.4f}")
    for k in np.argsort(pr)[:5]:
        say(f"      f = {f[k]:9.4f} Hz  P/Pw = {x[k]:.3f}  p1 = {pr[k]:.2e}  p_global = {min(1, pr[k]*ntr):.3f}")
        lrow.append(dict(seg=seg, f=f[k], ratio=x[k], p1=pr[k], pglob=min(1, pr[k] * ntr)))
    say("      1..10 Hz harmonics P/Pw: " + " ".join(f"{x[np.argmin(abs(f-h))]:.2f}" for h in range(1, 11)))
    w = (t >= lo) & (t < hi)
    for nm, per in (("1 Hz", 1.0), ("1/1.048576 s", 1.048576)):
        ph = 2 * np.pi * (t[w] / per)
        Z = 2 / w.sum() * sum(np.cos(k * ph).sum() ** 2 + np.sin(k * ph).sum() ** 2 for k in range(1, 5))
        say(f"      Z^2_4 at {nm}: {Z:.2f} (p = {stats.chi2.sf(Z, 8):.3f})")
pd.DataFrame(lrow).to_csv(f"{OUT}/null2_lines.csv", index=False, float_format="%.6g")

# [7] Haar noise ratio in the quiet interval (no source)
SC = 0.001 * 2.0 ** np.arange(0, 10)
hrow = []
lo, hi = BKG["quiet"]
for nm in ("sum", "dedup", "boxA"):
    tt = streams[nm]
    r = []
    for D in SC:
        nb = int(np.floor((hi - lo) / (2 * D))) * 2
        n = np.histogram(tt, lo + D * np.arange(nb + 1))[0].astype(float)
        d = n[1::2] - n[0::2]
        ratio = d.var() / (n[0::2] + n[1::2]).mean()
        err = np.sqrt(2.0 / len(d)) * ratio
        r.append(ratio)
        hrow.append(dict(stream=nm, D=D, ratio=ratio, err=err, snr_vs_poisson=(ratio - 1) / err))
    say(f"[7] quiet Haar var(d)/<N_l+N_r> {nm:5s}: " + " ".join(f"{D*1e3:.0f}ms:{x:.3f}" for D, x in zip(SC, r)))
H = pd.DataFrame(hrow)
H.to_csv(f"{OUT}/null2_haar.csv", index=False, float_format="%.6g")

# figure: g(u) (from null_pair_g.csv), bkg PDS, us PDS, Haar ratio
G = pd.read_csv(f"{OUT}/null_pair_g.csv")
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]
for nm, c in (("A", "C0"), ("B", "C1"), ("C", "C2"), ("AB", "C3"), ("AC", "C4"), ("BC", "C5")):
    r = G[(G.seg == "quiet") & (G.pair == nm)]
    a.errorbar(r.lag * 1e6, r.g, r.sig, fmt=".-", ms=3, lw=0.7, color=c,
               label=("box " if len(nm) == 1 else "cross ") + nm)
a.set_xscale("log"); a.set_yscale("log"); a.axhline(1, color="k", lw=0.5); a.set_ylim(0.05, 30)
a.set_xlabel("|lag| (us)"); a.set_ylabel("g(u)"); a.set_title("pair correlation, T0+20..170 s"); a.legend(fontsize=7)
a = ax[0, 1]
for nm, c in (("sum", "C0"), ("dedup", "C1"), ("boxA", "C2")):
    fc, Pc = PDS[("quiet", nm)]
    a.plot(fc, Pc, ".-", ms=3, lw=0.5, color=c, label=nm)
fc, Pc = PDS[("post", "sum")]
a.plot(fc, Pc, ".", ms=3, color="C3", label="sum, 620-750 s")
a.axhline(2, color="k", lw=0.5); a.axhline(pdt, color="C2", ls=":", lw=0.8)
a.set_xscale("log"); a.set_xlabel("f (Hz)"); a.set_ylabel("Leahy power"); a.set_title("HE background PDS"); a.legend(fontsize=7)
a = ax[1, 0]
for nm, c in (("sum", "C0"), ("boxA", "C2")):
    fuc, Puc, Pren = US[nm]
    a.plot(fuc, Puc, ".", ms=3, color=c, label=f"{nm}, 2^-20 s bins")
    a.plot(fuc, Pren, "-", lw=0.8, color=c, label=f"{nm}: dead time only (tau=3 us)")
a.axhline(2, color="k", lw=0.5); a.set_xscale("log"); a.set_xlabel("f (Hz)"); a.set_ylabel("Leahy power")
a.set_title("quiet interval up to 500 kHz"); a.legend(fontsize=7)
a = ax[1, 1]
for nm, c in (("sum", "C0"), ("dedup", "C1"), ("boxA", "C2")):
    h = H[H.stream == nm]
    a.errorbar(h.D * 1e3, h.ratio, h.err, fmt="o-", ms=3, color=c, label=nm)
a.axhline(1, color="k", lw=0.5); a.set_xscale("log"); a.set_xlabel("Haar scale D (ms)")
a.set_ylabel("var(d) / <N_l+N_r>"); a.set_title("Haar noise ratio, no source"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_null.png", dpi=130)
say("done")
