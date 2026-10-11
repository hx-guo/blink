"""Step 2 of the HE variability scouting: instrumental signals in the HE
event stream, measured on source-free / smooth intervals.

  [1] pair correlation g(u) of recorded events, per box and across boxes:
      the dead-time hole (direct tau), and the correlated-event excess that
      produces the 2-8 ms Haar excess of timing.md
  [2] Leahy PDS of background intervals (1/8192 s bins), compared with the
      PDS predicted from g(u) and with the dead-time formula
  [3] 2^-20 s PDS of the quiet interval (dead-time shape up to 500 kHz)
  [4] narrow-line search 0.06-2048 Hz, Z^2_4 at 1 Hz (SEC / 1 Hz frame) and
      at 1/1.048576 s (ptime wrap)
  [5] packet structure (inject npz, 287-440 s): dt across packet boundaries
  [6] GECAM-C pre-burst null PDS; writes out/gecam_c_evt.npz for later steps
Writes out/null_*.csv, out/null_summary.txt, out/fig_null.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from p221009_sci_lib import *

BKG = {"pre": (-50.0, -5.0), "quiet": (20.0, 170.0), "post": (620.0, 750.0)}
LOG = open(f"{OUT}/null_summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


t, box, ch, pw = he_obs()
say(f"HE observed events (ch>=20, no SEC): {len(t)}")
tb = [t[box == b] for b in range(3)]

# ---------------------------------------------------------------- [1] g(u)
rows = []
G = {}
for seg, (lo, hi) in list(BKG.items()) + [("flare440", (440.0, 500.0))]:
    rates = [((x >= lo) & (x < hi)).sum() / (hi - lo) for x in tb]
    pairs = [("A", 0, None), ("B", 1, None), ("C", 2, None), ("AB", 0, 1), ("AC", 0, 2), ("BC", 1, 2)]
    for nm, i, j in pairs:
        lag, g, num, den = pair_g(tb[i], lo, hi, cross=None if j is None else tb[j], maxlag=0.1)
        if j is not None:      # symmetrise cross lags: pairs at +u and -u
            K = len(lag) // 2
            w50 = np.abs(lag) < 50e-6
            gs = num[w50] / den[w50]
            say(f"[1] {seg:8s} cross {nm}: signed-lag g within +-50 us, top bins: "
                + ", ".join(f"{lag[w50][q]*1e6:+.1f}us:{gs[q]:.2f}" for q in np.argsort(gs)[::-1][:4]))
            nn = num[K:] + np.r_[0, num[:K][::-1]]
            dd = den[K:] + np.r_[0, den[:K][::-1]]
            lag, g, num, den = lag[K:], nn / dd, nn, dd
        r = lograte(lag, num, den, hi=0.1)
        G[(seg, nm)] = r
        for c, gg, ss in r:
            rows.append(dict(seg=seg, pair=nm, lag=c, g=gg, sig=ss))
        if j is None:
            lam = rates[i]
            # dead-time hole: integral of (1-g) over 0..12 us, in us
            pos = lag > 0
            dl = lag[1] - lag[0]
            hole = np.nansum(1 - g[pos & (lag < 12e-6)]) * dl
            # correlated excess: lambda * integral (g-1) over 12 us .. 20 ms
            mm = pos & (lag >= 12e-6) & (lag < 0.02)
            exc = lam * np.nansum(g[mm] - 1) * dl
            mm2 = pos & (lag >= 12e-6) & (lag < 1e-3)
            exc1 = lam * np.nansum(g[mm2] - 1) * dl
            say(f"[1] {seg:8s} box {nm}: rate {lam:8.1f}/s  tau_hole = {hole*1e6:5.2f} us  "
                f"excess pairs/event 12us-1ms {exc1:+.4f}, 12us-20ms {exc:+.4f}")
        else:
            m0 = (lag >= 0) & (lag < 4e-6)
            m1 = (lag >= 4e-6) & (lag < 1e-3)
            lam = rates[j]
            dl = lag[1] - lag[0]
            say(f"[1] {seg:8s} cross {nm}: g(|u|<4us) = {num[m0].sum()/den[m0].sum():.2f}  "
                f"coincident pairs/event(|u|<4us) {lam*np.nansum(g[m0]-1)*dl:+.5f}, 4us-1ms {lam*np.nansum(g[m1]-1)*dl:+.5f}")
pd.DataFrame(rows).to_csv(f"{OUT}/null_pair_g.csv", index=False, float_format="%.6g")

# ------------------------------------------------- [2] PDS of bkg intervals
def g_total(lo, hi):
    lag, g, num, den = pair_g(t, lo, hi, maxlag=0.05)
    K = len(lag) // 2
    return lag[K + 1:], g[K + 1:]


def pds_from_g(f, lag, g, lam):
    dl = lag[1] - lag[0]
    gg = np.nan_to_num(g - 1)
    # 2 [1 + 2 lam int (g-1) cos(2 pi f u) du]; chunked to limit memory
    out = np.empty(len(f))
    for i0 in range(0, len(f), 64):
        ff = f[i0:i0 + 64, None]
        out[i0:i0 + 64] = 2 * (1 + 2 * lam * (gg[None, :] * np.cos(2 * np.pi * ff * lag[None, :])).sum(1) * dl)
    return out


prow = []
PDS = {}
for seg, (lo, hi) in BKG.items():
    f, P, M, N, _ = leahy(t, lo, hi, 1.0, 2.0 ** -13)
    lam_b = np.array([((x >= lo) & (x < hi)).sum() / (hi - lo) for x in tb])
    pred_dt = 2 - 4 * TAU_TRIG * (lam_b ** 2).sum() / lam_b.sum()
    lag, g = g_total(lo, hi)
    fc, Pc, K = logbin(f, P, 1.15)
    Pg = pds_from_g(fc, lag, g, lam_b.sum())
    PDS[seg] = (fc, Pc, K, M, Pg, pred_dt)
    hf = (f > 1000) & (f < 4096)
    lf = (f > 2) & (f < 30)
    mf = (f > 60) & (f < 500)
    say(f"[2] {seg:6s} M={M} rate {N:.0f}/s  Leahy: 2-30 Hz {P[lf].mean():.4f}, 60-500 Hz {P[mf].mean():.4f}, "
        f"1-4 kHz {P[hf].mean():.4f} (+-{P[hf].std()/np.sqrt(hf.sum()):.4f});  "
        f"dead-time-only pred {pred_dt:.4f};  from g(u): 2-30 Hz {pds_from_g(np.array([10.0]), lag, g, lam_b.sum())[0]:.4f}, "
        f"200 Hz {pds_from_g(np.array([200.0]), lag, g, lam_b.sum())[0]:.4f}, 2 kHz {pds_from_g(np.array([2000.0]), lag, g, lam_b.sum())[0]:.4f}")
    for a, b, c, d in zip(fc, Pc, K, Pg):
        prow.append(dict(seg=seg, f=a, P=b, nbin=c, M=M, P_from_g=d, P_deadtime=pred_dt))
pd.DataFrame(prow).to_csv(f"{OUT}/null_pds.csv", index=False, float_format="%.6g")

# ---------------------------------------------- [3] 2^-20 s PDS, quiet bkg
lo, hi = BKG["quiet"]
fu, Pu, Mu, Nu, _ = leahy(t, lo, hi, 1.0, 2.0 ** -20)
fuc, Puc, Ku = logbin(fu, Pu, 1.2, fmin=10.0)
lam_b = np.array([((x >= lo) & (x < hi)).sum() / (hi - lo) for x in tb])


def renewal_thinned(f, lam_rec, tau, p_rec):
    """Leahy PDS of a thinned non-paralysable renewal process (per box)."""
    lam_t = lam_rec / p_rec
    r0 = lam_t / (1 - lam_t * tau)
    w = 2 * np.pi * f
    phi = np.exp(1j * w * tau) * r0 / (r0 - 1j * w)
    s = np.real((1 + phi) / (1 - phi))
    return 2 * (1 + p_rec * (s - 1))


Pren = sum(l * renewal_thinned(fuc, l, TAU_TRIG, 1 / 1.5) for l in lam_b) / lam_b.sum()
pd.DataFrame(dict(f=fuc, P=Puc, nbin=Ku, P_renewal=Pren)).to_csv(f"{OUT}/null_pds_us.csv", index=False, float_format="%.6g")
for fb in (100, 1e3, 1e4, 5e4, 1e5, 1.6e5, 2.5e5, 3.3e5, 4.5e5):
    k = np.argmin(abs(fuc - fb))
    say(f"[3] quiet 2^-20 s PDS at {fuc[k]:9.0f} Hz: {Puc[k]:.4f}  (renewal tau=3us, p=1/1.5: {Pren[k]:.4f})")

# ------------------------------------------------ [4] lines, Z^2 at 1 Hz
lrow = []
for seg, (lo, hi) in BKG.items():
    sl = 16.0 if hi - lo > 100 else 4.0
    f, P, M, N, _ = leahy(t, lo, hi, sl, 2.0 ** -12)
    Pn = P / np.median(P) * stats.chi2.median(2 * M) / (2 * M) * 2   # normalise to local white level
    x = Pn / 2
    pr = gamma_sf(x, 2 * M)
    ntr = len(f)
    top = np.argsort(pr)[:6]
    say(f"[4] {seg:6s} seg {sl:.0f} s x {M}: {ntr} freqs; white level (median-based) {np.median(P)/(stats.chi2.median(2*M)/(2*M)):.4f}")
    for k in top:
        say(f"      f = {f[k]:9.4f} Hz  P/Pw = {x[k]:.3f}  p1 = {pr[k]:.2e}  p_global = {min(1, pr[k]*ntr):.3f}")
        lrow.append(dict(seg=seg, f=f[k], ratio=x[k], p1=pr[k], pglob=min(1, pr[k] * ntr)))
    for fh in (1, 2, 3, 4, 5, 10):
        k = np.argmin(abs(f - fh))
        say(f"      harmonic {fh:2d} Hz: P/Pw = {x[k]:.3f} p1 = {pr[k]:.3f}")
    w = (t >= lo) & (t < hi)
    for nm, per in (("1 Hz (SEC / frame)", 1.0), ("ptime wrap 1.048576 s", 1.048576)):
        ph = 2 * np.pi * (t[w] / per)
        Z = 2 / w.sum() * sum(np.cos(k * ph).sum() ** 2 + np.sin(k * ph).sum() ** 2 for k in range(1, 5))
        say(f"      Z^2_4 at {nm}: {Z:.2f}  (chi2_8 p = {stats.chi2.sf(Z, 8):.3f})")
    # 1 s folded profile relative to the SEC marker (integer seconds)
    hph = np.histogram(t[w] % 1.0, 50, (0, 1))[0]
    say(f"      1-s fold (50 bins): chi2/dof = {((hph-hph.mean())**2/hph.mean()).sum()/49:.2f}; "
        f"first/last bins rel. {hph[0]/hph.mean():.3f} {hph[-1]/hph.mean():.3f}")
pd.DataFrame(lrow).to_csv(f"{OUT}/null_lines.csv", index=False, float_format="%.6g")

# ------------------------------------------------------- [5] packet test
p = np.load(EVT_PKT)
for bi, b in enumerate("abc"):
    tt = p[f"t_{b}"] - T0_MET; pk = p[f"pkt_{b}"]; sc = p[f"sec_{b}"].astype(bool)
    w = (tt >= 287) & (tt < 440) & ~sc
    tt, pk = tt[w], pk[w]
    o = np.argsort(tt, kind="stable"); tt, pk = tt[o], pk[o]
    d = np.diff(tt)
    bnd = np.diff(pk) != 0
    # local mean interval: mean over +-200 events
    cs = np.r_[0, np.cumsum(d)]
    i = np.arange(len(d)); a = np.clip(i - 200, 0, len(d)); c = np.clip(i + 200, 0, len(d))
    loc = (cs[c] - cs[a]) / (c - a)
    x = d / loc
    ks = stats.ks_2samp(x[bnd], x[~bnd])
    u, n = np.unique(pk, return_counts=True)
    nrec = ((t >= 287) & (t < 440) & (box == bi)).sum()
    say(f"[5] box {b.upper()}: {w.sum()} events ({nrec} in rec cache, ch>=20), {len(u)} packets, "
        f"events/packet 109:{(n==109).mean():.3f} 108:{(n==108).mean():.3f} other:{((n!=109)&(n!=108)).mean():.4f}; "
        f"<dt/local> boundary {x[bnd].mean():.4f}+-{x[bnd].std()/np.sqrt(bnd.sum()):.4f}, inside {x[~bnd].mean():.4f}; "
        f"KS p = {ks.pvalue:.3f}; boundary dt<3us {(d[bnd]<3e-6).mean():.4f} inside {(d[~bnd]<3e-6).mean():.4f}")
    # does a 108-event packet hide a missing event? dt across its leading boundary
    n_of = dict(zip(u, n))
    pk_next = pk[1:][bnd]
    short = np.array([n_of[q] == 108 for q in pk_next])
    if short.any():
        say(f"      boundary into 108-event packets: <dt/local> {x[bnd][short].mean():.4f} (n={short.sum()}), "
            f"into 109: {x[bnd][~short].mean():.4f}")

# ------------------------------------------------------ [6] GECAM-C null
ge = gecam_events(-300.0, 700.0)
np.savez(f"{OUT}/gecam_c_evt.npz", **ge)
gti = gecam_gti()
say(f"[6] GECAM-C events -300..700: {len(ge['t'])}; GTI rel T0: {np.round(gti, 2).tolist()}")
lo, hi = -268.0, -68.0
sel = {"all": np.ones(len(ge["t"]), bool),
       "HG": ge["gain"] == 0,
       "LG 0.4-3 MeV": (ge["gain"] == 1) & (ge["emid"] >= 400) & (ge["emid"] < 3000)}
grow = []
for nm, m in sel.items():
    tg = ge["t"][m]
    f, P, M, N, _ = leahy(tg, lo, hi, 1.0, 2.0 ** -12)
    fc, Pc, K = logbin(f, P, 1.15)
    lf = (f > 2) & (f < 30); hf = (f > 500)
    w = (tg >= lo) & (tg < hi)
    ph = 2 * np.pi * tg[w]
    Z = 2 / w.sum() * sum(np.cos(k * ph).sum() ** 2 + np.sin(k * ph).sum() ** 2 for k in range(1, 5))
    say(f"[6] GECAM-C pre-burst {nm:13s}: rate {N:.0f}/s, Leahy 2-30 Hz {P[lf].mean():.4f}, >500 Hz {P[hf].mean():.4f}; "
        f"Z^2_4(1 Hz) = {Z:.1f}")
    for a, b, c in zip(fc, Pc, K):
        grow.append(dict(sel=nm, f=a, P=b, nbin=c, M=M))
pd.DataFrame(grow).to_csv(f"{OUT}/null_gecam_pds.csv", index=False, float_format="%.6g")

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]
for nm, c in (("A", "C0"), ("B", "C1"), ("C", "C2"), ("AB", "C3"), ("AC", "C4"), ("BC", "C5")):
    r = G[("quiet", nm)]
    a.errorbar(r[:, 0] * 1e6, r[:, 1], r[:, 2], fmt=".-", ms=3, lw=0.7, color=c, label=("box " if len(nm) == 1 else "cross ") + nm)
a.set_xscale("log"); a.axhline(1, color="k", lw=0.5); a.set_ylim(0, 3)
a.set_xlabel("lag u (us)"); a.set_ylabel("g(u)"); a.set_title("pair correlation, quiet T0+20..170 s"); a.legend(fontsize=7)
a = ax[0, 1]
for seg, c in zip(BKG, ("C0", "C1", "C2")):
    fc, Pc, K, M, Pg, pdt = PDS[seg]
    a.plot(fc, Pc, ".", ms=3, color=c, label=f"{seg} (M={M})")
    a.plot(fc, Pg, "-", lw=0.8, color=c)
    a.axhline(pdt, ls=":", color=c, lw=0.8)
a.axhline(2, color="k", lw=0.5); a.set_xscale("log")
a.set_xlabel("f (Hz)"); a.set_ylabel("Leahy"); a.set_title("HE background PDS (line: from g(u); dotted: dead time only)"); a.legend(fontsize=7)
a = ax[1, 0]
a.plot(fuc, Puc, ".", ms=3, label="quiet, 2^-20 s bins"); a.plot(fuc, Pren, "-", lw=0.8, label="renewal, tau=3 us")
a.axhline(2, color="k", lw=0.5); a.set_xscale("log"); a.set_xlabel("f (Hz)"); a.set_ylabel("Leahy"); a.legend(fontsize=7)
a = ax[1, 1]
for nm in sel:
    d = pd.DataFrame(grow); d = d[d.sel == nm]
    a.plot(d.f, d.P, ".", ms=3, label=f"GECAM-C {nm}")
a.axhline(2, color="k", lw=0.5); a.set_xscale("log"); a.set_xlabel("f (Hz)"); a.set_ylabel("Leahy")
a.set_title("GECAM-C pre-burst (-268..-68 s)"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_null.png", dpi=130)
say("done")
