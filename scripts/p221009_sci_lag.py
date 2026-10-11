"""(B) MeV spectral lags tau(E) in the real-event intervals, per pulse, and
their precision; plus the packet-frequency null test asked for by novelty.

HE bands (observed events, dedup stream, low-gain channel bands of eband.md):
  NaI   pw 54-70, ch >= 20   (~0.07-1 MeV)
  C1    CsI pw > 75, ch 30-53   (~0.2-0.5 MeV)  <- reference
  C2    CsI ch 53-99   (~0.5-1 MeV)
  C3    CsI ch 99-224  (~1-2.5 MeV)
GECAM-C (events, T0+330-678 s only): HG 30-100 keV, HG 100-350 keV,
LG 0.4-1 MeV, LG 1-3 MeV, all 12 GRDs, light-travel corrected.
Lag tau = t_band - t_ref (positive: band arrives later). Two estimators:
  CCF  : 1 ms series, CCF smoothed by a triangle of half-width Delta
         (= binning at Delta), parabola over +-150 ms symmetric about the
         maximum; precision from the spread of the 3 HE boxes (std/sqrt 3)
  xspec: frequency-averaged lag from the cross spectrum (0.1-1, 1-4 Hz,
         10 s segments, 1/64 s bins) with the coherence-based phase error
         sqrt((1-g2)/(2 g2 N)) (Bendat & Piersol; Nowak+1999), analytic
(An event-parity split of one box is NOT an independent sub-sample: the two
halves are locked to n_even = n_odd +-1 in every bin, giving a lag-0 spike.)
Systematics:
  [N2] same band, box A against boxes B+C: inter-box timing, must be 0
  [N3] rate-dependent migration: a fraction eps * r(t)/r_max of the C1
       events moved to C3 (pile-up / baseline shift in proportion to rate),
       eps = 0.006 (pile-up indicator bound in real seconds) and 0.03
[P]  packet frequency: per-box quiet PDS at f_pkt = lambda_box/109 and
     harmonics; same in the 330-440 s real interval
Writes out/lag_*.csv, out/lag_summary.txt, out/fig_lag.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from p221009_sci_lib import *

LOG = open(f"{OUT}/lag_summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


PULSES = {
    "precursor -1..9": (-1.0, 9.0),
    "P0 rise 176-186": (176.0, 186.0),
    "192.6-215.4": (192.6, 215.4),
    "287.3-330": (287.3, 330.0),
    "330-440": (330.0, 440.0),
    "flare 440-480": (440.0, 480.0),
    "flare 480-503": (480.0, 503.0),
    "flare 530-600": (530.0, 600.0),
}
MAXLAG = 1.0

t_all, box, ch, pw = he_obs()
keep = dedup_cross(t_all, box)
t, bx, ch, pw = t_all[keep], box[keep], ch[keep], pw[keep]
BANDS = {"NaI": (pw >= 54) & (pw <= 70),
         "C1": (pw > 75) & (ch >= 30) & (ch < 53),
         "C2": (pw > 75) & (ch >= 53) & (ch < 99),
         "C3": (pw > 75) & (ch >= 99) & (ch < 224)}
ge = np.load(f"{OUT}/gecam_c_evt.npz")
GB = {"G_HG30-100": (ge["gain"] == 0) & (ge["emid"] >= 30) & (ge["emid"] < 100),
      "G_HG100-350": (ge["gain"] == 0) & (ge["emid"] >= 100) & (ge["emid"] < 350),
      "G_LG0.4-1": (ge["gain"] == 1) & (ge["emid"] >= 400) & (ge["emid"] < 1000),
      "G_LG1-3": (ge["gain"] == 1) & (ge["emid"] >= 1000) & (ge["emid"] < 3000)}


def series(tt, lo, hi):
    n = int(round((hi - lo) / 1e-3))
    i0, i1 = np.searchsorted(tt, [lo, hi])
    idx = ((tt[i0:i1] - lo) / 1e-3).astype(np.int64)
    return np.bincount(idx[(idx >= 0) & (idx < n)], minlength=n).astype(float)


def ccf_lag(a, b, delta_ms, halfwin_ms=150):
    """lag (s) of b relative to a; positive = b later. CCF smoothed at Delta;
    parabola over a window of +-halfwin symmetric about the smoothed maximum."""
    n = len(a)
    a = a - a.mean(); b = b - b.mean()
    L = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, L)) * np.fft.rfft(b, L), L)
    K = int(MAXLAG / 1e-3)
    cc = np.r_[c[L - K:], c[:K + 1]] / np.sqrt((a ** 2).sum() * (b ** 2).sum())
    lags = np.arange(-K, K + 1) * 1e-3
    w = np.r_[np.arange(1, delta_ms + 1), np.arange(delta_ms - 1, 0, -1)].astype(float)
    cs = np.convolve(cc, w / w.sum(), mode="same")
    core = slice(K - 500, K + 501)                      # peak searched within +-0.5 s
    j = np.argmax(cs[core]) + K - 500
    x = lags[j - halfwin_ms:j + halfwin_ms + 1]; y = cs[j - halfwin_ms:j + halfwin_ms + 1]
    A = np.polyfit(x, y, 2)
    peak = -A[1] / (2 * A[0]) if A[0] < 0 else lags[j]
    if not (x[0] <= peak <= x[-1]):
        peak = lags[j]
    return peak, cs[j]


def xspec_lag(ta, tb, lo, hi, seglen, fband, dt=1 / 64):
    """Frequency-averaged time lag from the cross spectrum (b relative to a,
    positive = b later), with the coherence-based phase error
    sigma_phi = sqrt((1-g2)/(2 g2 N)) (Bendat & Piersol; Nowak+1999)."""
    M = int(np.floor((hi - lo) / seglen + 1e-9))
    nb = int(round(seglen / dt))
    f = np.fft.rfftfreq(nb, dt)
    m = (f >= fband[0]) & (f < fband[1])
    C = 0; Pa = 0; Pb = 0
    for s in range(M):
        e = lo + s * seglen + dt * np.arange(nb + 1)
        A = np.fft.rfft(np.histogram(ta, e)[0].astype(float))
        B = np.fft.rfft(np.histogram(tb, e)[0].astype(float))
        C = C + (np.conj(A) * B)[m]; Pa = Pa + (np.abs(A) ** 2)[m]; Pb = Pb + (np.abs(B) ** 2)[m]
    Cs, Pas, Pbs = C.sum(), Pa.sum(), Pb.sum()
    N = M * m.sum()
    g2 = np.abs(Cs) ** 2 / (Pas * Pbs)
    phi = np.angle(Cs)
    fm = (f[m] * np.abs(C)).sum() / np.abs(C).sum()
    sphi = np.sqrt((1 - g2) / (2 * g2 * N))
    return phi / (2 * np.pi * fm), sphi / (2 * np.pi * fm), g2, fm


def sel_t(tt, msk, lo, hi):
    x = tt[msk]
    return x[np.searchsorted(x, lo - 1):np.searchsorted(x, hi + 1)]


box_subs = [bx == b for b in range(3)]
rows = []
for pn, (lo, hi) in PULSES.items():
    w = (t >= lo) & (t < hi)
    nb = {k: int((v & w).sum()) for k, v in BANDS.items()}
    seglen = min(10.0, hi - lo)
    pairs = [("C1", k, BANDS["C1"], BANDS[k], t, t) for k in ("NaI", "C2", "C3")]
    pairs += [("C1 boxA", "C1 boxB+C [N2]", BANDS["C1"] & (bx == 0), BANDS["C1"] & (bx != 0), t, t),
              ("C1 boxA", "C1 boxB [N2]", BANDS["C1"] & (bx == 0), BANDS["C1"] & (bx == 1), t, t),
              ("C1 boxA", "C1 boxC [N2]", BANDS["C1"] & (bx == 0), BANDS["C1"] & (bx == 2), t, t),
              ("C1 boxB", "C1 boxC [N2]", BANDS["C1"] & (bx == 1), BANDS["C1"] & (bx == 2), t, t),
              ("all boxA", "all boxB+C [N2]", bx == 0, bx != 0, t, t),
              ("C3 boxA", "C3 boxB+C [N2]", BANDS["C3"] & (bx == 0), BANDS["C3"] & (bx != 0), t, t)]
    if lo >= 330 and hi <= 678:
        pairs += [("C1", gn, BANDS["C1"], gm, t, ge["t"]) for gn, gm in GB.items()]
    for ref, bn, ms_a, ms_b, ta_all, tb_all in pairs:
        ta = sel_t(ta_all, ms_a, lo, hi); tb = sel_t(tb_all, ms_b, lo, hi)
        row = dict(pulse=pn, ref=ref, band=bn, n_ref=int(((ta >= lo) & (ta < hi)).sum()), n_band=int(((tb >= lo) & (tb < hi)).sum()))
        for delta in (16, 64):
            L, cm = ccf_lag(series(ta, lo, hi), series(tb, lo, hi), delta)
            # box spread (HE-HE pairs with both streams split by box; HE-GECAM: HE split by box)
            if tb_all is t and "[N2]" not in bn:
                subs = [ccf_lag(series(sel_t(t, ms_a & q, lo, hi), lo, hi), series(sel_t(t, ms_b & q, lo, hi), lo, hi), delta)[0] for q in box_subs]
                es = np.std(subs, ddof=1) / np.sqrt(3)
            else:
                es = np.nan
            row[f"ccf{delta}_ms"] = L * 1e3; row[f"ccf{delta}_boxerr_ms"] = es * 1e3; row[f"ccfmax{delta}"] = cm
        for fb in ((0.1, 1.0), (1.0, 4.0)):
            if fb[0] < 0.99 / seglen:
                continue
            tau, st, g2, fm = xspec_lag(ta, tb, lo, hi, seglen, fb)
            row[f"xs{fb[0]:g}-{fb[1]:g}_ms"] = tau * 1e3; row[f"xs{fb[0]:g}-{fb[1]:g}_err_ms"] = st * 1e3
            row[f"g2_{fb[0]:g}-{fb[1]:g}"] = g2
        rows.append(row)
    rr = [x for x in rows if x["pulse"] == pn]
    say(f"[{pn}] counts C1/C2/C3/NaI = {nb['C1']}/{nb['C2']}/{nb['C3']}/{nb['NaI']}")
    for x in rr:
        say(f"      {x['band']:16s} vs {x['ref']:8s}: CCF16 {x['ccf16_ms']:+7.1f} (box +-{x['ccf16_boxerr_ms']:.1f})  CCF64 {x['ccf64_ms']:+7.1f} (+-{x['ccf64_boxerr_ms']:.1f})  "
            f"xspec 0.1-1 Hz {x.get('xs0.1-1_ms', np.nan):+7.1f}+-{x.get('xs0.1-1_err_ms', np.nan):.1f} (g2 {x.get('g2_0.1-1', np.nan):.2f})  "
            f"1-4 Hz {x.get('xs1-4_ms', np.nan):+7.1f}+-{x.get('xs1-4_err_ms', np.nan):.1f} (g2 {x.get('g2_1-4', np.nan):.2f})")

    # [N3] rate-dependent migration C1 -> C3 (deterministic, expected counts)
    tt_ = sel_t(t, np.ones(len(t), bool), lo, hi)
    sm = np.convolve(series(tt_, lo, hi), np.ones(100) / 100, mode="same")
    c1 = series(sel_t(t, BANDS["C1"], lo, hi), lo, hi); c3 = series(sel_t(t, BANDS["C3"], lo, hi), lo, hi)
    base = ccf_lag(c1, c3, 16)[0]
    out = []
    for eps in (0.006, 0.03):
        mv = eps * sm / sm.max() * c1
        out.append(f"eps={eps}: {1e3*(ccf_lag(c1 - mv, c3 + mv, 16)[0] - base):+.2f} ms")
    say(f"      [N3] C3-vs-C1 CCF lag shift from rate-proportional C1->C3 migration: " + ", ".join(out))
    rows.append(dict(pulse=pn, ref="C1", band="[N3] migration", note=", ".join(out)))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/lag_results.csv", index=False, float_format="%.5g")

# [G] per-detector gain spread (1K events carry the detector id; in the
# unsaturated intervals 1K = 1B event by event, eband.md)
h = np.load(HE1K)
for seg, (lo, hi) in {"quiet 20-170": (20.0, 170.0), "flare 530-600": (530.0, 600.0)}.items():
    w = (h["t"] >= lo) & (h["t"] < hi) & (h["pw"] > 75) & (h["ch"] >= 30) & (h["ch"] < 224)
    d, c = h["det"][w], h["ch"][w].astype(float)
    med = np.array([np.median(c[d == k]) for k in range(18)])
    q75 = np.array([np.percentile(c[d == k], 75) for k in range(18)])
    frac = np.array([((c[d == k] >= 30) & (c[d == k] < 53)).mean() for k in range(18)])
    say(f"[G] {seg}: CsI ch30-224 median channel per detector 0..17: " + " ".join(f"{x:.0f}" for x in med))
    say(f"      75th percentile: " + " ".join(f"{x:.0f}" for x in q75) + f"; max/min of median {med.max()/med.min():.2f}")
    say(f"      fraction in C1 (ch30-53): " + " ".join(f"{x:.2f}" for x in frac))

# [P] packet frequency in per-box PDS
for seg, (lo, hi) in {"quiet 20-170": (20.0, 170.0), "real 330-440": (330.0, 440.0)}.items():
    for b in range(3):
        tb = t_all[box == b]
        lam = ((tb >= lo) & (tb < hi)).sum() / (hi - lo)
        f, P, M, N, _ = leahy(tb, lo, hi, 16.0, 2.0 ** -12)
        Pw = np.median(P) / (stats.chi2.median(2 * M) / (2 * M))
        s = []
        for h in (1, 2, 3):
            fp = h * lam / 109
            m = np.abs(f - fp) < 1.0          # +-1 Hz around (rate drifts within the interval)
            x = P[m].mean() / Pw
            dof = 2 * M * m.sum()
            s.append(f"{h}x{fp:.1f} Hz: {x:.3f} (p={gamma_sf(x, dof):.2f})")
        say(f"[P] {seg} box {'ABC'[b]}: lambda={lam:.0f}/s, f_pkt=lambda/109; mean P/Pw within +-1 Hz: " + "; ".join(s))

# figure: tau(E) per pulse (Delta = 16 ms)
E = {"NaI": 0.3, "C1": 0.32, "C2": 0.7, "C3": 1.6, "G_HG30-100": 0.055, "G_HG100-350": 0.19, "G_LG0.4-1": 0.63, "G_LG1-3": 1.7}
fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
for a, pn in zip(axs.flat, PULSES):
    d = R[(R.pulse == pn) & (R.band.isin(list(E)))]
    for _, x in d.iterrows():
        c = "C0" if not x.band.startswith("G_") else "C1"
        if "xs0.1-1_ms" in x and np.isfinite(x["xs0.1-1_ms"]):
            a.errorbar(E[x.band], x["xs0.1-1_ms"], x["xs0.1-1_err_ms"], fmt="o", color=c, ms=4)
        if np.isfinite(x["ccf16_ms"]):
            a.plot(E[x.band] * 1.08, x["ccf16_ms"], "x", color=c, ms=4)
    a.errorbar(E["C1"], 0, 0, fmt="s", color="k", ms=4)
    a.axhline(0, color="k", lw=0.5); a.set_xscale("log"); a.set_title(pn, fontsize=9)
    a.set_xlabel("approx. energy (MeV)", fontsize=8)
axs[0, 0].set_ylabel("lag vs HE C1 (ms)\no: cross-spectrum 0.1-1 Hz, x: CCF\n(blue HE, orange GECAM-C)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_lag.png", dpi=130)
say("done")
