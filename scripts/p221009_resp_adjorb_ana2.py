"""Adjacent-orbit background, step 3b: is the burst-day / reference-orbit
difference in the CsI rate explained by the particle environment?

For burst day and references (d1008, d1007): CsI ch 20-255 rate (step 2) and
particle counters LargeEvt and Veto (step 2b; 1K HK time stamps are shifted
to the event time axis by the lag maximising the pre-burst correlation).

Models for the burst-day background, all calibrated only on source-free data
(T0-400...-10 s and the Earth-occulted 1900...2580 s), each validated by
(i) leave-out 10-s windows in T0-100...-10 s and (ii) predicting the
occulted window halves from pre-burst-only calibration:
  orb_s         reference mean x constant
  orb+dLE       orb_s + k (LE_burst - LE_ref)   (k fitted on cal windows)
  orb+dVeto     orb_s + k (Veto_burst - Veto_ref)
  LE            a + b LE                         (= v3 / proxy family)
  Veto          a + b Veto
  LE+Veto       a + b LE + c Veto
Quiet-period net counts (and 0.3-3 MeV flux) for each.

    python p221009_resp_adjorb_ana2.py [OUTDIR]"""
import sys

import numpy as np

import matplotlib.pyplot as plt

from p221009_resp_adjorb_ana import AD, GROUP, OCC, PRE, WINS, load, sel, wsum

BASE = "/scratchfs2/gecam/guohx/p221009"
REFS = ("d1008", "d1007")
CONV = 7.6e8            # CsI ch20-255 counts/s per erg cm-2 s-1 (0.3-3 MeV), p221009_resp_quiet.py


def cnts(label, tr):
    z = np.load(f"{AD}/cnts_{label}.npz")
    out = {}
    for k in ("le", "veto"):
        v = z[k]
        ok = np.isfinite(v)
        out[k] = np.interp(tr, tr[ok], v[ok])
    return out


def shift(v, lag):
    return np.roll(v, lag) if lag else v


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    tr, gb, _ = load("burst")
    cb = gb.sum(1)
    pre = sel(tr, PRE)
    lines = []
    # lag of HK counters vs events (burst day, pre-burst; maximise corr of 10-s smoothed series)
    pb = cnts("burst", tr)
    k10 = np.ones(10) / 10
    best = {}
    for key in ("le", "veto"):
        cs = []
        for lag in range(-8, 9):
            a = np.convolve(cb, k10, "same")[pre]
            b = np.convolve(shift(pb[key], lag), k10, "same")[pre]
            cs.append((np.corrcoef(np.diff(a), np.diff(b))[0, 1], lag))
        best[key] = max(cs)[1]
        lines.append(f"HK lag for {key}: {best[key]:+d} s (corr of differenced 10-s series {max(cs)[0]:.3f})")

    def counters(label):
        c = cnts(label, tr)
        return {k: shift(v, best[k]) for k, v in c.items()}

    P = {"burst": counters("burst")}
    refs = {}
    for r in REFS:
        _, g, _ = load(r)
        refs[r] = g.sum(1)
        P[r] = counters(r)
    cr = np.mean([refs[r] for r in REFS], axis=0)
    le_r = np.mean([P[r]["le"] for r in REFS], axis=0)
    ve_r = np.mean([P[r]["veto"] for r in REFS], axis=0)
    lines.append(f"pre-burst means: CsI burst {cb[pre].mean():.0f} ref {cr[pre].mean():.0f}; "
                 f"LE burst {P['burst']['le'][pre].mean():.0f} ref {le_r[pre].mean():.0f}; "
                 f"Veto burst {P['burst']['veto'][pre].mean():.0f} ref {ve_r[pre].mean():.0f}")

    def design(name):
        le, ve = P["burst"]["le"], P["burst"]["veto"]
        if name == "orb_s":
            return np.column_stack([cr])
        if name == "orb+dLE":
            return np.column_stack([cr, le - le_r * (le[pre].mean() / le_r[pre].mean())])
        if name == "orb+dVeto":
            return np.column_stack([cr, ve - ve_r * (ve[pre].mean() / ve_r[pre].mean())])
        if name == "LE":
            return np.column_stack([np.ones(tr.size), le])
        if name == "Veto":
            return np.column_stack([np.ones(tr.size), ve])
        if name == "LE+Veto":
            return np.column_stack([np.ones(tr.size), le, ve])
        raise ValueError(name)

    def model(name, cal):
        X = design(name)
        m = np.zeros(tr.size, bool)
        for w in cal:
            m |= sel(tr, w)
        c = np.linalg.lstsq(X[m], cb[m], rcond=None)[0]
        return X @ c, c

    names = ("orb_s", "orb+dLE", "orb+dVeto", "LE", "Veto", "LE+Veto")
    vb = [(a, a + 10) for a in range(-100, -10, 10)]
    stat10 = np.sqrt(np.mean([wsum(tr, cb, w) for w in vb]))
    lines.append(f"VALIDATION (Poisson per 10 s: {stat10:.0f})")
    sysd = {}
    for n in names:
        res = []
        for w in vb:
            cal = [(-400, w[0] - 5)] if n == "orb_s" else [(-400, w[0] - 5), OCC]
            b, _ = model(n, cal)
            res.append(wsum(tr, cb - b, w))
        rms = np.sqrt(np.mean(np.square(res)))
        sysd[n] = np.sqrt(max(rms ** 2 - stat10 ** 2, 0))
        occ = []
        for w in ((1900, 2240), (2240, 2580)):
            b, _ = model(n, [PRE])
            occ.append(wsum(tr, cb - b, w) / np.ptp(w))
        _, c = model(n, [PRE, OCC])
        lines.append(f"  {n:10s}: pre-burst leave-out mean {np.mean(res):+6.0f} rms {rms:5.0f} (extra sys {sysd[n]:.0f}/10 s); "
                     f"occulted halves predicted from pre-burst only: {occ[0]:+6.1f}, {occ[1]:+6.1f} /s; coef {np.round(c, 4)}")
    lines.append("QUIET-PERIOD NET (CsI ch 20-255; calibration pre-burst + occulted)")
    series = {}
    for n in names:
        b, _ = model(n, [PRE, OCC])
        series[n] = cb - b
        for wn, w in WINS.items():
            net = wsum(tr, cb - b, w)
            nb = np.ptp(w) / 10
            st = np.sqrt(wsum(tr, cb, w) * (1 + (1 / len(REFS) if n.startswith("orb") else 0)))
            sg = np.sqrt(st ** 2 + (sysd[n] * nb) ** 2)        # sys coherent over the window
            ul = max(net, 0) + 3 * sg
            lines.append(f"  {wn:18s} {n:10s}: {net:+7.0f} +- {st:4.0f} (stat) +- {sysd[n] * nb:5.0f} (sys) = "
                         f"{net / np.ptp(w):+6.1f} +- {sg / np.ptp(w):4.1f} /s ({net / sg:+.1f} sig); "
                         f"F(0.3-3MeV) 3sig UL {ul / np.ptp(w) / CONV:.1e}")
    with open(f"{out}/adjorb2_summary.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            print(l, file=f)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    edges = np.arange(-400, 2620, 10)
    for n, s in series.items():
        y = [wsum(tr, s, (a, a + 10)) / 10 for a in edges]
        ax[0].plot(edges + 5, y, lw=0.8, label=n)
        ax[1].plot(edges + 5, y, lw=0.8, label=n)
    ax[0].set_ylim(-300, 500); ax[1].set_xlim(-150, 180); ax[1].set_ylim(-150, 350)
    for a in ax:
        a.axhline(0, c="k", lw=0.5); a.legend(fontsize=7, ncol=3); a.set_ylabel("net CsI counts/s")
        a.axvspan(1884, 2620, color="gray", alpha=0.2)
    ax[1].set_xlabel("t - T0 (s)")
    fig.tight_layout(); fig.savefig(f"{out}/adjorb2_net.png", dpi=110)


if __name__ == "__main__":
    main()
