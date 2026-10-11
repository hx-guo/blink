#!/usr/bin/env python3
"""Variability time scales of GRB 221009A from HE events (Haar structure function).

For a window and scale D, the event list is binned at D; each adjacent pair
of bins gives d = N_right - N_left. For a source without variability on
scale D, E[d^2] = E[N_left + N_right] (Poisson) plus the contribution of the
slow trend. The excess
    S(D) = mean(d^2) - mean(N_left + N_right)
is the Haar variance of the source signal; its 1-sigma uncertainty is taken
as sqrt(2/n_pairs) * mean(d^2) (Gaussian approximation for the estimator of
a variance). A linear trend over the pair contributes (g D^2)^2 with g the
local slope; we remove it by also subtracting the squared mean of d (the
first-order term) -- i.e. S uses the variance of d, not its second moment.

HE shows a significant Haar excess at 2-8 ms even in source-free windows
(instrumental: correlated multi-event bursts in the background). For HE we
therefore subtract the excess measured in the quiescent interval T0+20..170 s
(no source, verified in p221009_timing_precursor), scaled per pair by the ratio
of the background-model rates, and propagate its error.

Reported per window: the smallest scale at which S/sigma_S > 3 (shortest
significant variability scale, t_var), and the full S(D) curve.

HE is used as a measurement only where the light curve contains no
gap-filled events, so that the event times are measured, not templated;
GECAM-C gives the same quantity in the flare as a reference and, in the
saturated flare spike, a check of how the HE gap-fill alters sub-second
variability. The GECAM-C reference band is LG 0.4-3 MeV (GRD01-12), matching
the ~0.1-3 MeV deposited-energy range of the low-gain, CsI-dominated HE
light curve.

Outputs: data/p221009/timing/mvt.{json,png}
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from p221009_timing_common import OUT, he_events

SCALES = 0.002 * 2.0 ** np.arange(0, 11)       # 2 ms .. 2.048 s
WINDOWS = {
    "precursor (-1..11)": (-1.0, 11.0),
    "pulse187 rise (176..184)": (176.0, 184.0),
    "flare 440..500": (440.0, 500.0),
    "flare spike 500..516": (500.0, 516.0),
}


def haar(t, lo, hi, D):
    nb = int(np.floor((hi - lo) / (2 * D))) * 2
    if nb < 4:
        return np.nan, np.nan, 0
    n = np.histogram(t, lo + D * np.arange(nb + 1))[0].astype(float)
    l, r = n[0::2], n[1::2]
    d = r - l
    m2 = np.var(d)
    pois = np.mean(l + r)
    s = m2 - pois
    sig = np.sqrt(2.0 / len(d)) * max(m2, pois)
    return s, sig, len(d)


BKG_REF = (20.0, 170.0)


def bkg_rate(lo, hi):
    from p221009_timing_bkg import background_model
    tl, r, model, ok, coef, resid = background_model(chmin=20)
    w = (tl >= lo) & (tl < hi)
    return float(model[w].mean())


def main():
    t, fill, ch, box = he_events()
    keep = ~((ch == 0) & (np.abs(t - np.round(t)) < 1e-7)) & (ch >= 20)
    the, fhe = t[keep], fill[keep]
    g = np.load(OUT / "gecam_c.npz")
    eb = np.load(OUT / "gecam_ebounds.npz")
    ge = np.sqrt(eb["emin"] * eb["emax"])[g["pi"]]
    tgc = g["t"][(g["gain"] == 1) & (ge >= 400) & (ge < 3000)]
    res = {}
    fig, ax = plt.subplots(1, len(WINDOWS), figsize=(4 * len(WINDOWS), 3.6))
    for a, (wn, (lo, hi)) in zip(ax, WINDOWS.items()):
        w = (the >= lo) & (the < hi)
        ff = float(fhe[w].mean())
        row = {"fill_frac": ff}
        for nm, tt in [("HE", the), ("HE_obs_only", the[~fhe]), ("GECAM", tgc)]:
            if nm == "GECAM" and lo < 330:
                continue
            if nm == "HE_obs_only" and ff == 0:
                continue
            S = [haar(tt, lo, hi, D) for D in SCALES]
            if nm.startswith("HE"):
                scale = bkg_rate(lo, hi) / bkg_rate(*BKG_REF)
                Sb = [haar(tt, BKG_REF[0], BKG_REF[1], D) for D in SCALES]
                S = [(s - scale * sb, np.hypot(e, scale * eb), k)
                     for (s, e, k), (sb, eb, _) in zip(S, Sb)]
            snr = np.array([s / e if e > 0 else np.nan for s, e, _ in S])
            sig = np.where(snr > 3)[0]
            tvar = float(SCALES[sig[0]]) if len(sig) else None
            row[nm] = dict(t_var=tvar, scales=SCALES.tolist(),
                           S=[float(s) for s, _, _ in S], err=[float(e) for _, e, _ in S],
                           snr=snr.tolist())
            # plot fractional rms per scale: sqrt(S)/(mean counts per bin)
            mean_rate = ((tt >= lo) & (tt < hi)).sum() / (hi - lo)
            frac = [np.sqrt(max(s, 0)) / (mean_rate * D) for (s, _, _), D in zip(S, SCALES)]
            lim = [np.sqrt(3 * e) / (mean_rate * D) for (_, e, _), D in zip(S, SCALES)]
            p = a.loglog(SCALES, frac, "o-", ms=3, label=f"{nm}  t_var={tvar}")
            a.loglog(SCALES, lim, ":", color=p[0].get_color(), lw=0.8)
            print(f"{wn:26s} {nm:12s} fill {ff:.2f}  t_var = {tvar}  "
                  f"snr: {' '.join(f'{x:.1f}' for x in snr)}")
        res[wn] = row
        a.set_title(f"{wn}\nHE gap-fill {ff:.2f}", fontsize=8)
        a.set_xlabel("scale D (s)"); a.legend(fontsize=6)
    ax[0].set_ylabel("fractional Haar rms (dotted: 3σ noise)")
    fig.tight_layout(); fig.savefig(OUT / "mvt.png", dpi=130)
    (OUT / "mvt.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
