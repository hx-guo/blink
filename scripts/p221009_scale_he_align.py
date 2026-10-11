#!/usr/bin/env python3
"""GRB 221009A: HE internal hardness per phase, flare ratio against a local
continuum, and HXMT-GECAM time alignment / 1-s bin-edge sensitivity.

HE event classes: NaI = pulse width 54-70, CsI = pulse width > 70. CsI channel
bands use the 2021-12 normal-mode CsI ch->keV map (csi_ch2e_211211.npy) as an
approximate deposit scale: ch20-48 ~ 60-130, ch48-100 ~ 130-300, ch100-255
~ 300-800 keV. Outputs: he_hardness.csv, align.txt, he_align.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from p221009_scale_common import (OUT, PHASES, SCALE, gecam_btime_channels,
                                  hxmt_1s, hxmt_events, rebin_sum, robust, saa_mask)

PH = {"bkg -50..-10": (-50, -10), "calib 182-205": (182, 205), **PHASES}


def he_hardness():
    """Background-subtracted (pre-burst spectrum scaled by duration) HE hardness."""
    t, fill, ch, pw, box = hxmt_events()
    o = ~fill

    def counts(lo, hi):
        w = o & (t >= lo) & (t < hi)
        c, p = ch[w], pw[w]
        csi = p > 70
        nai = (p >= 54) & (p <= 70)
        return np.array([np.sum(csi & (c >= 20) & (c < 48)), np.sum(csi & (c >= 48) & (c < 100)),
                         np.sum(csi & (c >= 100)), np.sum(nai & (c >= 20) & (c < 43)),
                         np.sum(nai & (c >= 43) & (c < 155)), np.sum(nai), w.sum()], float)

    bk = counts(-50, -10) / 40.0
    lines = ["phase,n_obs,nai_frac_net,csi_mid_over_low,e,csi_high_over_low,e,nai_50_250_over_20_50,e"]
    print(f"{'phase (bkg-subtracted)':<22}{'n':>9}{'NaI frac':>9}{'CsI mid/low':>14}"
          f"{'CsI hi/low':>14}{'NaI hi/lo':>14}")
    for name, (lo, hi) in PH.items():
        if lo < 0:
            continue
        g = counts(lo, hi); b = bk * (hi - lo); n = g - b
        v = g + b
        def rat(i, j):
            r = n[i] / n[j]
            return r, r * np.sqrt(v[i] / n[i] ** 2 + v[j] / n[j] ** 2)
        (r1, e1), (r2, e2), (r3, e3) = rat(1, 0), rat(2, 0), rat(4, 3)
        nf = n[5] / n[6]
        print(f"{name:<22}{int(g[6]):9d}{nf:9.3f}{r1:8.3f}±{e1:5.3f}{r2:8.3f}±{e2:5.3f}{r3:8.3f}±{e3:5.3f}")
        lines.append(f"{name},{int(g[6])},{nf:.4f},{r1:.4f},{e1:.4f},{r2:.4f},{e2:.4f},{r3:.4f},{e3:.4f}")
    (OUT / "he_hardness.csv").write_text("\n".join(lines) + "\n")
    print("note: observed events only; on P1/P2 the channel/pulse-width spectra are distorted "
          "by pile-up and dead time (f_live 0.1-0.3), so only unsaturated phases are clean")


def flare_continuum(h, lg):
    """Flare HXMT/GECAM after subtracting a linear local continuum from both."""
    x = h["x"]
    anc = ((x >= 400) & (x < 425)) | ((x >= 615) & (x < 640))
    out = []
    for name, y in [("he", h["net_dt"]), ("lg", lg)]:
        k = anc & np.isfinite(y)
        a, b = np.polyfit(x[k], y[k], 1)
        out.append(y - (a * x + b))
    he_ex, lg_ex = out
    w = (x >= 442) & (x < 593) & (lg_ex > 50)
    r = robust(he_ex[w] / (SCALE * lg_ex[w]))
    s = he_ex[(x >= 442) & (x < 593)].sum() / (SCALE * lg_ex[(x >= 442) & (x < 593)].sum())
    w0 = (x >= 442) & (x < 593) & (lg > 50) & (h["net"] > 0)
    return r, s, robust(h["net_dt"][w0] / (SCALE * lg[w0]))


def xcorr(a, b, maxlag):
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    lags = np.arange(-maxlag, maxlag + 1)
    cc = np.array([np.mean(a[max(0, -L):len(a) - max(0, L)] * b[max(0, L):len(b) - max(0, -L)])
                   for L in lags])
    return lags, cc


def main():
    he_hardness()
    t, fill, ch, pw, box = hxmt_events()
    tg, c, cb, e = gecam_btime_channels()
    lgnet50 = (c - cb)[:, 9:13].sum(1)
    log = []
    # --- 50 ms cross-correlation on unsaturated windows (HE gross, no DT issue) ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (lo, hi) in enumerate([(182, 212), (500, 520)]):
        ed = np.arange(lo, hi + 1e-9, 0.05)
        hh = np.histogram(t, ed)[0].astype(float)
        gg = rebin_sum(tg, lgnet50, ed)
        lags, cc = xcorr(hh, gg, 40)
        k = np.argmax(cc)
        # parabolic refinement
        if 0 < k < len(cc) - 1:
            d = 0.5 * (cc[k - 1] - cc[k + 1]) / (cc[k - 1] - 2 * cc[k] + cc[k + 1])
        else:
            d = 0.0
        lag_s = (lags[k] + d) * 0.05
        msg = (f"xcorr {lo}-{hi}: peak lag {lag_s*1e3:+.0f} ms (positive = GECAM later), "
               f"cc_max {cc[k]:.3f}")
        print(msg); log.append(msg)
        ax[i].plot(lags * 50, cc); ax[i].axvline(0, color="0.6")
        ax[i].set_xlabel("lag (ms), +: GECAM lags HE"); ax[i].set_title(f"T0+{lo}-{hi}, 50 ms bins")
    fig.savefig(OUT / "he_align.png", dpi=140, bbox_inches="tight")

    # --- sensitivity of the main-window ratio to a GECAM time shift ---
    edges = np.arange(-50, 701, 1.0)
    h = hxmt_1s(edges)
    x = h["x"]; bad = saa_mask(x)
    n = rebin_sum(tg, np.ones_like(tg), edges)
    for sh in (-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5):
        lg = rebin_sum(tg + sh, lgnet50, edges) / (rebin_sum(tg + sh, np.ones_like(tg), edges) * 0.05)
        for lo, hi in [(170, 310), (440, 610)]:
            w = (x >= lo) & (x < hi) & (lg > 50) & (h["net"] > 0) & ~bad
            r = robust(h["net_dt"][w] / (SCALE * lg[w]))
            pk = (x >= 225) & (x < 229)
            msg = (f"GECAM shift {sh:+.1f} s, window {lo}-{hi}: median {r[0]:.3f} ± {r[1]:.3f} "
                   f"(n={r[2]})" + (f"; P1 225-228 bins {np.round(h['net_dt'][pk]/(SCALE*lg[pk]),3)}"
                                    if lo == 170 else ""))
            print(msg); log.append(msg)
    lg0 = rebin_sum(tg, lgnet50, edges) / (n * 0.05)
    r, s, r0 = flare_continuum(h, lg0)
    msg = (f"flare 442-593 local-continuum-subtracted: median {r[0]:.3f} ± {r[1]:.3f} (n={r[2]}), "
           f"integrated {s:.3f}; without continuum: median {r0[0]:.3f} ± {r0[1]:.3f}")
    print(msg); log.append(msg)
    (OUT / "align.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
