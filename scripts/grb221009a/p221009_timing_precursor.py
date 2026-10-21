#!/usr/bin/env python3
"""GRB 221009A precursor as seen by Insight-HXMT/HE (1K events, unsaturated).

- background: LargeEvt proxy model (p221009_timing_bkg) and, as a cross-check,
  a quadratic in time fitted to T0-300..-10 and T0+30..+170 s;
- Bayesian blocks (astropy, binned 'events' fitness, 0.05 s bins, p0 = 0.01)
  on T0-60..+175 s for the change points;
- T90/T50 from the background-subtracted cumulative counts (5%/95%, 25%/75%)
  on T0-5..+30 s, quoted for both background models;
- band ratios in raw-channel bands (HE_BANDS; low-gain mode, ~0.2-2.5 MeV)
  for the precursor vs the rise of the first main-emission pulse
  (T0+176..186 s: no gap-filled events, f_live > 0.95).
All deterministic; uncertainties are the spread between background models
plus the analytic Poisson error of the cumulative count at the 5/95% points.

Outputs: data/p221009/timing/precursor.{png,json}
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from p221009_timing_common import OUT, HE_BANDS
from p221009_timing_bkg import background_model, he1k_hour


def quad_bkg(t, ch, chmin):
    e = np.arange(-300.0, 171.0, 1.0)
    r = np.histogram(t[ch >= chmin], e)[0].astype(float)
    c = e[:-1] + 0.5
    m = (c < -10) | (c > 30)
    p = np.polyfit(c[m], r[m], 2)
    return p, np.std(r[m] - np.polyval(p, c[m]))


def t_frac(tc, cum, f):
    """First crossing time of cum/total = f (linear interpolation)."""
    y = cum / cum[-1]
    i = np.argmax(y >= f)
    if i == 0:
        return tc[0]
    return tc[i - 1] + (f - y[i - 1]) / (y[i] - y[i - 1]) * (tc[i] - tc[i - 1])


def durations(t_ev, bkg_rate_fn, lo=-5.0, hi=30.0, bw=0.05):
    e = np.arange(lo, hi + bw / 2, bw)
    n = np.histogram(t_ev, e)[0].astype(float)
    c = 0.5 * (e[:-1] + e[1:])
    net = n - bkg_rate_fn(c) * bw
    cum = np.cumsum(net)
    # plateau: the background-subtracted total is read as the mean cumulative
    # over the last 10 s (flat once the precursor is over)
    tot = cum[c > hi - 10].mean()
    out = {}
    for nm, f1, f2 in [("T90", 0.05, 0.95), ("T50", 0.25, 0.75)]:
        y = cum / tot
        t1 = c[np.argmax(y >= f1)]
        t2 = c[np.argmax(y >= f2)]
        out[nm] = (float(t1), float(t2), float(t2 - t1))
    out["net_counts"] = float(tot)
    out["gross_counts"] = float(n[(c >= out["T90"][0]) & (c <= out["T90"][1])].sum())
    return out, c, cum, tot


def main():
    t, ch = he1k_hour()
    chmin = 20  # 20 keV
    sel = (ch >= chmin)
    tl, r, model, ok, coef, resid = background_model(chmin=chmin)
    proxy = dict(zip(tl.astype(int), model))
    def bkg_proxy(c):
        return np.array([proxy[int(np.floor(x))] for x in c])
    pq, rq = quad_bkg(t, ch, chmin)
    def bkg_quad(c):
        return np.polyval(pq, c)

    tw = t[sel & (t > -60) & (t < 175)]
    # Bayesian blocks on 0.05 s binned counts
    bw = 0.05
    e = np.arange(-60.0, 175.0 + bw / 2, bw)
    n = np.histogram(tw, e)[0]
    c = 0.5 * (e[:-1] + e[1:])
    edges_bb = bayesian_blocks(c, n, fitness="events", p0=0.01)
    blocks = []
    for a, b in zip(edges_bb[:-1], edges_bb[1:]):
        m = (c >= a) & (c < b)
        rate = n[m].sum() / (m.sum() * bw)
        blocks.append((float(a), float(b), float(rate),
                       float(rate - bkg_proxy(c[m]).mean())))

    res = {"bkg_proxy_coef": coef.tolist(), "bkg_proxy_resid": float(resid),
           "bkg_quad_coef": pq.tolist(), "bkg_quad_resid": float(rq),
           "bayesian_blocks": blocks}
    curves = {}
    for nm, fn in [("proxy", bkg_proxy), ("quad", bkg_quad)]:
        d, cc, cum, tot = durations(t[sel], fn)
        res[f"dur_{nm}"] = d
        curves[nm] = (cc, cum / tot)

    # peak at 0.1 / 0.5 s
    for pbw in (0.1, 0.5, 1.0):
        e2 = np.arange(-5.0, 15.0 + pbw / 2, pbw)
        n2 = np.histogram(t[sel], e2)[0] / pbw
        c2 = 0.5 * (e2[:-1] + e2[1:])
        net2 = n2 - bkg_proxy(c2)
        k = int(np.argmax(net2))
        res[f"peak_{pbw}s"] = {"t_center": float(c2[k]), "net_rate": float(net2[k]),
                               "err": float(np.sqrt(n2[k] / pbw))}

    # band ratios: precursor (T90 window) vs first main-emission pulse (180-200 s)
    t90 = res["dur_proxy"]["T90"]
    br = {}
    for nm, (a, b) in [("precursor", (t90[0], t90[1])), ("pulse187_rise", (176.0, 186.0))]:
        row = {}
        for bn, (c1, c2) in HE_BANDS.items():
            m = (ch >= c1) & (ch < c2)
            gross = ((t >= a) & (t < b) & m).sum()
            # band background from the same pre-burst window, scaled by the proxy ratio
            pre = ((t >= -300) & (t < -10) & m).sum() / 290.0
            scale = bkg_proxy(np.array([0.5 * (a + b)]))[0] / bkg_proxy(np.arange(-299.5, -10, 1.0)).mean()
            net = gross - pre * scale * (b - a)
            row[bn] = (float(net), float(np.sqrt(gross)))
        br[nm] = row
    for nm, row in br.items():
        b1, b2, b3 = list(HE_BANDS)
        h = row[b2][0] / row[b1][0]
        h2 = row[b3][0] / row[b2][0]
        e_h = h * np.hypot(row[b2][1] / row[b2][0], row[b1][1] / row[b1][0])
        e_h2 = h2 * np.hypot(row[b3][1] / row[b3][0], row[b2][1] / row[b2][0])
        row[f"HR1({b2}/{b1})"] = (float(h), float(e_h))
        row[f"HR2({b3}/{b2})"] = (float(h2), float(e_h2))
    res["band_counts"] = br

    # quiescent-interval test: mean excess over the proxy in 20-170 s
    w = (tl >= 20) & (tl < 170)
    res["quiescent_excess_20_170"] = {
        "mean": float(np.mean(r[w] - model[w])),
        "err_stat": float(np.sqrt(np.mean(r[w]) / w.sum())),
        "resid_floor": float(resid / np.sqrt(w.sum()))}
    # main-emission onset: first BB block after 30 s whose proxy-net rate > 5 sigma(1-s resid)
    onset = next((b for b in blocks if b[0] > 30 and b[3] > 5 * resid), None)
    res["main_onset_block"] = onset
    (OUT / "precursor.json").write_text(json.dumps(res, indent=1))

    # ---- figure ----
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    pbw = 0.25
    e3 = np.arange(-10.0, 30.0 + pbw / 2, pbw)
    n3 = np.histogram(t[sel], e3)[0] / pbw
    c3 = 0.5 * (e3[:-1] + e3[1:])
    ax[0].step(e3[:-1], n3 - bkg_proxy(c3), where="post", color="C0", lw=0.9,
               label="HE all channels (ch>=20), net (0.25 s)")
    for a, b, rr, net in blocks:
        if b > -10 and a < 30:
            ax[0].hlines(net, max(a, -10), min(b, 30), color="k", lw=1.6)
    ax[0].axvspan(*t90[:2], color="C1", alpha=0.15, label=f"T90 = {t90[2]:.1f} s")
    ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_xlim(-10, 30); ax[0].set_ylabel("net rate (evt/s)")
    ax[0].legend(fontsize=8)
    for nm, (cc, y) in curves.items():
        ax[1].plot(cc, y, lw=1, label=f"bkg: {nm}")
    ax[1].axhline(0.05, color="0.6", ls=":"); ax[1].axhline(0.95, color="0.6", ls=":")
    ax[1].set_xlim(-5, 30); ax[1].set_ylabel("cumulative fraction"); ax[1].legend(fontsize=8)
    e4 = np.arange(-60.0, 200.0, 2.0)
    n4 = np.histogram(t[sel], e4)[0] / 2.0
    c4 = 0.5 * (e4[:-1] + e4[1:])
    ax[2].step(e4[:-1], n4, where="post", lw=0.8, label="gross (2 s)")
    ax[2].plot(c4, bkg_proxy(c4), color="C3", lw=1, label="LargeEvt background model")
    ax[2].plot(c4, bkg_quad(c4), color="C2", lw=1, ls="--", label="quadratic background")
    ax[2].set_ylim(8500, 14000); ax[2].set_xlabel("T - T0 (s)"); ax[2].set_ylabel("evt/s")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "precursor.png", dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "bayesian_blocks"}, indent=1))
    print("blocks (t1, t2, gross, net):")
    for b in blocks:
        print("  %.2f %.2f %.0f %.0f" % b)


if __name__ == "__main__":
    main()
