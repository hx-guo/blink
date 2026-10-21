"""Main-emission onset with common criteria for HE/CsI and GBM bands.

HE: CsI events (pw 76-255), unwrapped ch 20-255 (0.17-3.1 MeV) and three
sub-bands, rates divided by the 1-s mean box live fraction (resp ts_1s.csv).
GBM: n4+n7+n8 summed NaI bands, b0+b1 summed BGO bands, deadtime-corrected.
All times in HXMT T0 (GBM - C.HXMT_DT = 0.0108 s).  Background: linear in time on
145-172 s (64 ms bins).  Criteria per band:
  bb      first Bayesian-block change point after 166 s (0.05 s bins,
          events fitness, p0 = 0.01; as timing.md for HE);
  sig3    start of the first run of three 64 ms bins each > 3 sigma above
          background;
  fNN     first time the 0.512 s-smoothed net rate exceeds NN% (5, 10, 20)
          of its 176-192 s maximum and stays above it for 0.5 s;
          uncertainty from 100 Poisson resamplings (seed 1).
Outputs $GBM_OUT/onset/onset_common.csv, onset_common.png.
"""
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks
import p221009_gbm_common as C

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
import p221009_resp_spec as H     # noqa: E402

od = C.outdir("onset")
DT = 0.064
EDGES = np.arange(140.0, 200.0 + 1e-9, DT)
TC = 0.5 * (EDGES[:-1] + EDGES[1:])
BKGW = (145.0, 172.0)
REF = (182.0, 183.0)   # normalisation below the HE FIFO ceiling (peak ~187 s may be truncated in HE)
RNG = np.random.default_rng(1)

# ---------------------------------------------------------------- HE
t_he, ch_he = H.load_events()
fl = {}
with open(H.TS) as f:
    for r in csv.DictReader(f):
        fl.setdefault(int(float(r["t"])), []).append(float(r["flive"]))
fl = {k: np.mean(v) for k, v in fl.items()}
live_he = np.array([fl.get(int(np.floor(x)), 1.0) for x in TC])
bands = {}
for name, (c1, c2) in {"HE 0.17-3.1 MeV": (20, 256), "HE 0.17-0.59 MeV": (20, 53),
                       "HE 0.59-1.17 MeV": (53, 99), "HE 1.17-3.1 MeV": (99, 256)}.items():
    s = (ch_he >= c1) & (ch_he < c2)
    tt = t_he[s & (t_he >= 135) & (t_he < 205)]
    bands[name] = (tt, np.histogram(tt, EDGES)[0], DT * live_he)

# ---------------------------------------------------------------- GBM
groups = {"NaI": ["n4", "n7", "n8"], "BGO": ["b0", "b1"]}
gb = {"NaI": [(8, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 900)],
      "BGO": [(300, 700), (700, 1500), (1500, 3000), (3000, 10000)]}
for gname, dets in groups.items():
    data = []
    for d in dets:
        tte = C.open_tte(d, (130.0, 210.0))
        data.append((np.asarray(tte.data.times) - C.HXMT_DT, np.asarray(tte.data.channels), tte.ebounds))
    for e1, e2 in gb[gname]:
        cnt = 0; expo = 0; tts = []
        for t, pha, eb in data:
            lo, hi = C.chan_range(eb, e1, e2)
            c, x = C.lightcurve(t, pha, EDGES, lo, hi)
            cnt = cnt + c; expo = expo + x / len(dets)
            tts.append(t[(pha >= lo) & (pha <= hi) & (t >= 135) & (t < 205)])
        bands["GBM %s %g-%g keV" % (gname, e1, e2)] = (np.sort(np.concatenate(tts)), cnt, expo)


def onset(cnt, expo, tt=None):
    rate = cnt / expo
    m = (TC >= BKGW[0]) & (TC < BKGW[1])
    A = np.stack([np.ones(m.sum()), TC[m] - 170.0], 1)
    p = np.linalg.lstsq(A, rate[m], rcond=None)[0]
    b = p[0] + p[1] * (TC - 170.0)
    sig = (rate - b) / np.sqrt(np.maximum(b, 1e-9) / expo)
    out = {}
    run = (sig > 3) & (TC > 170)
    idx = [i for i in range(len(run) - 2) if run[i] and run[i + 1] and run[i + 2]]
    out["sig3"] = EDGES[idx[0]] if idx else np.nan
    k = int(round(0.512 / DT))
    net = np.convolve(rate - b, np.ones(k) / k, "same")
    pk = (TC >= 176) & (TC <= 192)
    peak = net[pk].max()
    out["peak_net"] = peak; out["bkg"] = b[pk].mean()
    hold = int(round(0.5 / DT))
    rm = (TC >= REF[0]) & (TC < REF[1])
    ref = (rate - b)[rm].mean()
    out["ref_net"] = ref
    for key, norm, fracs in (("f", peak, (0.05, 0.10, 0.20)), ("r", ref, (0.10, 0.25, 0.50))):
        for fr in fracs:
            above = net > fr * norm
            ii = [i for i in np.where(above & (TC > 170))[0] if above[i:i + hold].all()]
            out["%s%02d" % (key, fr * 100)] = TC[ii[0]] if ii else np.nan
    return out


def bb_first(tt):
    e = np.arange(160.0, 195.0 + 1e-9, 0.05)
    h = np.histogram(tt, e)[0] if tt is not None else None
    return h, e


def bb_from_hist(h, e):
    cp = bayesian_blocks(0.5 * (e[:-1] + e[1:]), h, fitness="events", p0=0.01)
    after = cp[cp > 166]
    return after[0] if len(after) else np.nan


rows = []
for name, (tt, cnt, expo) in bands.items():
    r = dict(band=name, **onset(cnt, expo))
    # Bayesian blocks on 0.05 s bins (+ Poisson resampling of the bins)
    h, e = bb_first(tt)
    r["bb"] = bb_from_hist(h, e)
    bbs = np.array([bb_from_hist(RNG.poisson(h), e) for _ in range(30)])
    r["bb_err"] = np.nanstd(bbs); r["bb_p16"], r["bb_p84"] = np.nanpercentile(bbs, [16, 84])
    # Poisson resampling for the threshold times
    sims = [onset(RNG.poisson(np.maximum(cnt, 0)), expo) for _ in range(100)]
    for key in ("sig3", "f05", "f10", "f20", "r10", "r25", "r50"):
        v = np.array([s[key] for s in sims], float)
        r[key + "_err"] = np.nanstd(v)
    r["S_over_B_peak"] = r["peak_net"] / r["bkg"]
    rows.append(r)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "band"}, flush=True)
O = pd.DataFrame(rows)
C.write_csv(O, "onset", "onset_common.csv")

# ---------------------------------------------------------------- figure
sel = ["HE 0.17-3.1 MeV", "HE 0.59-1.17 MeV", "GBM NaI 25-50 keV", "GBM NaI 100-200 keV", "GBM NaI 400-900 keV",
       "GBM BGO 700-1500 keV", "GBM BGO 1500-3000 keV"]
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, name in enumerate(sel):
    tt, cnt, expo = bands[name]
    rate = cnt / expo
    m = (TC >= BKGW[0]) & (TC < BKGW[1])
    p = np.polyfit(TC[m] - 170.0, rate[m], 1)
    net = np.convolve(rate - np.polyval(p, TC - 170.0), np.ones(8) / 8, "same")
    r = O[O.band == name].iloc[0]
    ax.plot(TC, net / r.peak_net, lw=0.8, color="C%d" % i, label=name)
    ax.plot(r.f10, 0.10, "o", color="C%d" % i, ms=4)
ax.axhline(0.1, color="k", lw=0.4, ls=":"); ax.axhline(0.05, color="k", lw=0.4, ls=":")
ax.set_xlim(168, 192); ax.set_ylim(-0.05, 0.6)
ax.set_xlabel("t - T0(HXMT) (s)"); ax.set_ylabel("net rate / peak (176-192 s)")
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(od, "onset_common.png"), dpi=140)
print("done")
