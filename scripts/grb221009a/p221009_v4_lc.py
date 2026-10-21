"""v4 precursor light curves: HE/CsI (0.17-3.1 MeV) and Fermi-GBM NaI/BGO bands.

    run_gbm.sh -argu RUN p221009_v4_lc      (gdt venv; imports p221009_resp_spec)

Time axis: paper T0 = 2022-10-09T13:17:00.000 UTC = HXMT MET 339945423.0, the zero of
the HE event cache.  GBM: t_paper = t_GBM - 0.0108 s (GBM trigger 13:16:59.9875 UTC
-> -0.0125 s; Fermi-HXMT light travel towards the burst +0.0017 s; p221009_v4_ltt.py).
Bins 0.256 s on T0-20...+25 s.
HE: CsI events (pw 76-255), unwrapped ch 20-255 (0.17-3.1 MeV) and two
sub-bands, rate divided by the 1-s mean box live fraction (resp ts_1s.csv);
background linear in time on T0-101...-6 s (resp 'tloc', unbiased on
pre-burst null windows, resp.md 3.1), divided by the mean live fraction of that window.
GBM: n4+n8 NaI and b1 (the spectral-fit detectors of gbm.md 2/5), deadtime-
corrected; background per band quadratic in time on 1.024 s bins over
T0-250...-10 and +95...+118 s (the main-line background of gbm.md 2.1).
Output $GBM_OUT/v4/prec_lc.csv (rate, bkg, err per band; err = sqrt(counts)/expo).
"""
import csv
import sys
import numpy as np
import pandas as pd
import p221009_gbm_common as C

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
import p221009_resp_spec as H     # noqa: E402

od = C.outdir("v4")
G2P = -0.0108         # t_paper = t_GBM + G2P (p221009_v4_ltt.py, t0_offsets.txt)
DT = 0.256
EDGES = np.arange(-20.0, 25.0 + 1e-9, DT)
TC = 0.5 * (EDGES[:-1] + EDGES[1:])
out = {"t_lo": EDGES[:-1], "t_hi": EDGES[1:]}

# ---------------------------------------------------------------- HE
t_he, ch_he = H.load_events()
fl = {}
with open(H.TS) as f:
    for r in csv.DictReader(f):
        fl.setdefault(int(np.floor(float(r["t"]))), []).append(float(r["flive"]))
fl = {k: np.mean(v) for k, v in fl.items()}
live = np.array([fl.get(int(np.floor(x)), 1.0) for x in TC])
BW = (-101.0, -6.0)
be = np.arange(BW[0], BW[1] + 1e-9, 1.0)
bc = 0.5 * (be[:-1] + be[1:])
live_b = np.mean([fl.get(int(np.floor(x)), 1.0) for x in bc])
print("mean live fraction in the background window %.4f" % live_b, flush=True)
for name, (c1, c2) in {"he": (20, 256), "he_lo": (20, 53), "he_hi": (53, 256)}.items():
    s = (ch_he >= c1) & (ch_he < c2)
    cnt = np.histogram(t_he[s], EDGES)[0]
    hb = np.histogram(t_he[s], be)[0]
    p = np.polyfit(bc, hb, 1)
    expo = DT * live
    out[name + "_rate"] = cnt / expo
    out[name + "_bkg"] = np.polyval(p, TC) / live_b  # raw counts/s -> live-corrected, as the source rate
    out[name + "_err"] = np.sqrt(np.maximum(cnt, 1)) / expo
    print(name, "bkg at T0 %.1f /s" % np.polyval(p, 0.0), flush=True)

# ---------------------------------------------------------------- GBM
BKG = [(-250.0, -10.0), (95.0, 118.0)]
FE = np.arange(-260.0, 130.0 + 1e-9, 1.024)
FC = 0.5 * (FE[:-1] + FE[1:])
fm = np.zeros(FC.size, bool)
for a, b in BKG:
    fm |= (FC >= a) & (FC < b)
bands = {"nai_lo": (["n4", "n8"], 8.0, 50.0), "nai_mid": (["n4", "n8"], 50.0, 300.0),
         "nai_hi": (["n4", "n8"], 300.0, 900.0),
         "bgo_lo": (["b1"], 400.0, 1500.0), "bgo_hi": (["b1"], 1500.0, 38000.0)}
data = {}
for d in ("n4", "n8", "b1"):
    tte = C.open_tte(d, (-300.0, 150.0))
    data[d] = (np.asarray(tte.data.times) + G2P, np.asarray(tte.data.channels), tte.ebounds)
eb_real = {}
for name, (dets, e1, e2) in bands.items():
    rate = 0; bkg = 0; var = 0
    for d in dets:
        t, pha, eb = data[d]
        le_ = np.asarray(eb.low_edges())
        lo, hi = int(np.searchsorted(le_, e1)), int(np.searchsorted(le_, e2)) - 1   # low edge in [e1, e2): no shared channels
        eb_real[name] = (float(np.asarray(eb.low_edges())[lo]), float(np.asarray(eb.high_edges())[hi]))
        c, x = C.lightcurve(t, pha, EDGES, lo, hi)
        cf, xf = C.lightcurve(t, pha, FE, lo, hi)
        p = np.polyfit(FC[fm], (cf / xf)[fm], 2)
        rate = rate + c / x
        bkg = bkg + np.polyval(p, TC)
        var = var + np.maximum(c, 1) / x ** 2
    out[name + "_rate"] = rate
    out[name + "_bkg"] = bkg
    out[name + "_err"] = np.sqrt(var)
    print(name, dets, "%.1f-%.1f keV" % eb_real[name], "bkg at T0 %.1f /s" % np.polyval(p, 0.0), flush=True)

pd.DataFrame(out).to_csv(f"{od}/prec_lc.csv", index=False, float_format="%.6g")
with open(f"{od}/prec_lc_bands.txt", "w") as f:
    f.write("he 0.17-3.1 MeV (ch 20-255); he_lo ch 20-52 (0.17-0.59 MeV); he_hi ch 53-255 (0.59-3.1 MeV)\n")
    for k, (lo, hi) in eb_real.items():
        f.write("%s %s %.1f-%.1f keV\n" % (k, "+".join(bands[k][0]), lo, hi))
print("done")
