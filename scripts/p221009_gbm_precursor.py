"""Step 2-3a: GBM precursor (triggering pulse) spectroscopy.

1. Reproduce Lesage+2023 region Ia (-0.003...8.576 s, N4+N8+B1, NaI 20-900 keV,
   BGO > 400 keV, COMP) without LLE.
2. Systematics of Ia: background model, BGO lower bound 200 vs 400 keV,
   NaI only, more NaI, both BGOs.
3. Region I (-1.343...42.881 s), the HE precursor window, and time-resolved
   sub-intervals (hard-to-soft evolution).
Outputs $GBM_OUT/precursor/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import p221009_gbm_common as C
from p221009_gbm_specfit import DetData, fit

od = C.outdir("precursor")
BKG = {"two": ([(-250.0, -10.0), (95.0, 118.0)], 2), "pre": ([(-150.0, -5.0)], 1)}
DS = ["n3", "n4", "n6", "n7", "n8", "b0", "b1"]
D = {k: {d: DetData(d, *v) for d in DS} for k, v in BKG.items()}
print("data loaded", flush=True)

rows = []


def run(tag, t1, t2, dets, models, bkg="two", eranges=None):
    for m in models:
        r = fit([D[bkg][d] for d in dets], t1, t2, m, eranges)
        r.pop("_fitter")
        r.update(tag=tag, bkg=bkg, eranges=str(eranges or ""))
        rows.append(r)
        print(tag, m, bkg, "stat %.1f/%d" % (r["stat"], r["dof"]),
              "Ep", r.get("Epeak"), "idx", r.get("index", r.get("alpha")), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(od, "prec_fits.csv"), index=False, float_format="%.6g")


L = ["n4", "n8", "b1"]
IA = (-0.003, 8.576)
# 1. Lesage Ia
run("Ia_lesage", *IA, L, ["COMP", "Band", "PL"])
# 2. systematics
run("Ia_bkgpre", *IA, L, ["COMP", "Band"], bkg="pre")
run("Ia_bgo200", *IA, L, ["COMP", "Band"], eranges={"b1": (200.0, 38000.0)})
run("Ia_naionly", *IA, ["n4", "n8"], ["COMP", "Band", "PL"])
run("Ia_5nai_b1", *IA, ["n3", "n4", "n6", "n7", "n8", "b1"], ["COMP", "Band"])
run("Ia_n4n8_b0b1", *IA, ["n4", "n8", "b0", "b1"], ["COMP", "Band"])
run("Ia_bgoonly", *IA, ["b1"], ["COMP", "PL"])
run("Ia_nai300", *IA, L, ["COMP", "Band"], eranges={"n4": (20.0, 300.0), "n8": (20.0, 300.0)})
# 3. whole region I, HE window
run("I_lesage", -1.343, 42.881, L, ["COMP", "Band"])
run("HEwin", -0.84, 10.06, L, ["COMP", "Band"])
run("HEwin_naionly", -0.84, 10.06, ["n4", "n8"], ["COMP", "Band"])
# time-resolved
edges = [-1.3, 0.0, 1.0, 2.0, 3.0, 4.5, 6.5, 8.576, 13.0, 20.0, 30.0, 43.0]
for a, b in zip(edges[:-1], edges[1:]):
    run("tr", a, b, L, ["COMP", "Band", "PL"])

R = pd.DataFrame(rows)
# ---------------------------------------------------------------- figure
tr = R[(R.tag == "tr") & (R.model == "COMP")]
edges_lc = np.arange(-10, 50.01, 0.256)
fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
for d, lab in [("n4", "n4 8-900 keV"), ("n8", "n8 8-900 keV"), ("b1", "b1 0.4-38 MeV")]:
    dd = D["two"][d]
    t = C.rel(dd.tte.data.times); pha = dd.tte.data.channels
    lo, hi = C.chan_range(dd.tte.ebounds, *((8, 900) if d[0] == "n" else (400, 38000)))
    c, e = C.lightcurve(t, pha, edges_lc, lo, hi)
    ax[0].step(edges_lc[:-1], c / e, where="post", lw=0.7, label=lab)
ax[0].set_ylabel("counts/s"); ax[0].set_yscale("log"); ax[0].legend(fontsize=7)
xm = 0.5 * (tr.t1 + tr.t2); xe = 0.5 * (tr.t2 - tr.t1)
ax[1].errorbar(xm, tr.Epeak, xerr=xe, yerr=[tr.Epeak_elo.abs(), tr.Epeak_ehi.abs()],
               fmt="o", ms=3, label="GBM COMP (n4+n8+b1)")
ia = R[(R.tag == "Ia_lesage") & (R.model == "COMP")].iloc[0]
ax[1].axhline(3980, color="r", ls="--", lw=0.8, label="Lesage+23 Ia 3.98 MeV")
ax[1].axhline(ia.Epeak, color="k", ls=":", lw=0.8, label="this work Ia")
ax[1].axhline(243, color="g", ls="--", lw=0.8, label="Song&Zhang 243 keV")
ax[1].set_yscale("log"); ax[1].set_ylabel("Ep (keV)"); ax[1].legend(fontsize=6)
ax[2].errorbar(xm, tr["index"], xerr=xe, yerr=[tr.index_elo.abs(), tr.index_ehi.abs()], fmt="o", ms=3)
ax[2].set_ylabel("COMP alpha"); ax[2].set_xlabel("t - T0(GBM) (s)")
fig.tight_layout(); fig.savefig(os.path.join(od, "precursor_evolution.png"), dpi=130)
print("done")
