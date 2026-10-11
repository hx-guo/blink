"""Precursor Ep likelihood scans: PG-stat profile vs a fixed Epeak (COMP and
Band, other parameters free) for Lesage region Ia, the HE precursor window and
the 1-s peak, n4+n8+b1 (NaI 20-900 keV, BGO 0.4-38 MeV) and with BGO from
200 keV.  Tests the literature values Ep ~ 243 keV (Song & Zhang 2023),
0.97-1.5 MeV (Konus-Wind) and 3.98 MeV (Lesage+2023).
Outputs $GBM_OUT/precursor/epscan.csv, epscan.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import p221009_gbm_common as C
from p221009_gbm_specfit import DetData, fit, STARTS
import p221009_gbm_specfit as S

od = C.outdir("precursor")
D = {d: DetData(d, [(-250.0, -10.0), (95.0, 118.0)], 2) for d in ["n4", "n8", "b1"]}
EPS = np.logspace(np.log10(80.0), np.log10(40000.0), 28)
CASES = [("Ia", -0.003, 8.576, None), ("HEwin", -0.84, 10.06, None), ("peak0-1", 0.0, 1.024, None),
         ("Ia_bgo200", -0.003, 8.576, {"b1": (200.0, 38000.0)})]
orig = S.make_model
rows = []
for tag, t1, t2, er in CASES:
    for model in ["COMP", "Band"]:
        free = fit([D[d] for d in D], t1, t2, model, er, profile=())
        rows.append(dict(case=tag, model=model, Ep_fixed=np.nan, stat=free["stat"], dof=free["dof"],
                         Ep=free["Epeak"], alpha=free.get("index", free.get("alpha")), beta=free.get("beta", np.nan)))
        for ep in EPS:
            def mk(name, ep=ep):
                f = orig(name)
                k = [n for n, _, _ in f.param_list].index("Epeak")
                dv = list(f.default_values); dv[k] = ep; f.default_values = dv
                fr = list(f.free); fr[k] = False; f.free = fr
                return f
            S.make_model = mk
            S.STARTS = {**STARTS, model: [None]}
            r = fit([D[d] for d in D], t1, t2, model, er, profile=())
            S.make_model = orig; S.STARTS = STARTS
            rows.append(dict(case=tag, model=model, Ep_fixed=ep, stat=r["stat"], dof=r["dof"], Ep=ep,
                             alpha=r.get("index", r.get("alpha")), beta=r.get("beta", np.nan)))
        print(tag, model, "best Ep %.0f stat %.1f" % (free["Epeak"], free["stat"]), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(od, "epscan.csv"), index=False, float_format="%.6g")

R = pd.DataFrame(rows)
fig, axs = plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 3.4), sharey=True)
for ax, (tag, *_r) in zip(axs, CASES):
    for model, col in [("COMP", "C0"), ("Band", "C1")]:
        g = R[(R.case == tag) & (R.model == model)]
        best = g.stat.min()
        s = g[np.isfinite(g.Ep_fixed)]
        ax.plot(s.Ep_fixed, s.stat - best, "-o", ms=2, color=col, label=model)
    for x, lab in [(243, "243 keV"), (1000, ""), (1500, "KW"), (3980, "Lesage")]:
        ax.axvline(x, color="0.6", ls=":", lw=0.8)
    ax.axhline(1, color="k", lw=0.5); ax.axhline(9, color="k", lw=0.5, ls="--")
    ax.set_xscale("log"); ax.set_ylim(-0.5, 60); ax.set_title(tag, fontsize=9)
    ax.set_xlabel("fixed Ep (keV)")
axs[0].set_ylabel("delta PG-stat"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(od, "epscan.png"), dpi=130)
print("done")
