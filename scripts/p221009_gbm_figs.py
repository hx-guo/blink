"""Summary figure of the GBM precursor spectral evolution from
precursor/prec_fits.csv (run after p221009_gbm_precursor.py in the same RUN or
with GBM_PREC pointing to that file).  Unconstrained Ep upper errors are drawn
as lower limits.  Output $GBM_OUT/precursor/precursor_evolution_v2.png."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import p221009_gbm_common as C

od = C.outdir("precursor")
R = pd.read_csv(os.environ.get("GBM_PREC", os.path.join(od, "prec_fits.csv")))
tr = R[(R.tag == "tr") & (R.model == "COMP")].copy()
pl = R[(R.tag == "tr") & (R.model == "PL")].set_index("t1")
tr["dPG"] = pl.loc[tr.t1, "stat"].values - tr.stat.values
fig, ax = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.3, 1]))
edges = np.arange(-5, 45.01, 0.256)
for d, lab, e in [("n8", "n8 20-900 keV", (20, 900)), ("n4", "n4 20-900 keV", (20, 900)), ("b1", "b1 0.4-38 MeV", (400, 38000))]:
    tte = C.open_tte(d, (-400.0, 200.0))
    t = np.asarray(tte.data.times); pha = np.asarray(tte.data.channels)
    lo, hi = C.chan_range(tte.ebounds, *e)
    c, x = C.lightcurve(t, pha, edges, lo, hi)
    pre = np.histogram(t[(pha >= lo) & (pha <= hi)], [-150, -5])[0][0] / 145.0
    ax[0].step(edges[:-1], c / x - pre, where="post", lw=0.7, label=lab)
ax[0].set_ylabel("net counts/s"); ax[0].legend(fontsize=7); ax[0].axhline(0, color="k", lw=0.4)
xm = 0.5 * (tr.t1 + tr.t2); xe = 0.5 * (tr.t2 - tr.t1)
good = (tr.Epeak < 1e5) & np.isfinite(tr.Epeak_elo) & (tr.dPG > 2.0)
lim = ~good & np.isfinite(tr.Epeak_elo) & (tr.Epeak < 1e5)
ehi = tr.Epeak_ehi.where(np.isfinite(tr.Epeak_ehi), 0)
ax[1].errorbar(xm[good], tr.Epeak[good], xerr=xe[good], yerr=[tr.Epeak_elo[good], ehi[good]], fmt="o", ms=3, color="k",
               label="COMP, n4+n8+b1")
ll = (tr.Epeak - tr.Epeak_elo)[lim]
ax[1].errorbar(xm[lim], ll, xerr=xe[lim], yerr=0.4 * ll, lolims=True, fmt="none", color="0.4", label="lower limit (1σ)")
for y, lab, col in [(3980, "Lesage+23 Ia (with LLE)", "r"), (243, "243 keV (Song & Zhang 2023)", "g"), (1000, "", "b"), (1500, "Konus-Wind 0.97-1.5 MeV", "b")]:
    ax[1].axhline(y, color=col, ls="--", lw=0.7, label=lab or None)
ax[1].axvspan(-0.003, 8.576, color="y", alpha=0.15, label="Ia")
ax[1].set_yscale("log"); ax[1].set_ylim(50, 1e5); ax[1].set_ylabel("Ep (keV)"); ax[1].legend(fontsize=6, ncol=2)
ax[2].errorbar(xm, tr["index"], xerr=xe, yerr=[tr.index_elo.abs(), tr.index_ehi.abs().fillna(0)], fmt="o", ms=3, color="k")
ax[2].set_ylabel("COMP α"); ax[2].set_xlabel("t − T0(GBM) (s)"); ax[2].set_ylim(-2.1, -0.2)
ax[2].set_xlim(-3, 44)
fig.tight_layout(); fig.savefig(os.path.join(od, "precursor_evolution_v2.png"), dpi=150)
print("done")
