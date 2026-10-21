"""GBM spectra of the bright, HE-clean intervals and their prediction in the
HE/CsI low-gain DRM (GRB 221009A response study).

Uses the gbm group's gdt-based fitter (gbm/scripts/grb221009a/p221009_gbm_specfit.py:
PG-stat, polynomial background) on n4+n8+b1 for the intervals where HE has
no FIFO loss and negligible pile-up (eband.md): 180-216, 288-440, 440-503 s
(HXMT T0; GBM times +0.06 s; GBM BTIs 219-277 and 508-514 s are avoided).
Each GBM best fit (COMP and Band) is folded through the HE DRM of that
direction and compared with the HE net spectrum (Cnt_LargeEvt-proxy
background, mean box live fraction): HE/GBM normalisation with the GBM shape
fixed, chi2 of that shape, and the HE free-fit Ep for comparison.

Run with the gdt venv:  resp/run_gdt.sh p221009_resp_gbmmain.py"""
import os
import sys

import numpy as np

sys.path.insert(0, "/scratchfs2/gecam/guohx/p221009/gbm/scripts")
from p221009_gbm_specfit import DetData, fit as gbm_fit   # noqa: E402

from p221009_resp_spec import band, counts, cpl, fit, flive_mean, group_drm, load_events, photon_flux  # noqa: E402
from p221009_resp_prec import efluxband  # noqa: E402

BASE = "/scratchfs2/gecam/guohx/p221009"
DT = 0.06
INTERVALS = {"rise": (180, 216), "mid": (288, 440), "flare": (440, 503)}
BKG = ([(-250.0, -10.0), (95.0, 118.0), (720.0, 790.0)], 2)
DETS = ("n4", "n8", "b1")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    t, ch = load_events()
    D = [DetData(d, *BKG, trange=(-400.0, 800.0)) for d in DETS]
    lines = []
    for name, (a, b) in INTERVALS.items():
        R = group_drm(z[f"{name}_per"].astype(float))
        src, bkg, bvar, expo = counts(t, ch, [(a, b)], mode="proxy")
        fl = flive_mean([(a, b)])
        for model in ("COMP", "Band"):
            r = gbm_fit(D, a + DT, b + DT, model)
            if model == "COMP":
                p = [np.log(r["A"]), r["index"], np.log(r["Epeak"])]
                mdl = cpl
                desc = f"a={r['index']:.3f} Ep={r['Epeak']:.0f}"
            else:
                p = [np.log(r["A"]), r["alpha"], np.log(r["Epeak"]), r["beta"]]
                mdl = band
                desc = f"a={r['alpha']:.3f} Ep={r['Epeak']:.0f} b={r['beta']:.2f}"
            fixed = {i: v for i, v in enumerate(p) if i > 0}
            q, c2 = fit(mdl, p, R, elo, ehi, src, bkg, bvar, expo, fl, fixed=fixed)
            q3, c23 = fit(mdl, p, R, elo, ehi, src, bkg, bvar, expo, fl, fixed=fixed, sys_frac=0.03)
            mu = photon_flux(mdl, p, elo, ehi) @ R * expo * fl
            resid = (src - bkg - mu) / mu
            lines.append(f"{name} [{a},{b}] GBM {model} {desc} PGstat {r['stat']:.0f}/{r['dof']}: "
                         f"HE/GBM norm {np.exp(q[0] - p[0]):.3f} (chi2 {c2:.0f}/{R.shape[1] - 1}; with 3% sys {c23:.1f}); "
                         f"F200-3000 GBM {r['eflux_200_3000']:.3e}; HE/GBM-pred counts {(src - bkg).sum() / mu.sum():.3f}; "
                         f"resid by group (HE-GBM)/GBM: " + " ".join(f"{x:+.2f}" for x in resid))
            print(lines[-1], flush=True)
    with open(f"{out}/xcal_main.txt", "w") as f:
        for l in lines:
            print(l, file=f)


if __name__ == "__main__":
    main()
