"""HE/CsI vs Fermi/GBM cross-normalisation on the precursor (GRB 221009A
response study).

Reads the gbm group's precursor fits (gbm/runs/prec4/precursor/prec_fits.csv,
n4+n8+b1 joint fits) and, for the same time windows (GBM T0 = HXMT T0 -
0.06 s), fits the HE/CsI spectrum with (a) the GBM best-fit shape fixed and
the normalisation free -> HE/GBM normalisation ratio, chi2 of the GBM shape
in the HE band; (b) a free power law -> 0.2-3 MeV energy flux and index,
compared with the GBM model flux in the same band and with the GBM BGO-only
power-law index.

    python p221009_resp_xcal.py [OUTDIR]"""
import csv
import sys

import numpy as np

from p221009_resp_spec import band, counts, cpl, fit, flive_mean, group_drm, load_events, photon_flux, pl
from p221009_resp_prec import efluxband

BASE = "/scratchfs2/gecam/guohx/p221009"
GBM_FITS = f"{BASE}/gbm/runs/prec4/precursor/prec_fits.csv"
DT = 0.06            # HXMT T0 - GBM T0 (s)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    R = group_drm(z["precursor_per"].astype(float))
    t, ch = load_events()
    rows = list(csv.DictReader(open(GBM_FITS)))
    lines = []
    for r in rows:
        if r["dets"] != "n4+n8+b1" or r["tag"] not in ("Ia_lesage", "HEwin", "tr") or r["bkg"] != "two":
            continue
        if r["model"] not in ("COMP", "Band", "PL"):
            continue
        t1, t2 = float(r["t1"]) - DT, float(r["t2"]) - DT
        w = [(t1, t2)]
        src, bkg, bvar, expo = counts(t, ch, w)
        fl = flive_mean([(np.floor(t1), np.ceil(t2))])
        A = float(r["A"])
        if r["model"] == "COMP":
            mdl, p = cpl, [np.log(A), float(r["index"]), np.log(float(r["Epeak"]))]
        elif r["model"] == "Band":
            if not r["beta"] or float(r["beta"]) <= -9:
                continue
            mdl, p = band, [np.log(A), float(r["alpha"]), np.log(float(r["Epeak"])), float(r["beta"])]
        else:
            mdl, p = pl, [np.log(A), float(r["index"])]
        fixed = {i: v for i, v in enumerate(p) if i > 0}
        q, c2 = fit(mdl, p, R, elo, ehi, src, bkg, bvar, expo, fl, fixed=fixed)
        mu = photon_flux(mdl, p, elo, ehi) @ R * expo * fl
        c2_gbmnorm = np.sum((src - bkg - mu) ** 2 / (np.maximum(mu + bkg, 1) + bvar))
        pp, c2p = fit(pl, [np.log(A), -1.9], R, elo, ehi, src, bkg, bvar, expo, fl)
        f_he = efluxband(pl, pp, 200, 3000)
        f_gbm = efluxband(mdl, p, 200, 3000)
        lines.append(f"[{r['t1']},{r['t2']}] GBM {r['model']} ({r['tag']}): HE/GBM norm (GBM shape) = "
                     f"{np.exp(q[0] - p[0]):.3f}, chi2 {c2:.1f}/{R.shape[1] - 1} (GBM norm fixed: {c2_gbmnorm:.1f}); "
                     f"HE PL index {pp[1]:.3f} chi2 {c2p:.1f}; F200-3000 HE(PL) {f_he:.3e} vs GBM model {f_gbm:.3e} "
                     f"(csv {float(r['eflux_200_3000']):.3e}) ratio {f_he / f_gbm:.3f}; net HE {src.sum() - bkg.sum():.0f}")
    with open(f"{out}/xcal_precursor.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            print(l, file=f)


if __name__ == "__main__":
    main()
