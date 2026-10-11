"""Supplementary figure data: HE/CsI vs GBM spectral shape at T0+180-216 s
(referee 2 round 2, R2; resp.md 9.1), the 'clean' variant of
p221009_resp_r2.py: observed events in the loss-free seconds 180-186 and
193-215 s, Cnt_LargeEvt-proxy background, rate-weighted live fraction; GBM
n4+n8+b1 Band fitted on the same seconds (gdt), folded through the rise-
direction HE DRM and summed over the two sub-windows.  C fitted with a 3%
per-group systematic (resp.md 9.1: C = 0.986, chi2 56.3/26).

    resp/run_gdt.sh p221009_v4_supp180.py OUTDIR       (gdt venv)

Output OUTDIR/supp180_groups.csv (per channel group: deposited-energy edges,
counts, background, model) and supp180_summary.txt.
"""
import sys

import numpy as np

from p221009_resp_r2 import BASE, BKG, DETS, DT, EC, live, f_eff, fit_c_delta
from p221009_resp_r2 import DetData, gbm_fit
from p221009_resp_spec import GROUP, band, counts, group_drm, load_events, photon_flux

WINS = [(180, 186), (193, 215)]
SYS = 0.03


def main():
    out = sys.argv[1]
    z = np.load(f"{BASE}/resp/out/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    R = group_drm(z["rise_per"].astype(float))
    t, ch = load_events()
    o, f = live()
    src, bkg, bvar, expo = counts(t, ch, WINS, mode="proxy")
    fe = f_eff(WINS, o, f)
    D = [DetData(d, *BKG, trange=(-400.0, 800.0)) for d in DETS]
    mu = np.zeros(R.shape[1])
    desc = []
    for a, b in WINS:
        r = gbm_fit(D, a + DT, b + DT, "Band")
        p = [np.log(r["A"]), r["alpha"], np.log(r["Epeak"]), r["beta"]]
        desc.append(f"[{a},{b}] alpha={r['alpha']:.2f} Ep={r['Epeak']:.0f} beta={r['beta']:.2f}")
        mu += photon_flux(band, p, elo, ehi) @ R * (b - a)
    mu_f = mu * fe
    net = src - bkg
    var = np.maximum(src, 1) + bvar + (SYS * mu_f) ** 2
    c = np.sum(net * mu_f / var) / np.sum(mu_f ** 2 / var)
    x2 = np.sum((net - c * mu_f) ** 2 / var)
    c0, d0, dlo, dhi, x2d = fit_c_delta(mu_f, net, var)
    k, b0 = np.median(EC[:, 0]), np.median(EC[:, 1])
    elo_g, ehi_g = k * GROUP[:-1] + b0, k * GROUP[1:] + b0
    np.savetxt(f"{out}/supp180_groups.csv",
               np.column_stack([GROUP[:-1], GROUP[1:], elo_g, ehi_g, src, bkg, bvar, mu_f]),
               delimiter=",", fmt="%.6g", comments="",
               header="ch_lo,ch_hi,E_lo,E_hi,src,bkg,bvar,mu_f")
    lines = [f"windows {WINS}, exposure {expo:.0f} s, f_eff {fe:.4f}, sys {SYS}",
             f"C = {c:.4f}  chi2 = {x2:.2f}/{len(net) - 1}",
             f"with tilt: C = {c0:.4f}  delta = {d0:+.3f} [{dlo:+.3f}, {dhi:+.3f}]  chi2 = {x2d:.2f}/{len(net) - 2}",
             "GBM Band: " + "; ".join(desc)]
    with open(f"{out}/supp180_summary.txt", "w") as fo:
        for l in lines:
            print(l, flush=True)
            print(l, file=fo)


if __name__ == "__main__":
    main()
