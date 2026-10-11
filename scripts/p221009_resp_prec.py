"""Precursor CsI spectroscopy, main-emission checks and quiet-period limits
with the low-gain DRM (GRB 221009A response study).

Builds on p221009_resp_spec.py (same events, grouping, background, DRM).

A. Precursor: power-law index over 0.2-3 MeV with profile errors; separate
   indices below/above ch 53 (~0.6 MeV) as a curvature test; shape tests of
   published precursor spectra (norm free); time-resolved PL/CPL in the
   Song & Zhang 2023 segments; index systematics from DRM variants, +-5%
   gain and background order.
B. Bright clean intervals: CPL/Band with a 3% per-group systematic, and a
   scan of a common gain factor f (E_true = f * E_EC).
(The quiet-period limits are in p221009_resp_quiet.py.)

    python p221009_resp_prec.py [OUTDIR]"""
import sys

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz

import matplotlib.pyplot as plt

from p221009_resp_spec import (GROUP, band, counts, cpl, fit, flive_mean, gain_shift,
                               group_drm, load_events, photon_flux, pl)

BASE = "/scratchfs2/gecam/guohx/p221009"
LIT = {  # published precursor fits (precursor_lit.md section 1)
    "Lesage23 COMP a=-1.69 Ep=3980": (cpl, [np.nan, -1.69, np.log(3980)]),
    "Song&Zhang23 Band a=-1.55 b=-2.02 Ep=243": (band, [np.nan, -1.55, np.log(243), -2.02]),
    "Ror24 Band a=-1.66 b=-3.09 Ep=1531": (band, [np.nan, -1.661, np.log(1531), -3.09]),
    "Ror24 CPL a=-1.60 Ep=1310": (cpl, [np.nan, -1.598, np.log(1310)]),
    "KW23 CPL a=-1.65 Ep=1495": (cpl, [np.nan, -1.65, np.log(1495)]),
    "Axelsson25 seg5 COMP a=-1.88 Ep=330": (cpl, [np.nan, -1.88, np.log(330)]),
}
SEGS = [(-1, 1.2), (1.2, 2.4), (2.4, 3.9), (3.9, 6.7), (6.7, 10)]


def efluxband(model, p, e1, e2):
    e = np.geomspace(e1, e2, 400)
    return _trapz(e * model(e, *p), e) * 1.602e-9      # erg cm-2 s-1


def sub(mask, R, src, bkg, bvar):
    return R[:, mask], src[mask], bkg[mask], bvar[mask]


def pl_index(R, elo, ehi, src, bkg, bvar, expo, fl, sys_frac=0.0):
    p, c2 = fit(pl, [-3, -1.8], R, elo, ehi, src, bkg, bvar, expo, fl, bounds={1: (-4, 1)}, sys_frac=sys_frac)
    grid = np.linspace(p[1] - 0.4, p[1] + 0.4, 161)
    prof = np.array([fit(pl, [p[0], g], R, elo, ehi, src, bkg, bvar, expo, fl, fixed={1: g},
                         sys_frac=sys_frac)[1] for g in grid])
    ok = prof <= c2 + 1.0
    return p, c2, grid[ok].min(), grid[ok].max()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    t, ch = load_events()
    per_prec = z["precursor_per"].astype(float)
    Rp = group_drm(per_prec)
    lo_mask = GROUP[:-1] < 53
    hi_mask = ~lo_mask
    log = open(f"{out}/prec_summary.txt", "w")

    def say(*a):
        print(*a, flush=True)
        print(*a, file=log)

    # ---------- A. precursor
    for wins in ([(0, 9)], [(-1, 9)]):
        src, bkg, bvar, expo = counts(t, ch, wins)
        fl = flive_mean(wins)
        p, c2, g1, g2 = pl_index(Rp, elo, ehi, src, bkg, bvar, expo, fl)
        say(f"A {wins}: net={src.sum() - bkg.sum():.0f}  PL index {p[1]:.3f} [{g1:.3f},{g2:.3f}] (1sig) "
            f"chi2={c2:.1f}/{GROUP.size - 3}  F(0.3-3MeV)={efluxband(pl, p, 300, 3000):.3e} erg/cm2/s")
        for nm, m in (("ch20-53 (~0.2-0.6 MeV)", lo_mask), ("ch53-256 (~0.6-3.1 MeV)", hi_mask)):
            pp, cc, a1, a2 = pl_index(*sub(m, Rp, src, bkg, bvar)[:1], elo, ehi,
                                      *sub(m, Rp, src, bkg, bvar)[1:], expo, fl)
            say(f"   {nm}: PL index {pp[1]:.3f} [{a1:.3f},{a2:.3f}] chi2={cc:.1f}/{m.sum() - 2}")
        pc, cc = fit(cpl, [p[0], -1.6, np.log(1500)], Rp, elo, ehi, src, bkg, bvar, expo, fl,
                     bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))})
        say(f"   CPL: alpha={pc[1]:.3f} Ep={np.exp(pc[2]):.0f} chi2={cc:.1f}  (dchi2 vs PL {c2 - cc:.1f})")
        for name, (mdl, p0) in LIT.items():
            fixed = {i: v for i, v in enumerate(p0) if i > 0}
            q, cq = fit(mdl, [p[0]] + p0[1:], Rp, elo, ehi, src, bkg, bvar, expo, fl, fixed=fixed)
            say(f"   shape test {name}: chi2={cq:.1f}/{GROUP.size - 2} (dchi2 vs PL {cq - c2:+.1f}); "
                f"F(0.3-3MeV)={efluxband(mdl, q, 300, 3000):.3e}")
    # systematics on the 0-9 s index
    wins = [(0, 9)]
    src, bkg, bvar, expo = counts(t, ch, wins)
    fl = flive_mean(wins)
    src2, bkg2, bvar2, _ = counts(t, ch, wins, mode="tlin", bkg_order=2)
    src3, bkg3, bvar3, _ = counts(t, ch, wins, mode="proxy")
    variants = {"nominal": per_prec, "fwhm_orbit": z["prec_fwhm_orbit"], "b_lo": z["prec_b_lo"],
                "b_hi": z["prec_b_hi"], "theta+5": z["prec_theta5"],
                "gain+5%": gain_shift(per_prec, 1.05), "gain-5%": gain_shift(per_prec, 0.95)}
    for vn, per in variants.items():
        R = group_drm(np.asarray(per, float))
        p, c2 = fit(pl, [-3, -1.8], R, elo, ehi, src, bkg, bvar, expo, fl)
        plo, _ = fit(pl, [-3, -1.8], *sub(lo_mask, R, src, bkg, bvar)[:1], elo, ehi,
                     *sub(lo_mask, R, src, bkg, bvar)[1:], expo, fl)
        phi, _ = fit(pl, [-3, -1.8], *sub(hi_mask, R, src, bkg, bvar)[:1], elo, ehi,
                     *sub(hi_mask, R, src, bkg, bvar)[1:], expo, fl)
        say(f"A syst {vn:10s}: index {p[1]:.3f}  low {plo[1]:.3f}  high {phi[1]:.3f}  chi2 {c2:.1f}")
    p, c2 = fit(pl, [-3, -1.8], Rp, elo, ehi, src2, bkg2, bvar2, expo, fl)
    say(f"A syst bkg_tlin2 : index {p[1]:.3f}  chi2 {c2:.1f}")
    p, c2 = fit(pl, [-3, -1.8], Rp, elo, ehi, src3, bkg3, bvar3, expo, fl)
    say(f"A syst bkg_proxy : index {p[1]:.3f}  chi2 {c2:.1f}")
    # time-resolved
    for w in SEGS:
        src, bkg, bvar, expo = counts(t, ch, [w])
        fl = flive_mean([w])
        p, c2, g1, g2 = pl_index(Rp, elo, ehi, src, bkg, bvar, expo, fl)
        pc, cc = fit(cpl, [p[0], -1.6, np.log(1500)], Rp, elo, ehi, src, bkg, bvar, expo, fl,
                     bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))})
        say(f"A seg {w}: net={src.sum() - bkg.sum():.0f} PL {p[1]:.3f} [{g1:.3f},{g2:.3f}] chi2={c2:.1f}; "
            f"CPL a={pc[1]:.2f} Ep={np.exp(pc[2]):.0f} dchi2={c2 - cc:.1f}; F(0.3-3)={efluxband(pl, p, 300, 3000):.2e}")

    # ---------- B. bright clean intervals
    for name, wins in (("rise", [(180, 216)]), ("mid", [(288, 440)]), ("flare", [(440, 503)])):
        per = z[f"{name}_per"].astype(float)
        R = group_drm(per)
        src, bkg, bvar, expo = counts(t, ch, wins, mode="proxy")
        fl = flive_mean(wins)
        for sf in (0.0, 0.03):
            pc, cc = fit(cpl, [0, -1.5, np.log(1500)], R, elo, ehi, src, bkg, bvar, expo, fl,
                         bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))}, sys_frac=sf)
            pb, cb = fit(band, [pc[0], pc[1], pc[2], -2.6], R, elo, ehi, src, bkg, bvar, expo, fl,
                         bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5)), 3: (-6, -1.6)}, sys_frac=sf)
            say(f"B {name} sys={sf}: CPL a={pc[1]:.3f} Ep={np.exp(pc[2]):.0f} chi2={cc:.1f}/{GROUP.size - 4}; "
                f"Band a={pb[1]:.3f} Ep={np.exp(pb[2]):.0f} b={pb[3]:.2f} chi2={cb:.1f}/{GROUP.size - 5}; "
                f"F_CPL(0.3-3MeV)={efluxband(cpl, pc, 300, 3000):.3e}")
        p_ref, _ = fit(cpl, [0, -1.5, np.log(1500)], R, elo, ehi, src, bkg, bvar, expo, fl,
                       bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))})
        res = []
        for f in np.arange(0.90, 1.081, 0.01):
            Rf = group_drm(gain_shift(per, f))
            pc, cc = fit(cpl, list(p_ref), Rf, elo, ehi, src, bkg, bvar, expo, fl,
                         bounds={1: (-2.5, 1.5), 2: (np.log(30), np.log(1e5))})
            res.append((f, cc, pc[1], np.exp(pc[2])))
        res = np.array(res)
        k = np.argmin(res[:, 1])
        say(f"B {name} gain scan (CPL, no sys): best f={res[k, 0]:.2f} chi2={res[k, 1]:.1f} (f=1: "
            f"{res[np.argmin(np.abs(res[:, 0] - 1)), 1]:.1f}); a={res[k, 2]:.2f} Ep={res[k, 3]:.0f}")
        np.savetxt(f"{out}/gainscan_{name}.txt", res, fmt="%.3f", header="f chi2 alpha Ep")

    log.close()


if __name__ == "__main__":
    main()
