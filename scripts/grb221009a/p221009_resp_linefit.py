"""Per-detector background-feature centroids in normal and low gain, and an
independent low-gain energy scale (GRB 221009A response study).

Input: bkg_summed.npz (p221009_resp_lines.py).  ACD-clean spectra.

CsI (pw 76-120): three features are seen in normal gain near 190, 490 and
690 keV (per-detector normal EC) and two in low gain near 470 and 680 keV
(low-gain EC), plus a weak one near 2.7 MeV.  Each feature is fitted per
detector with a Gaussian on a quadratic-in-log continuum.  The normal-gain
energies (normal EC is anchored on the on-board 241Am source) then calibrate
the low-gain channels without using the group's LHV EC: a two-point line
through the ~490 and ~690 keV features per detector.

NaI (pw 54-70): normal-gain features near ch 50 and 150 (CALDB NaI gain,
third segment) and low-gain features near ch 41 and 99; the low-gain NaI
scale follows from matching ch 41 to the normal-gain 150-channel feature and
ch 99 to the CsI ~490 keV feature (the same line in both crystals).

    python p221009_resp_linefit.py [OUTDIR]"""
import sys

import numpy as np
from astropy.io import fits

from p221009_resp_lines import EC, gfit

GAIN = "/cvmfs/hxmt.ihep.ac.cn/hxmtsoft/CALDB/data/hxmt/he/bcf/hxmt_he_gain_20171030_v1.fits"


def nai_gain():
    with fits.open(GAIN) as h:
        d = h[1].data
        return [(np.array(d["GC1"][i]), np.array(d["GC2"][i]), np.array(d["GC3"][i])) for i in range(18)]


def nai_energy(g, ch, seg):
    return g[0][seg] * ch * ch + g[1][seg] * ch + g[2][seg]


def fit_feature(spec, c0, half):
    r = gfit(np.arange(spec.size), spec, c0, half)
    if r is None:
        return (np.nan,) * 5
    return r


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/p221009/resp/out"
    z = np.load(f"{out}/bkg_summed.npz")
    hist = z["hist"]                      # mode, acd, pwclass, det, ch
    rows = []
    ng_csi_E = (190.0, 490.0, 690.0)
    lg_csi_E = (470.0, 680.0, 2700.0)
    for d in range(18):
        kN, bN = EC[0][d, :2]
        kL, bL = EC[1][d, :2]
        sN = hist[0, 1, 2, d].astype(float)
        sL = hist[1, 1, 2, d].astype(float)
        r = dict(det=d)
        for i, E in enumerate(ng_csi_E):
            c0 = (E - bN) / kN
            mu, emu, sg, a, chi = fit_feature(sN, c0, max(6, 0.12 * c0))
            r.update({f"N{i}_ch": mu, f"N{i}_err": emu, f"N{i}_sig": sg, f"N{i}_chi2": chi,
                      f"N{i}_E": kN * mu + bN})
        for i, E in enumerate(lg_csi_E):
            c0 = (E - bL) / kL
            mu, emu, sg, a, chi = fit_feature(sL, c0, max(5, 0.12 * c0))
            r.update({f"L{i}_ch": mu, f"L{i}_err": emu, f"L{i}_sig": sg, f"L{i}_chi2": chi,
                      f"L{i}_E": kL * mu + bL})
        # independent low-gain line through the two matched features
        x1, x2 = r["L0_ch"], r["L1_ch"]
        y1, y2 = r["N1_E"], r["N2_E"]
        k_ind = (y2 - y1) / (x2 - x1)
        r.update(k_ind=k_ind, b_ind=y1 - k_ind * x1, k_ec=kL, b_ec=bL,
                 ratio_ch_490=r["N1_ch"] / x1, ratio_ch_690=r["N2_ch"] / x2)
        rows.append(r)
    g = nai_gain()
    for d in range(18):
        sN = hist[0, 1, 0, d].astype(float)
        sL = hist[1, 1, 0, d].astype(float)
        a = fit_feature(sN, 50, 8)
        b = fit_feature(sN, 150, 14)
        c = fit_feature(sL, 41, 7)
        e = fit_feature(sL, 99, 12)
        EnA, EnB = nai_energy(g[d], a[0], 1), nai_energy(g[d], b[0], 2)
        E490 = rows[d]["N1_E"]
        k = (E490 - EnB) / (e[0] - c[0])
        rows[d].update(nN0_ch=a[0], nN0_E=EnA, nN1_ch=b[0], nN1_E=EnB, nL0_ch=c[0], nL0_err=c[1],
                       nL1_ch=e[0], nL1_err=e[1], nai_k_lg=k, nai_b_lg=EnB - k * c[0],
                       nN1_chi2=b[4], nL0_chi2=c[4], nL1_chi2=e[4])
    keys = list(rows[0].keys())
    with open(f"{out}/linefit_perdet.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4g}" for k in keys) + "\n")
    A = {k: np.array([r[k] for r in rows], float) for k in keys}
    med = lambda x: (np.nanmedian(x), np.nanstd(x))  # noqa: E731
    print("CsI NG feature energies (normal EC), median/std over det:",
          [tuple(np.round(med(A[f"N{i}_E"]), 1)) for i in range(3)])
    print("CsI LG feature energies (LHV EC):", [tuple(np.round(med(A[f"L{i}_E"]), 1)) for i in range(3)])
    print("LG(EC)-NG(EC) energy of matched features: 490:", np.round(med(A["L0_E"] - A["N1_E"]), 1),
          " 690:", np.round(med(A["L1_E"] - A["N2_E"]), 1))
    print("relative: 490:", np.round(med(A["L0_E"] / A["N1_E"] - 1), 4), " 690:",
          np.round(med(A["L1_E"] / A["N2_E"] - 1), 4))
    print("independent LG k (keV/ch):", np.round(med(A["k_ind"]), 3), " EC k:", np.round(med(A["k_ec"]), 3),
          " k_ind/k_ec:", np.round(med(A["k_ind"] / A["k_ec"]), 4))
    print("independent LG b:", np.round(med(A["b_ind"]), 1), " EC b:", np.round(med(A["b_ec"]), 1))
    for ch in (20, 30, 53, 99, 224, 255, 275):
        Ei = A["k_ind"] * ch + A["b_ind"]
        Ee = A["k_ec"] * ch + A["b_ec"]
        print(f"ch {ch}: E_ind {np.nanmedian(Ei):.0f} (det spread {np.nanstd(Ei):.0f})  E_EC {np.nanmedian(Ee):.0f}"
              f"  ratio {np.nanmedian(Ei / Ee):.3f}")
    print("channel ratio NG/LG at 490 / 690:", np.round(med(A["ratio_ch_490"]), 3), np.round(med(A["ratio_ch_690"]), 3))
    print("L2 (~2.7 MeV, LHV EC) E:", np.round(med(A["L2_E"]), 0), " with independent scale:",
          np.round(med(A["k_ind"] * A["L2_ch"] + A["b_ind"]), 0))
    print("NaI NG features (CALDB): ch", np.round(med(A["nN0_ch"]), 1), "->", np.round(med(A["nN0_E"]), 1),
          " ch", np.round(med(A["nN1_ch"]), 1), "->", np.round(med(A["nN1_E"]), 1))
    print("NaI LG features: ch", np.round(med(A["nL0_ch"]), 1), np.round(med(A["nL1_ch"]), 1),
          " LG NaI k,b:", np.round(med(A["nai_k_lg"]), 2), np.round(med(A["nai_b_lg"]), 1))
    for ch in (20, 54, 100, 150, 200, 255):
        E = A["nai_k_lg"] * ch + A["nai_b_lg"]
        print(f"  NaI LG ch {ch}: {np.nanmedian(E):.0f} keV (spread {np.nanstd(E):.0f})")
    print("fit chi2/dof medians:", {k: round(float(np.nanmedian(A[k])), 2) for k in keys if k.endswith("chi2")})


if __name__ == "__main__":
    main()
