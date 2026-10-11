#!/usr/bin/env python3
"""GRB 221009A: expected change of the photon-number ratio
N(20-250 keV)/N(0.4-6 MeV) between phases, from published Band fits.

Photon-number ratio only (no detector response); a check of how well each
parameter set reproduces the GECAM LG photon ratio N(0.78-1.5)/N(0.38-0.78 MeV)
is printed alongside the measured ch10/ch9 (which includes the LaBr3 response,
so only the trend between phases is comparable).
Output: band_ratio.csv
"""
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).parent))
from p221009_scale_common import OUT

# (label, phase, alpha, beta, Epeak keV, source)
SETS = [
    ("KW 27-58",   "calib",  -1.17, -2.60, 1011, "Frederiks+23 T1"),
    ("GBM III",    "calib",  -1.158, -3.28, 1023, "Lesage+23"),
    ("KW 60",      "P1 rise", -1.18, -2.49, 2733, "Frederiks+23 T1"),
    ("KW 61",      "P1 peak", -0.76, -2.13, 3038, "Frederiks+23 T1"),
    ("An 227-231", "P1 peak", -0.76, -2.18, 1250, "An+23 table (mean of 4 s)"),
    ("KW 64",      "P2",     -0.97, -2.51, 1886, "Frederiks+23 T1"),
    ("An ~P2",     "P2",     -0.85, -2.25, 750,  "An+23 Fig. read-off"),
    ("An 280-290", "valley", -1.45, -2.57, 360,  "An+23 Fig. read-off"),
    ("Zh 280-285", "valley", -1.50, -2.35, 840,  "Zhang+24"),
    ("GBM Vc",     "flare",  -1.61, -2.46, 527,  "Lesage+23"),
    ("GBM VIa",    "flare",  -1.512, -2.55, 564, "Lesage+23"),
    ("GBM VIc",    "flare",  -1.484, -3.53, 1133, "Lesage+23"),
    ("GBM VII",    "flare",  -1.648, -2.23, 280, "Lesage+23"),
    ("An flare",   "flare",  -1.60, -2.40, 357,  "An+23 Eiso table Ep; alpha,beta assumed"),
]


def band(E, a, b, ep):
    ec = ep / (2 + a)
    eb = (a - b) * ec
    if E < eb:
        return (E / 100) ** a * np.exp(-E / ec)
    return (E / 100) ** b * ((a - b) * ec / 100) ** (a - b) * np.exp(b - a)


def nph(lo, hi, p):
    edges = np.geomspace(lo, hi, 40)
    return sum(quad(band, edges[i], edges[i + 1], args=p, limit=200)[0]
               for i in range(len(edges) - 1))


def main():
    rows = []
    for lab, ph, a, b, ep, src in SETS:
        p = (a, b, ep)
        r = nph(20, 250, p) / nph(400, 6000, p)
        hr = nph(780, 1500, p) / nph(380, 780, p)
        rows.append((lab, ph, a, b, ep, r, hr, src))
    ref = np.mean([r[5] for r in rows if r[1] == "calib"])
    print(f"{'set':<12}{'phase':<9}{'alpha':>7}{'beta':>7}{'Ep':>6}"
          f"{'N20-250/N0.4-6':>16}{'rel. calib':>11}{'N.78-1.5/N.38-.78':>19}")
    with open(OUT / "band_ratio.csv", "w") as f:
        f.write("set,phase,alpha,beta,Ep,ratio,rel_calib,hr_photon,source\n")
        for lab, ph, a, b, ep, r, hr, src in rows:
            print(f"{lab:<12}{ph:<9}{a:7.2f}{b:7.2f}{ep:6.0f}{r:16.2f}{r/ref:11.2f}{hr:19.3f}")
            f.write(f"{lab},{ph},{a},{b},{ep},{r:.4g},{r/ref:.4g},{hr:.4g},{src}\n")


if __name__ == "__main__":
    main()


def csi_variant():
    """Same ratio with the HE band replaced by CsI deposit-like bands."""
    print("\nHE band -> N(band)/N(0.4-6 MeV), relative to the calib mean")
    bands = [(20, 250), (40, 800), (200, 3000), (150, 3000), (300, 3000)]
    res = {}
    for lab, ph, a, b, ep, src in SETS:
        res[lab] = [nph(lo, hi, (a, b, ep)) / nph(400, 6000, (a, b, ep)) for lo, hi in bands]
    ref = np.mean([res[l] for l, ph, *_ in SETS if ph == "calib"], axis=0)
    print(f"{'set':<12}{'phase':<9}" + "".join(f"{f'{lo}-{hi}':>11}" for lo, hi in bands))
    with open(OUT / "band_ratio_csi.csv", "w") as f:
        f.write("set,phase," + ",".join(f"rel_{lo}_{hi}" for lo, hi in bands) + "\n")
        for lab, ph, *_ in SETS:
            rel = np.array(res[lab]) / ref
            print(f"{lab:<12}{ph:<9}" + "".join(f"{v:11.2f}" for v in rel))
            f.write(f"{lab},{ph}," + ",".join(f"{v:.4g}" for v in rel) + "\n")


if __name__ == "__main__" and "--csi" in sys.argv:
    csi_variant()
