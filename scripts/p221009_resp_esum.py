"""Energy-axis stacked background spectra and line centroids (GRB 221009A
response study).

Each detector's ACD-clean CsI (pw 76-120) spectrum is mapped onto a common
deposited-energy grid with that detector's EC (normal: 2022-08-26 Nor, low
gain: 2022-10-04 LHV; uniform redistribution within a channel), and the 18
detectors are summed.  If the per-detector EC are right, lines stay sharp and
sit at their nominal energies.  Line centroids are fitted (Gaussian on a
quadratic-in-log continuum) near 511, 690, 1369 and 2754 keV.

    python p221009_resp_esum.py [OUTDIR]"""
import sys

import numpy as np

import matplotlib.pyplot as plt

from p221009_resp_lines import EC, gfit, residual

EGRID = {0: np.arange(60, 900, 4.0), 1: np.arange(150, 3500, 10.0)}
LINES = {0: (190, 511, 690), 1: (511, 690, 1369, 2754)}


def to_energy(spec, k, b, edges):
    ch = np.arange(spec.size)
    lo, hi = k * (ch - 0.5) + b, k * (ch + 0.5) + b
    cum = np.concatenate([[0], np.cumsum(spec)])
    ce = np.interp(edges, np.concatenate([[lo[0]], hi]), cum)
    return np.diff(ce)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/p221009/resp/out"
    z = np.load(f"{out}/bkg_summed.npz")
    hist, expo = z["hist"], z["expo"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for m in (0, 1):
        edges = EGRID[m]
        mid = 0.5 * (edges[1:] + edges[:-1])
        tot = np.zeros(mid.size)
        for d in range(18):
            s = hist[m, 1, 2, d].astype(float)
            s[:21] = 0                     # below threshold / first channels
            s[270:] = 0                    # upper-threshold roll-off and wrap edge
            tot += to_energy(s, EC[m][d, 0], EC[m][d, 1], edges)
        r = residual(tot, win=21)
        ax[m, 0].semilogy(mid, tot / expo[m]); ax[m, 0].set_title(f"{'NG' if m == 0 else 'LG'} CsI, EC-stacked")
        ax[m, 1].plot(mid, r); ax[m, 1].set_title("log excess over running median")
        for E in LINES[m]:
            ax[m, 1].axvline(E, ls=":", c="gray")
            step = edges[1] - edges[0]
            res = gfit(mid, tot, E, max(0.12 * E, 6 * step))
            if res is None:
                print(f"{'NG' if m == 0 else 'LG'} line {E}: fit failed")
                continue
            mu, emu, sg, a, chi = res
            print(f"{'NG' if m == 0 else 'LG'} line {E} keV: centroid {mu:.1f} +- {emu:.1f} keV "
                  f"(rel {mu / E - 1:+.4f}), sigma {sg:.1f} keV (FWHM/E {2.355 * sg / mu:.3f}), "
                  f"chi2/dof {chi:.1f}", flush=True)
        for a in ax[m]:
            a.set_xscale("log"); a.set_xlabel("deposited energy (keV)")
    fig.tight_layout(); fig.savefig(f"{out}/bkg_esum.png", dpi=110)
    np.savez_compressed(f"{out}/bkg_esum.npz", **{f"m{m}_edges": EGRID[m] for m in (0, 1)})


if __name__ == "__main__":
    main()
