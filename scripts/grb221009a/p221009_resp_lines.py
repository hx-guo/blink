"""Background-line energy scale check for HE low gain (GRB 221009A response).

Reads bkgspec_*.npz (p221009_resp_bkgspec.py), sums the days, and for each
gain mode (NG/LG) and pulse-width class (NaI 54-70, CsI 76-120) looks for
spectral features as excesses over a broad running continuum.  Features are
then fitted per detector (Gaussian + quadratic in log counts) to get channel
centroids, which are converted to deposited energy with the group's EC files
(normal: last Nor before 2022-10-09; low gain: 2022-10-04 LHV).

    python p221009_resp_lines.py [OUTDIR]"""
import glob
import os
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

LIB = "/hxmtfs2/work/GRB/Software/RSPgenerator_v1/EC_FWHM"
EC = {0: np.loadtxt(f"{LIB}/2022-08-26_EC_Nor.txt", usecols=[1, 2, 3, 4]),
      1: np.loadtxt(f"{LIB}/2022-10-04_EC_LHV.txt", usecols=[1, 2, 3, 4])}
MODE = {0: "NG", 1: "LG"}
PWC = {0: "NaI 54-70", 2: "CsI 76-120", 3: "CsI 121-255"}


def load(out):
    files = sorted(glob.glob(f"{out}/bkgspec_2022*.npz"))
    hist = None
    expo = np.zeros(2)
    for f in files:
        z = np.load(f)
        hist = z["hist"].astype(float) if hist is None else hist + z["hist"]
        secs = z["secs"]
        expo += [np.sum(secs[:, 1] == 0), np.sum(secs[:, 1] == 1)]
    return files, hist, expo


def residual(spec, win=31):
    """log excess over a running median continuum (feature finder)."""
    ls = np.log(np.clip(spec, 1, None))
    return ls - median_filter(ls, size=win, mode="nearest")


def gfit(ch, y, c0, half):
    s = (ch >= c0 - half) & (ch <= c0 + half) & (y > 0)
    x, yy = ch[s], y[s]
    if x.size < 7:
        return None

    def f(x, a, mu, sg, p0, p1, p2):
        u = x - c0
        return a * np.exp(-0.5 * ((x - mu) / sg) ** 2) + np.exp(p0 + p1 * u + p2 * u * u)

    ly = np.log(yy)
    p = np.polyfit(x - c0, ly, 2)
    amp = max(yy[np.argmin(np.abs(x - c0))] - np.exp(np.polyval(p, 0)), yy.max() * 0.05)
    try:
        popt, pcov = curve_fit(f, x, yy, p0=[amp, c0, half / 3, p[2], p[1], p[0]],
                               sigma=np.sqrt(yy), absolute_sigma=True, maxfev=20000,
                               bounds=([0, c0 - half, 0.5, -np.inf, -np.inf, -np.inf],
                                       [np.inf, c0 + half, half * 1.5, np.inf, np.inf, np.inf]))
    except Exception:
        return None
    e = np.sqrt(np.diag(pcov))
    chi2 = np.sum(((f(x, *popt) - yy) / np.sqrt(yy)) ** 2) / max(x.size - 6, 1)
    return popt[1], e[1], popt[2], popt[0], chi2


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/p221009/resp/out"
    files, hist, expo = load(out)
    print("days:", [os.path.basename(f) for f in files], "exposure NG/LG (s):", expo)
    ch = np.arange(hist.shape[-1])
    # 1. summed spectra and feature residuals
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    lines = []
    for j, pc in enumerate((0, 2, 3)):
        for m in (0, 1):
            spec = hist[m, :, pc].sum(axis=(0, 1)) / max(expo[m], 1)   # both ACD classes
            spec_v = hist[m, 1, pc].sum(axis=0) / max(expo[m], 1)      # ACD-clean
            r = residual(spec_v * expo[m])
            a = ax[j, 0]
            a.semilogy(ch, spec, label=f"{MODE[m]} all", lw=0.8)
            a.semilogy(ch, spec_v, label=f"{MODE[m]} ACD-clean", lw=0.8, ls="--")
            ax[j, 1].plot(ch, r, label=MODE[m], lw=0.8)
            for c in range(25, 270):
                if r[c] == r[c - 6:c + 7].max() and r[c] > 0.03:
                    lines.append((MODE[m], PWC[pc], c, round(float(r[c]), 3)))
        ax[j, 0].set_title(f"{PWC[pc]} (counts/s/ch, 18 det)")
        ax[j, 1].set_title(f"{PWC[pc]}: log excess over 31-ch running median (ACD-clean)")
        ax[j, 0].legend(fontsize=7); ax[j, 1].legend(fontsize=7)
        ax[j, 1].axhline(0, c="k", lw=0.5)
        for a in ax[j]:
            a.set_xlim(15, 290)
    ax[-1, 0].set_xlabel("unwrapped channel"); ax[-1, 1].set_xlabel("unwrapped channel")
    fig.tight_layout(); fig.savefig(f"{out}/bkg_lines_overview.png", dpi=110)
    with open(f"{out}/bkg_lines_candidates.txt", "w") as fo:
        for l in lines:
            print(*l, file=fo)
            print("candidate", *l)
    # 2. per-detector stacked residual images (gain spread check) for CsI
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for m in (0, 1):
        for j, pc in enumerate((0, 2)):
            img = np.array([residual(hist[m, 1, pc, d]) for d in range(18)])
            ax[m, j].imshow(img, aspect="auto", origin="lower", vmin=-0.2, vmax=0.2,
                            extent=[0, hist.shape[-1], -0.5, 17.5], cmap="RdBu_r")
            ax[m, j].set_title(f"{MODE[m]} {PWC[pc]} per-det residual (ACD-clean)")
            ax[m, j].set_xlim(15, 290)
    fig.tight_layout(); fig.savefig(f"{out}/bkg_lines_perdet.png", dpi=110)
    np.savez_compressed(f"{out}/bkg_summed.npz", hist=hist, expo=expo)


if __name__ == "__main__":
    main()
