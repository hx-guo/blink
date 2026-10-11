"""Step 3b: GBM quiescent interval (T0+10...176 s).

Light curves from the daily CTIME files (8 channels, 0.256 s -> 1.024 s),
with background models per channel:
  poly2/3 : polynomial over T0-400...-10 and T0+95...118 s (the precursor tail
            is visible in NaI to ~90 s);
  orb30/15: orbital background, mean of the same CTIME channel at +/-30
            (+/-15) orbits (Fitzpatrick+2011; Lesage+2023), shift found by
            matching the ground track over the window;
  orb*s   : orbital shape x constant scale fitted over T0-400...-10 s;
  orb*fit : orbital shape x (a + b t) over the polynomial windows (Lesage-like).
Background errors are split into a time-correlated part (fit parameters) and
an independent part (orbital counting noise).  Net counts / significance per
channel and in detector/energy groups for 121.219-164.228 s (Lesage region II)
and other windows; ratio to the precursor 1-s peak (spectral fits:
p221009_gbm_quietspec.py).
Outputs $GBM_OUT/quiet/.
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import p221009_gbm_common as C

od = C.outdir("quiet")
DAILY = C.DAILY
DETS = ["n3", "n4", "n6", "n7", "n8", "b0", "b1"]
DT = 1.024
EDGES = np.arange(-400.0, 800.0 + 1e-6, DT)
TC = 0.5 * (EDGES[:-1] + EDGES[1:])
POLY_WIN = [(-400.0, -10.0), (95.0, 118.0)]   # precursor tail is visible in NaI to ~90 s
PRE_WIN = [(-400.0, -10.0)]
DAY = C.DAY


P, shifts = C.orbital_shifts()
with open(os.path.join(od, "orbital_shifts.txt"), "w") as fo:
    fo.write("orbital period from ascending nodes: %.2f s\n" % P)
    for k, (g, d) in shifts.items():
        fo.write("k=%+d  shift=%.1f s (k*P%+.1f)  mean ground-track offset over window %.2f deg\n" % (k, g, g - k * P, d))
print(open(os.path.join(od, "orbital_shifts.txt")).read(), flush=True)


# ---------------------------------------------------------------- CTIME rates
def ctime(det, day):
    f = glob.glob(os.path.join(DAILY, "glg_ctime_%s_%s_v*.pha" % (det, day)))[0]
    with fits.open(f) as h:
        e = h["EBOUNDS"].data
        d = h["SPECTRUM"].data
        return (np.array(d["TIME"], float), np.array(d["ENDTIME"], float), np.array(d["COUNTS"], float),
                np.array(d["EXPOSURE"], float), np.array(e["E_MIN"]), np.array(e["E_MAX"]))


def rebin(ct, shift):
    """Counts and exposure on the 1.024 s grid (relative to T0) of the CTIME
    bins whose centre lies in each grid bin, for data taken at T0+t+shift."""
    t, te, c, x, _, _ = ct
    tr = 0.5 * (t + te) - C.T0 - shift
    m = (tr >= EDGES[0]) & (tr < EDGES[-1])
    idx = np.searchsorted(EDGES, tr[m]) - 1
    cnt = np.zeros((len(TC), c.shape[1])); expo = np.zeros(len(TC))
    np.add.at(cnt, idx, c[m]); np.add.at(expo, idx, x[m])
    return cnt, expo


def polyfit_w(t, y, var, win, order, basis=None):
    """Weighted least squares of y(t) on [basis * t^k] in the windows."""
    m = np.zeros(len(t), bool)
    for a, b in win:
        m |= (t >= a) & (t < b)
    m &= np.isfinite(y) & (var > 0)
    base = basis if basis is not None else np.ones_like(t)
    A = np.stack([base * (t / 100.0) ** k for k in range(order + 1)], -1)
    w = 1.0 / var[m]
    cov = np.linalg.inv(A[m].T @ (A[m] * w[:, None]))
    p = cov @ (A[m].T @ (y[m] * w))
    mod = A @ p
    err = np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A))
    chi2 = (((y - mod) ** 2 / var)[m]).sum()
    return mod, err, chi2, m.sum() - order - 1


rows, lcrows = [], {"t": TC}
WINDOWS = [("10-60", 10.0, 60.0), ("60-118", 60.0, 118.0), ("121-164(II)", 121.219, 164.228),
           ("150-170", 150.0, 170.0), ("164-174", 164.228, 174.0), ("20-170(HE)", 20.0, 170.0)]
PEAK = (0.0, 1.024)
models_all = {}
for det in DETS:
    ct0 = ctime(det, DAY[0])
    elo, ehi = ct0[4], ct0[5]
    c0, x0 = rebin(ct0, 0.0)
    orb = {}
    for k in [-30, 30, -15, 15]:
        ck, xk = rebin(ctime(det, DAY[k]), shifts[k][0])
        orb[k] = (ck, xk)
    for ch in range(1, 8):
        good = x0 > 0.5 * DT
        y = np.where(good, c0[:, ch] / np.maximum(x0, 1e-9), np.nan)
        var = np.where(good, np.maximum(c0[:, ch], 1) / np.maximum(x0, 1e-9) ** 2, np.nan)
        # each model: (rate, err_corr, err_ind, chi2, dof); err_corr is correlated
        # in time (fit-parameter error), err_ind independent per bin (orbital
        # counting noise)
        bk = {}
        zero = np.zeros_like(TC)
        m2, e2, c2, d2 = polyfit_w(TC, y, var, POLY_WIN, 2); bk["poly2"] = (m2, e2, zero, c2, d2)
        m3, e3, c3, d3 = polyfit_w(TC, y, var, POLY_WIN, 3); bk["poly3"] = (m3, e3, zero, c3, d3)
        for tag, ks in [("orb30", (-30, 30)), ("orb15", (-15, 15))]:
            cs = sum(orb[k][0][:, ch] for k in ks); xs = sum(orb[k][1] for k in ks)
            ok = xs > 0.5 * DT * len(ks)
            r = np.where(ok, cs / np.maximum(xs, 1e-9), np.nan)
            rerr = np.where(ok, np.sqrt(np.maximum(cs, 1)) / np.maximum(xs, 1e-9), np.nan)
            bk[tag] = (r, zero, rerr, np.nan, 0)
            # orbital shape x scale (pre-burst only) and x linear (both sides)
            for suf, win, order in [("s", PRE_WIN, 0), ("fit", POLY_WIN, 1)]:
                mod, err, chi2, dof = polyfit_w(TC, y, var + rerr ** 2, win, order, basis=r)
                scale = mod / np.where(r > 0, r, np.nan)
                bk[tag + suf] = (mod, err, rerr * np.abs(scale), chi2, dof)
        lcrows["%s_ch%d_rate" % (det, ch)] = y
        for tag, (mod, ecor, eind, chi2, dof) in bk.items():
            lcrows["%s_ch%d_%s" % (det, ch, tag)] = mod
            for wname, a, b in WINDOWS + [("peak0-1", *PEAK)]:
                m = (TC >= a) & (TC < b) & good & np.isfinite(mod)
                n = c0[m, ch].sum(); bexp = (mod[m] * x0[m]).sum()
                bvar = (ecor[m] * x0[m]).sum() ** 2 + ((eind[m] * x0[m]) ** 2).sum()
                net = n - bexp
                rows.append(dict(det=det, ch=ch, elo=elo[ch], ehi=ehi[ch], bkg=tag, window=wname,
                                 t1=a, t2=b, counts=n, bkg_counts=bexp, net=net, live=x0[m].sum(),
                                 net_rate=net / max(x0[m].sum(), 1e-9), bkg_var=bvar,
                                 sig_stat=net / np.sqrt(max(bexp, 1)),
                                 sig_tot=net / np.sqrt(max(bexp, 1) + bvar),
                                 chi2=chi2, dof=dof))
    print(det, "done", flush=True)

R = pd.DataFrame(rows)
C.write_csv(R, "quiet", "quiet_window_net.csv")
LC = pd.DataFrame(lcrows)
C.write_csv(LC, "quiet", "quiet_lc_ctime_1s.csv")

# grouped significance (channels/detectors independent)
GROUPS = [("NaI n4+n8 27-300 keV", ["n4", "n8"], [2, 3, 4]), ("NaI n4+n8 300-1000 keV", ["n4", "n8"], [5, 6]),
          ("NaI n3+n4+n6+n7+n8 27-300 keV", ["n3", "n4", "n6", "n7", "n8"], [2, 3, 4]),
          ("BGO b1 0.38-2.1 MeV", ["b1"], [1, 2]), ("BGO b1 2.1-10 MeV", ["b1"], [3, 4]),
          ("BGO b0+b1 0.38-2.1 MeV", ["b0", "b1"], [1, 2])]
grow = []
for (wname, bkg), g in R.groupby(["window", "bkg"]):
    for gname, dets, chans in GROUPS:
        h = g[g.det.isin(dets) & g.ch.isin(chans)]
        var = h.bkg_counts.clip(lower=1).sum() + h.bkg_var.sum()
        live = h.groupby("det").live.first().mean()
        grow.append(dict(window=wname, bkg=bkg, group=gname, net=h.net.sum(), bkg_counts=h.bkg_counts.sum(),
                         sigma=np.sqrt(var), signif=h.net.sum() / np.sqrt(var),
                         net_rate_per_det=h.net.sum() / live / len(dets),
                         ul2_rate_per_det=(max(h.net.sum(), 0) + 2 * np.sqrt(var)) / live / len(dets)))
G = pd.DataFrame(grow)
C.write_csv(G, "quiet", "quiet_group_signif.csv")

# ratio of quiet-period net rate to the precursor 1-s peak, per channel
pk = R[R.window == "peak0-1"].set_index(["det", "ch", "bkg"]).net_rate
qq = R[R.window == "121-164(II)"].set_index(["det", "ch", "bkg"])
qq = qq.assign(peak_rate=pk.reindex(qq.index).values)
qq["ratio_to_peak"] = qq.net_rate / qq.peak_rate
C.write_csv(qq.reset_index(), "quiet", "quiet_vs_peak.csv")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
panels = [("n8", c) for c in (1, 2, 3, 4, 5, 6)] + [("b1", 1), ("b1", 2)]
sm = 8
for ax, (det, ch) in zip(axs.T.ravel(), panels):
    y = LC["%s_ch%d_rate" % (det, ch)]
    ys = np.convolve(np.nan_to_num(y), np.ones(sm) / sm, "same")
    ax.plot(TC, ys, color="k", lw=0.7, label="data (8 s smooth)")
    for tag, col in [("poly2", "C0"), ("orb30", "C1"), ("orb30s", "C2"), ("orb30fit", "C3")]:
        ax.plot(TC, np.convolve(np.nan_to_num(LC["%s_ch%d_%s" % (det, ch, tag)]), np.ones(sm) / sm, "same"),
                color=col, lw=0.8, label=tag)
    r = R[(R.det == det) & (R.ch == ch) & (R.bkg == "poly2")].iloc[0]
    ax.set_title("%s ch%d %.0f-%.0f keV" % (det, ch, r.elo, r.ehi), fontsize=8)
    ax.axvspan(121.219, 164.228, color="y", alpha=0.2)
    ax.set_xlim(-100, 200)
    m = (TC > -100) & (TC < 172)
    lo, hi = np.nanpercentile(ys[m], [1, 99]); ax.set_ylim(lo * 0.97, hi * 1.03)
axs[0, 0].legend(fontsize=6)
for a in axs[-1]:
    a.set_xlabel("t - T0(GBM) (s)")
fig.tight_layout(); fig.savefig(os.path.join(od, "quiet_lc.png"), dpi=130)

print("done")
