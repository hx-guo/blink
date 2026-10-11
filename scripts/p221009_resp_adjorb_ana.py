"""Adjacent-orbit background, step 3: quiet-period HE/CsI net counts and
spectra, null tests, comparison with the other background models, and the
GBM quiet-period spectra folded through the HE DRM (GRB 221009A).

Inputs: out/adjorb/hist_{burst,d1008,d1007,d1003}.npz (step 2).
References: d1008 = -15 orbits (2022-10-08), d1007 = -30 orbits (10-07):
both low gain for the whole window, track offset 1.8 / 3.7 deg, same
Galactic-plane scan (Earth-angle difference rms 7-8 deg).  d1003 (-89
orbits, low gain, track 2.0 deg, other pointing; Earth-angle rms 25 deg) is
used only as a cross-check.

Background models for the burst day (per channel group, per second):
  orb_raw   mean of d1008 and d1007, unscaled
  orb_s     orb_raw x constant per group, from T0-400...-10 s
  orb_fit   orb_raw x (a + b t) per group, from T0-400...-10 and 1900...2580 s
            (source Earth-occulted from T0+1884 s)
  proxy / proxypre / tloc   as in p221009_resp_quiet.py / gbm.md 6.3

    python p221009_resp_adjorb_ana.py [OUTDIR]"""
import csv
import sys

import numpy as np

import matplotlib.pyplot as plt

from p221009_resp_spec import GROUP, band, cpl, fit, group_drm, largeevt, photon_flux, pl
from p221009_resp_prec import efluxband

BASE = "/scratchfs2/gecam/guohx/p221009"
AD = f"{BASE}/resp/out/adjorb"
GBMQ = f"{BASE}/gbm/runs/qspec1/quiet/quietspec_fits.csv"
REFS = ("d1008", "d1007")
PRE = (-400, -10)
OCC = (1900, 2580)
WINS = {"20-175": (20, 175), "60-118": (60, 118), "115-172": (115, 172),
        "II 121.16-164.17": (121.159, 164.168), "150-172": (150, 172)}


def load(label):
    z = np.load(f"{AD}/hist_{label}.npz")
    h = z["h"].astype(float)
    tr = z["t_rel"]
    g = np.array([h[:, GROUP[i]:GROUP[i + 1]].sum(1) for i in range(GROUP.size - 1)]).T   # sec x 27
    return tr, g, z["hv"]


def sel(tr, w):
    return (tr >= w[0]) & (tr < w[1])


def wsum(tr, x, w):
    """Sum over [w0, w1) with fractional edge seconds."""
    lo, hi = tr, tr + 1
    f = np.clip(np.minimum(hi, w[1]) - np.maximum(lo, w[0]), 0, 1)
    return (f[:, None] * x).sum(0) if x.ndim == 2 else (f * x).sum()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    tr, gb, hvb = load("burst")
    refs = {r: load(r)[1:] for r in REFS + ("d1003",)}
    for r, (g, hv) in refs.items():
        print(f"{r}: LG seconds {np.mean((hv > -880) & (hv < -760)):.3f}, HV-off {np.sum(hv > -700)}, "
              f"rate pre {g[sel(tr, PRE)].sum() / np.ptp(PRE):.0f} /s (burst {gb[sel(tr, PRE)].sum() / np.ptp(PRE):.0f})")
    gr = np.mean([refs[r][0] for r in REFS], axis=0)          # orb_raw (sec x 27)
    nref = len(REFS)
    lines = []

    def scale_const(g_ref, cal):
        m = np.zeros(tr.size, bool)
        for w in cal:
            m |= sel(tr, w)
        return g_ref * (gb[m].sum(0) / g_ref[m].sum(0))[None, :]

    def scale_lin(g_ref, cal):
        m = np.zeros(tr.size, bool)
        for w in cal:
            m |= sel(tr, w)
        out_ = np.empty_like(g_ref)
        for k in range(g_ref.shape[1]):
            # gb ~ (a + b t) g_ref  ->  least squares in a, b
            A = np.column_stack([g_ref[m, k], g_ref[m, k] * tr[m] / 1000.0])
            c = np.linalg.lstsq(A, gb[m, k], rcond=None)[0]
            out_[:, k] = g_ref[:, k] * (c[0] + c[1] * tr / 1000.0)
        return out_

    # other models (total-count level, ch 20-255)
    lm = largeevt()
    x = np.array([lm.get(int(k), np.nan) for k in tr])
    ok = np.isfinite(x)
    print(f"LargeEvt missing seconds: {np.sum(~ok)} (interpolated)")
    x = np.interp(tr, tr[ok], x[ok])
    tot_b = gb.sum(1)

    def proxy(cal):
        m = np.zeros(tr.size, bool)
        for w in cal:
            m |= sel(tr, w)
        m &= np.isfinite(x)
        c = np.polyfit(x[m], tot_b[m], 1)
        return np.polyval(c, x)

    def tloc():
        m = sel(tr, (-101, -6))
        c = np.polyfit(tr[m] + 0.5, tot_b[m], 1)
        return np.polyval(c, tr + 0.5)

    models_tot = {
        "orb_raw": gr.sum(1),
        "orb_s": scale_const(gr, [PRE]).sum(1),
        "orb_fit": scale_lin(gr, [PRE, OCC]).sum(1),
        "orb_s d1008 only": scale_const(refs["d1008"][0], [PRE]).sum(1),
        "orb_s d1007 only": scale_const(refs["d1007"][0], [PRE]).sum(1),
        "orb_s d1003 only (other pointing)": scale_const(refs["d1003"][0], [PRE]).sum(1),
        "proxy (pre+occ, = v3 family)": proxy([(-420, -10), OCC]),
        "proxypre (pre only)": proxy([(-420, -10)]),
        "tloc (-101..-6 linear)": tloc(),
    }
    # ---- null tests: (a) pre-burst 10-s windows predicted with scale fitted without them
    lines.append("NULL TESTS (ch 20-255 total counts)")
    vb = [(a, a + 10) for a in range(-100, -10, 10)]
    for name, fn in (("orb_s", lambda cal: scale_const(gr, cal)), ("orb_fit", lambda cal: scale_lin(gr, cal))):
        res = []
        for w in vb:
            cal = [(-400, w[0] - 5)] if name == "orb_s" else [(-400, w[0] - 5), OCC]
            b = fn(cal).sum(1)
            res.append(wsum(tr, tot_b - b, w))
        stat = np.sqrt(np.mean([wsum(tr, tot_b, w) for w in vb]))
        lines.append(f"  pre-burst 10-s windows ({name}, refit without window): mean {np.mean(res):+.0f}, "
                     f"rms {np.sqrt(np.mean(np.square(res))):.0f}, Poisson {stat:.0f} (x sqrt(1+1/{nref}) with ref noise "
                     f"{stat * np.sqrt(1 + 1 / nref):.0f})")
    # (b) Earth-occulted window predicted from pre-burst scaling only (2000-s extrapolation)
    for name, b in (("orb_raw", gr.sum(1)), ("orb_s", scale_const(gr, [PRE]).sum(1)),
                    ("proxypre", models_tot["proxypre (pre only)"]), ("tloc", models_tot["tloc (-101..-6 linear)"])):
        for w in ((1900, 2240), (2240, 2580)):
            n = wsum(tr, tot_b - b, w)
            s = np.sqrt(wsum(tr, tot_b, w) * (1 + 1 / nref))
            lines.append(f"  occulted {w} predicted by {name} (no occulted data used): net {n:+.0f} +- {s:.0f} "
                         f"({n / s:+.1f} sig; {n / np.ptp(w):+.1f} /s)")
    # 10-s residual time series for the record
    b_s = scale_const(gr, [PRE]).sum(1)
    b_f = scale_lin(gr, [PRE, OCC]).sum(1)
    ts = [(a, wsum(tr, tot_b - b_s, (a, a + 10)), wsum(tr, tot_b - b_f, (a, a + 10)),
           wsum(tr, tot_b - models_tot["proxy (pre+occ, = v3 family)"], (a, a + 10)),
           wsum(tr, tot_b - models_tot["proxypre (pre only)"], (a, a + 10)))
          for a in range(-400, 2620, 10)]
    np.savetxt(f"{out}/adjorb_net_10s.txt", ts, fmt="%.0f %.1f %.1f %.1f %.1f",
               header="t_start net_orb_s net_orb_fit net_proxy net_proxypre (CsI ch20-255 counts/10 s)")
    # ---- quiet-period net counts, all models
    lines.append("QUIET-PERIOD NET COUNTS (CsI ch 20-255; stat includes reference noise for orb models)")
    for wn, w in WINS.items():
        for name, b in models_tot.items():
            n = wsum(tr, tot_b - b, w)
            s = np.sqrt(wsum(tr, tot_b, w) * ((1 + 1 / nref) if name.startswith("orb") else 1))
            lines.append(f"  {wn:18s} {name:34s}: {n:+7.0f} +- {s:5.0f} ({n / s:+.1f} sig) = {n / np.ptp(w):+.1f} counts/s")
    # ---- spectrum in the quiet window with orb_fit / orb_s
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    R = group_drm(z["rise_per"].astype(float))     # theta 12.7 (direction at ~170 s)
    rows = list(csv.DictReader(open(GBMQ)))
    lines.append("SPECTRA AND GBM COMPARISON")
    for wn in ("115-172", "II 121.16-164.17", "20-175"):
        w = WINS[wn]
        expo = np.ptp(w)
        for bn, bg in (("orb_fit", scale_lin(gr, [PRE, OCC])), ("orb_s", scale_const(gr, [PRE]))):
            src = wsum(tr, gb, w)
            bkg = wsum(tr, bg, w)
            bvar = bkg / nref                            # reference Poisson noise
            net = src - bkg
            pp, c2 = fit(pl, [np.log(1e-3), -2.0], R, elo, ehi, src, bkg, bvar, expo, 1.0, bounds={1: (-5, 0)})
            # chi2 of zero signal
            c0 = np.sum(net ** 2 / (src + bvar))
            lines.append(f"  {wn} {bn}: net {net.sum():+.0f}; chi2(no signal) {c0:.1f}/27; PL index {pp[1]:.2f} "
                         f"chi2 {c2:.1f}/25; F(0.3-3 MeV) {efluxband(pl, pp, 300, 3000):.2e}")
            # GBM models
            gw = {"II 121.16-164.17": ("121.219", "164.228"), "20-175": ("20", "170"), "115-172": ("121.219", "164.228")}[wn]
            for r in rows:
                if r["t1"] != gw[0] or r["t2"] != gw[1] or r["dets"] != "n4+n8+b1":
                    continue
                try:
                    if r["model"] == "PL":
                        mdl, p = pl, [np.log(float(r["A"])), float(r["index"])]
                    elif r["model"] == "COMP":
                        mdl, p = cpl, [np.log(float(r["A"])), float(r["index"]), np.log(float(r["Epeak"]))]
                    elif r["model"] == "Band":
                        mdl, p = band, [np.log(float(r["A"])), float(r["alpha"]), np.log(float(r["Epeak"])), float(r["beta"])]
                    else:
                        continue
                except (KeyError, ValueError):
                    continue
                mu = photon_flux(mdl, p, elo, ehi) @ R * expo
                cm = np.sum((net - mu) ** 2 / (src + bvar))
                q, cq = fit(mdl, p, R, elo, ehi, src, bkg, bvar, expo, 1.0, fixed={i: v for i, v in enumerate(p) if i > 0})
                lines.append(f"     GBM {r['model']:5s} ({r['t1']}-{r['t2']}): predicted HE {mu.sum():6.0f} counts (scaled to {wn}); "
                             f"chi2 at GBM norm {cm:.1f}/27, HE/GBM norm {np.exp(q[0] - p[0]):.2f} (chi2 {cq:.1f}/26)")
    with open(f"{out}/adjorb_summary.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            print(l, file=f)
    # figure: 10-s residual series
    ts = np.array(ts)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    for k, lab in ((1, "orb_s"), (2, "orb_fit"), (3, "proxy"), (4, "proxypre")):
        ax[0].plot(ts[:, 0] + 5, ts[:, k] / 10, label=lab, lw=0.8)
        ax[1].plot(ts[:, 0] + 5, ts[:, k] / 10, label=lab, lw=0.8)
    ax[0].set_xlim(-400, 2620); ax[0].set_ylim(-300, 600)
    ax[1].set_xlim(-100, 180); ax[1].set_ylim(-200, 400)
    for a in ax:
        a.axhline(0, c="k", lw=0.5); a.set_ylabel("net CsI counts/s"); a.legend(fontsize=7)
        a.axvspan(1884, 2620, color="gray", alpha=0.2)
    ax[1].set_xlabel("t - T0 (s)")
    fig.tight_layout(); fig.savefig(f"{out}/adjorb_net.png", dpi=110)


if __name__ == "__main__":
    main()
