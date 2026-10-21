"""Referee round 2, R2: is the HE/GBM area ratio 0.81 at 180-216 s a dead-time
or FIFO artefact?  (GRB 221009A response study)

The earlier comparison (p221009_resp_gbmmain.py) used observed 1K events over
180-216 s divided by the mean box live fraction (0.966).  But 186-192 s and
215 s have FIFO losses (gap-fillers in the recovery: ~9.7e4 events), which
the observed events do not contain.  Here, for the same GBM fits (gdt,
n4+n8+b1), the HE side is redone three ways:
  clean   observed events only in the loss-free seconds 180-186 and 193-215 s
          (GBM fitted on the same seconds), rate-weighted live fraction
  rec     recovered events (observed + fillers, scale/hxmt_events.npz) over
          180-216 s, rate-weighted live fraction
  obs     the original (observed only over 180-216 s), for reference
and a response tilt delta (HE/GBM ~ C (E/1 MeV)^delta in deposited energy)
is fitted together with C.

Run with the gdt venv: resp/run_gdt.sh p221009_resp_r2.py"""
import csv
import sys

import numpy as np

sys.path.insert(0, "/scratchfs2/gecam/guohx/p221009/gbm/scripts")
from p221009_gbm_specfit import DetData, fit as gbm_fit   # noqa: E402

from p221009_resp_spec import GROUP, band, counts, cpl, group_drm, load_events, photon_flux  # noqa: E402

BASE = "/scratchfs2/gecam/guohx/p221009"
DT = 0.06
BKG = ([(-250.0, -10.0), (95.0, 118.0), (720.0, 790.0)], 2)
DETS = ("n4", "n8", "b1")
EC = np.loadtxt("/hxmtfs2/work/GRB/Software/RSPgenerator_v1/EC_FWHM/2022-10-04_EC_LHV.txt", usecols=[1, 2])
ECH = np.median(EC[:, 0]) * 0.5 * (GROUP[:-1] + GROUP[1:]) + np.median(EC[:, 1])     # keV, group centres


def live():
    o, f = {}, {}
    for r in csv.DictReader(open(f"{BASE}/resp/in/ts_1s.csv")):
        t = int(float(r["t"]))
        o.setdefault(t, []).append(float(r["o_all"]))
        f.setdefault(t, []).append(float(r["flive"]))
    return o, f


def f_eff(windows, o, f):
    """Rate-weighted live fraction: sum N / sum (N/f), per box then summed."""
    num = den = 0.0
    for a, b in windows:
        for t in range(int(np.floor(a)), int(np.ceil(b))):
            for ob, fb in zip(o.get(t, []), f.get(t, [])):
                num += ob
                den += ob / fb
    return num / den


def rec_counts(windows):
    z = np.load(f"{BASE}/blink/data/p221009/scale/hxmt_events.npz")
    t, ch, pw = z["t"], z["ch"].astype(int), z["pw"].astype(int)
    ch = np.where(ch < 20, ch + 256, ch)
    s = (pw >= 76) & (pw <= 255) & (ch >= GROUP[0]) & (ch < GROUP[-1])
    gi = np.digitize(ch[s], GROUP) - 1
    tt = t[s]
    src = np.zeros(GROUP.size - 1)
    for a, b in windows:
        m = (tt >= a) & (tt < b)
        src += np.bincount(gi[m], minlength=GROUP.size - 1)
    return src


def fit_c_delta(mu, net, var):
    best = None
    for d in np.linspace(-0.3, 0.3, 601):
        m = mu * (ECH / 1000.0) ** d
        c = np.sum(net * m / var) / np.sum(m * m / var)
        x2 = np.sum((net - c * m) ** 2 / var)
        if best is None or x2 < best[0]:
            best = (x2, c, d)
    x2min, c0, d0 = best
    ds = [d for d in np.linspace(-0.3, 0.3, 601)
          if np.sum((net - (np.sum(net * mu * (ECH / 1e3) ** d / var) / np.sum((mu * (ECH / 1e3) ** d) ** 2 / var))
                     * mu * (ECH / 1e3) ** d) ** 2 / var) <= x2min + 1]
    return c0, d0, min(ds), max(ds), x2min


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    z = np.load(f"{out}/drm_221009a.npz")
    elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
    R = group_drm(z["rise_per"].astype(float))
    t, ch = load_events()
    o, f = live()
    D = [DetData(d, *BKG, trange=(-400.0, 800.0)) for d in DETS]
    lines = []
    variants = {"clean 180-186+193-215": ([(180, 186), (193, 215)], "obs"),
                "rec 180-216": ([(180, 216)], "rec"),
                "obs 180-216 (original)": ([(180, 216)], "obs")}
    for name, (wins, kind) in variants.items():
        src, bkg, bvar, expo = counts(t, ch, wins, mode="proxy")
        if kind == "rec":
            src = rec_counts(wins)
        fe = f_eff(wins, o, f)
        fm = np.mean([np.mean(f[tt]) for a, b in wins for tt in range(a, b)])
        net = src - bkg
        for model in ("Band", "COMP"):
            # GBM: fit each sub-window and sum the folded predictions (exposure-weighted)
            mu = np.zeros(R.shape[1])
            desc = []
            for a, b in wins:
                r = gbm_fit(D, a + DT, b + DT, model)
                if model == "COMP":
                    p, mdl = [np.log(r["A"]), r["index"], np.log(r["Epeak"])], cpl
                    desc.append(f"[{a},{b}] a={r['index']:.2f} Ep={r['Epeak']:.0f}")
                else:
                    p, mdl = [np.log(r["A"]), r["alpha"], np.log(r["Epeak"]), r["beta"]], band
                    desc.append(f"[{a},{b}] a={r['alpha']:.2f} Ep={r['Epeak']:.0f} b={r['beta']:.2f}")
                mu += photon_flux(mdl, p, elo, ehi) @ R * (b - a)
            mu_f = mu * fe
            for sf in (0.0, 0.03):
                var = np.maximum(src, 1) + bvar + (sf * mu_f) ** 2
                c = np.sum(net * mu_f / var) / np.sum(mu_f ** 2 / var)
                x2 = np.sum((net - c * mu_f) ** 2 / var)
                c0, d0, dlo, dhi, x2d = fit_c_delta(mu_f, net, var)
                lines.append(f"{name:24s} {model:4s} sys={sf}: f_eff={fe:.3f} (mean {fm:.3f}); C={c:.3f} chi2 {x2:.1f}/26; "
                             f"with tilt: C={c0:.3f} delta={d0:+.3f} [{dlo:+.3f},{dhi:+.3f}] chi2 {x2d:.1f}/25 | " + "; ".join(desc))
                print(lines[-1], flush=True)
            resid = (net - c * mu_f) / (c * mu_f)
            lines.append(f"   resid (sys 3%) by group: " + " ".join(f"{x:+.2f}" for x in resid))
    with open(f"{out}/r2_summary.txt", "w") as fo:
        for l in lines:
            print(l, file=fo)


if __name__ == "__main__":
    main()
