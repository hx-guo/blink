"""v4 HE/CsI products: quiet-period 1-s light curve with the external-slope
LargeEvt background, and HE-only precursor photon indices on the joint-fit
segments (GRB 221009A).

    run_resp.sh p221009_v4_he.py OUTDIR      (resp venv)

(a) Quiet period (resp.md 7.3-7.4): CsI ch 20-255 per second from
    resp/out/adjorb/hist_burst.npz; background B = a + s*LE with the slope s
    fitted on a source-free low-gain reference pass (d1008 nominal; d1007,
    d1003 as checks; LE < 5500) and a from the burst-day T0-420...-10 s.
    Window nets are printed to check the resp numbers (+5/s II, -1/s 20-175).
(b) HE-only PL index (0.17-3.1 MeV, tloc background, precursor DRM) on the
    GBM joint-fit segments (gbm.md 5).  HE event times are relative to the paper T0
    (13:17:00.000 UTC, HXMT MET 339945423.0).  Two window sets: 'jointfit' = t_GBM - 0.06 s
    (the HE windows actually used in p221009_gbm_joint.py) and 'T0paper' = t_GBM - 0.0108 s
    (simultaneous with the GBM interval; p221009_v4_ltt.py); plus
    0-9 s and the resp segments as a reproduction check (resp.md 4.1-4.2).
Outputs OUTDIR/quiet_he_1s.csv, quiet_he_windows.csv, he_prec_index.csv.
"""
import sys

import numpy as np
import pandas as pd

from p221009_resp_adjorb_ana import load, sel, wsum
from p221009_resp_adjorb_ana2 import cnts
from p221009_resp_spec import BASE, counts, fit, flive_mean, group_drm, load_events, pl
from p221009_resp_prec import efluxband

out = sys.argv[1]
SYS = 15.0          # counts/s, resp.md 7.3 systematic for <=175 s extrapolation
WINS = {"II": (121.159, 164.168), "20-175": (20, 175), "115-172": (115, 172), "60-118": (60, 118),
        "-100..-10": (-100, -10), "II_T0paper": (121.219 - 0.0108, 164.228 - 0.0108)}



def pl_index(R, elo, ehi, src, bkg, bvar, expo, fl):
    """As p221009_resp_prec.pl_index (Delta chi2 = 1 profile), on a +-1.2 grid so
    that the faint segments are not clipped."""
    p, c2 = fit(pl, [-3, -1.8], R, elo, ehi, src, bkg, bvar, expo, fl, bounds={1: (-5, 1)})
    grid = np.linspace(p[1] - 1.2, p[1] + 1.2, 241)
    prof = np.array([fit(pl, [p[0], g], R, elo, ehi, src, bkg, bvar, expo, fl, fixed={1: g})[1] for g in grid])
    ok = prof <= c2 + 1.0
    return p, c2, grid[ok].min(), grid[ok].max()


# ---------------------------------------------------------------- (a) quiet period
tr, gb, _ = load("burst")
cb = gb.sum(1).astype(float)
le = cnts("burst", tr)["le"]
pre = sel(tr, (-420, -10))
lc = {"t": tr, "cnt": cb, "le": le}
rows = []
for r in ("d1008", "d1007", "d1003"):
    _, g, _ = load(r)
    cr = g.sum(1)
    ler = cnts(r, tr)["le"]
    ok = np.isfinite(ler) & (cr > 0) & (ler < 5500)
    s, _ = np.polyfit(ler[ok], cr[ok], 1)
    a = np.mean(cb[pre] - s * le[pre])
    b = a + s * le
    lc["bkg_" + r] = b
    for wn, w in WINS.items():
        dur = np.ptp(w)
        net = wsum(tr, cb - b, w) / dur
        stat = np.sqrt(wsum(tr, cb, w)) / dur
        # offset uncertainty from the 410-s pre-burst mean (fully correlated in the window)
        aerr = np.std(cb[pre] - s * le[pre]) / np.sqrt(pre.sum())
        rows.append(dict(ref=r, slope=round(s, 4), window=wn, t1=w[0], t2=w[1], net_rate=net,
                         stat=np.hypot(stat, aerr), sys=SYS))
        print(f"{r} slope {s:.4f}: {wn:10s} net {net:+7.1f} +- {np.hypot(stat, aerr):5.1f} (stat) /s", flush=True)
pd.DataFrame(lc).to_csv(f"{out}/quiet_he_1s.csv", index=False, float_format="%.6g")
pd.DataFrame(rows).to_csv(f"{out}/quiet_he_windows.csv", index=False, float_format="%.6g")

# ---------------------------------------------------------------- (b) HE-only precursor indices
z = np.load(f"{BASE}/resp/out/drm_221009a.npz")
elo, ehi = z["ein_lo"].astype(float), z["ein_hi"].astype(float)
R = group_drm(z["precursor_per"].astype(float))
t, ch = load_events()
D = 0.06               # HE window offset of the joint fits (gbm.md 5)
G2P = -0.0108          # t_paper = t_GBM + G2P
SEGS = [("tr", -1, 0), ("tr", 0, 1), ("tr", 1, 2), ("tr", 2, 3), ("tr", 3, 4.5), ("tr", 4.5, 6.5),
        ("tr", 6.5, 8.576), ("Ia", -0.003, 8.576), ("HEwin", -0.84, 10.06),
        ("axel1", -0.69, 0.23), ("axel2", 0.23, 1.0), ("axel3", 1.0, 2.72), ("axel4", 2.72, 7.0)]
res = []
for wset, sh in (("jointfit", -D), ("T0paper", G2P)):
    for tag, g1, g2 in SEGS:
        w = [(g1 + sh, g2 + sh)]
        src, bkg, bvar, expo = counts(t, ch, w)
        fl = flive_mean(w)
        p, c2, lo, hi = pl_index(R, elo, ehi, src, bkg, bvar, expo, fl)
        res.append(dict(tag=tag, wset=wset, t1_gbm=g1, t2_gbm=g2, t1_hxmt=g1 + sh, t2_hxmt=g2 + sh,
                        net=src.sum() - bkg.sum(), index=p[1], index_lo=lo, index_hi=hi, chi2=c2, dof=R.shape[1] - 2,
                        eflux_300_3000=efluxband(pl, p, 300, 3000)))
        print(res[-1], flush=True)
for w in ([(0, 9)], [(-1, 1.2)], [(1.2, 2.4)], [(2.4, 3.9)], [(3.9, 6.7)], [(6.7, 10)]):     # resp.md 4.1/4.2 check
    src, bkg, bvar, expo = counts(t, ch, w)
    p, c2, lo, hi = pl_index(R, elo, ehi, src, bkg, bvar, expo, flive_mean(w))
    res.append(dict(tag="resp_check", wset="T0paper", t1_hxmt=w[0][0], t2_hxmt=w[0][1], net=src.sum() - bkg.sum(), index=p[1],
                    index_lo=lo, index_hi=hi, chi2=c2, dof=R.shape[1] - 2, eflux_300_3000=efluxband(pl, p, 300, 3000)))
    print(res[-1], flush=True)
pd.DataFrame(res).to_csv(f"{out}/he_prec_index.csv", index=False, float_format="%.6g")
print("done")
