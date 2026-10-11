"""Adjacent-orbit study, step 3c: LargeEvt lever arm and calibration-window
dependence of the LE-proxy background (GRB 221009A quiet period).

    python p221009_resp_adjorb_lever.py"""
import numpy as np

from p221009_resp_adjorb_ana import load, sel, wsum
from p221009_resp_adjorb_ana2 import cnts

tr, gb, _ = load("burst")
cb = gb.sum(1)
le = cnts("burst", tr)["le"]
for w in ((-1020, -10), (-420, -10), (-100, -10), (20, 175), (115, 172), (1900, 2240), (2240, 2580)):
    m = sel(tr, w)
    if m.any():
        print(f"window {w}: LE mean {le[m].mean():.0f} (min {le[m].min():.0f} max {le[m].max():.0f}), CsI {cb[m].mean():.0f}/s")
for cal in ([(-420, -10)], [(-420, -110)], [(-420, -10), (1900, 2580)], [(-420, -10), (2240, 2580)],
            [(-420, -10), (1900, 2240)]):
    m = np.zeros(tr.size, bool)
    for w in cal:
        m |= sel(tr, w)
    c = np.polyfit(le[m], cb[m], 1)
    b = np.polyval(c, le)
    q = wsum(tr, cb - b, (121.159, 164.168)) / 43.0
    q2 = wsum(tr, cb - b, (20, 175)) / 155.0
    v = np.mean([wsum(tr, cb - b, (a, a + 10)) for a in range(-100, -10, 10)])
    print(f"cal {cal}: slope {c[0]:.3f} const {c[1]:.0f}; net II {q:+.1f}/s, 20-175 {q2:+.1f}/s; mean resid -100..-10 {v / 10:+.1f}/s")

# CsI-vs-LE slope from the source-free reference windows (same ground track,
# low gain), offset from the burst-day pre-burst only
print("--- slope from reference windows (-420..2620 s, LE < 5500), offset from burst pre-burst")
for r in ("d1008", "d1007", "d1003"):
    _, g, _ = load(r)
    cr = g.sum(1)
    ler = cnts(r, tr)["le"]
    ok = np.isfinite(ler) & (cr > 0) & (ler < 5500)          # LE range of the burst-day calibration/quiet data
    sr, cc = np.polyfit(ler[ok], cr[ok], 1)
    # quadratic check
    q2 = np.polyfit(ler[ok], cr[ok], 2)
    rr = cr[ok] - np.polyval([sr, cc], ler[ok])
    print(f"{r}: LE range {ler[ok].min():.0f}-{ler[ok].max():.0f}; slope {sr:.3f} (quad coef {q2[0]:.2e}); resid rms 10-s {np.std(np.convolve(rr, np.ones(10), 'valid')[::10]):.0f}")
    pre = sel(tr, (-420, -10))
    a = np.mean(cb[pre] - sr * le[pre])
    b = a + sr * le
    for w in ((-100, -10), (20, 175), (115, 172), (121.159, 164.168), (1900, 2240), (2240, 2580)):
        print(f"   net {w}: {wsum(tr, cb - b, w) / np.ptp(w):+.1f}/s")
