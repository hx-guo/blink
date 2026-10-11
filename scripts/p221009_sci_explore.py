"""Inventory of the per-event files used by the variability scouting."""
import numpy as np
from astropy.io import fits
from p221009_sci_common import *

d = np.load(EVT_REC)
t, fill, ch, pw, box = d["t"], d["fill"], d["ch"], d["pw"], d["box"]
print("REC cache:", len(t), "t range", t.min(), t.max())
print("  fill frac", fill.mean(), " boxes", np.bincount(box))
z = (ch == 0) & (pw == 0)
print("  ch==0&pw==0:", z.sum(), " of which fill", (z & fill).sum())
fr = t[z] - np.floor(t[z])
print("  frac-second of ch0pw0 events: hist", np.histogram(fr, bins=10, range=(0, 1))[0])
print("  first ch0pw0 times", t[z][:12])
print("  ch hist (obs, first 40)", np.bincount(ch[~fill], minlength=256)[:40])
print("  pw hist (obs)", np.bincount(pw[~fill], minlength=256)[::8])
# duplicates / min spacing in observed events per box
for b in range(3):
    tb = t[(box == b) & ~fill & (t < -5)]
    dt = np.diff(tb)
    print(f"  box {b} pre-burst obs: n={len(tb)} rate={len(tb)/(tb[-1]-tb[0]):.1f}/s; "
          f"dt==0: {(dt==0).sum()}, dt<1us {(dt<1e-6).sum()}, <2us {(dt<2e-6).sum()}, <5us {(dt<5e-6).sum()}, <20us {(dt<2e-5).sum()}")
    print("    dt quantum: unique smallest", np.unique(np.round(dt[dt > 0] * 1e6, 3))[:10])

p = np.load(EVT_PKT)
for b in "abc":
    tt, pk, sec = p[f"t_{b}"], p[f"pkt_{b}"], p[f"sec_{b}"]
    print(f"PKT box {b}: n={len(tt)} sec={sec.sum()} range {tt.min()-T0_MET:.2f}..{tt.max()-T0_MET:.2f}")
    s = tt[sec] - T0_MET
    print("   SEC frac part", np.round((s - np.floor(s))[:8], 6), " spacing", np.round(np.diff(s)[:5], 6))
    u, n = np.unique(pk, return_counts=True)
    print("   events/packet: median", np.median(n), "hist", np.bincount(n)[100:115])

h = np.load(HE1K)
print("HE1K:", {k: (h[k].shape, h[k].dtype) for k in h.files})
for k in h.files:
    print("  ", k, h[k][:5])

f = fits.open(GECAM_EVT, memmap=True)
print([ (x.name, getattr(x, 'data', None) is not None and x.data is not None and len(x.data)) for x in f])
g0 = gecam_t0()
for x in f:
    if x.name.startswith("GTI") or x.name == "GTI":
        print(x.name, x.columns.names)
        print(np.c_[x.data["START"] - g0, x.data["STOP"] - g0][:20])
for i in (1, 7):
    ev = f[f"EVENTS{i:02d}"].data
    tr = ev["TIME"] - g0
    print(f"EVENTS{i:02d} cols", ev.columns.names, "n", len(tr), "range", tr.min(), tr.max())
    m = (tr > -200) & (tr < 800)
    tt = np.sort(tr[m]); dd = np.diff(tt)
    k = np.where(dd > 1.0)[0]
    print("   gaps >1 s in -200..800:", [(round(tt[j],2), round(tt[j+1],2)) for j in k])
b = fits.open(GECAM_BTIME, memmap=True)
print("BTIME HDUs", [(x.name) for x in b])
for x in b[1:3]:
    print(x.name, x.columns.names if hasattr(x, 'columns') else None, len(x.data))
