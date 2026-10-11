"""Direct check of the cross-box lag-0 coincidences seen in pair_g:
exact nearest-neighbour counts and the variance of the summed stream."""
import numpy as np
from p221009_sci_lib import *

t, box, ch, pw = he_obs()
d = np.load(EVT_REC)
for lo, hi in ((20.0, 170.0), (440.0, 500.0)):
    tb = [t[(box == b) & (t >= lo) & (t < hi)] for b in range(3)]
    lam = [len(x) / (hi - lo) for x in tb]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = tb[i], tb[j]
        k = np.searchsorted(b, a)
        k0 = np.clip(k - 1, 0, len(b) - 1); k1 = np.clip(k, 0, len(b) - 1)
        dn = np.minimum(np.abs(a - b[k0]), np.abs(b[k1] - a))
        for w in (0.1e-6, 1.1e-6, 2.1e-6, 4.1e-6, 10e-6):
            print(f"{lo:.0f}-{hi:.0f} {i}{j} |dt|<={w*1e6:.1f}us: frac {np.mean(dn <= w):.5f}  expect {2*w*lam[j]:.5f}")
        same = dn == 0
        print("   exact equal times:", same.sum(), " ch equal among them:",
              np.mean(ch[(box == i) & (t >= lo) & (t < hi)][same] if same.any() else 0))
    # variance test, 1 ms and 10 ms bins
    for D in (1e-3, 1e-2):
        e = np.arange(lo, hi + D / 2, D)
        n = [np.histogram(x, e)[0].astype(float) for x in tb]
        s = sum(n)
        print(f"  D={D}: var(sum)/mean {s.var()/s.mean():.4f};  per box {[round(x.var()/x.mean(),4) for x in n]};"
              f"  cov AB/mean {np.cov(n[0], n[1])[0,1]/s.mean():+.5f} AC {np.cov(n[0], n[2])[0,1]/s.mean():+.5f}")
# raw cache: are there exact duplicates (same t) across boxes, incl. fill?
w = (d["t"] >= 20) & (d["t"] < 170)
tt = d["t"][w]; bb = d["box"][w]
u, c = np.unique(tt, return_counts=True)
print("raw cache 20-170: events", len(tt), " distinct times", len(u), " times shared by >1 event", (c > 1).sum())
