"""Shared HE/GBM normalisation C for the precursor Ep(t) (gbm.md 8.2), recomputed
from the gbm group's fixed-C fits and Ep profiles (GRB 221009A).

    python p221009_v4_sharedC.py IN OUT

IN/referee2_joint_fits.csv, IN/referee2_joint_epprofile.csv: gbm/runs/joint4
(var = C0.80 ... C1.50, DT_HE = 0.0108, COMP, n4+n8+b1 + HE, 7 intervals).
  S_tot(C) = sum_i min_Ep S_i(Ep, C); C_hat and sigma from a parabola fitted to
  the grid points with S_tot - min <= 15.
  P_i(Ep) = min_C [S_i(Ep, C) + sum_{j!=i} min S_j(C)]; best Ep at the minimum,
  1sigma (2sigma) from P_i - min = 1 (4), linear interpolation in log Ep.
Outputs OUT/sharedC_ep.csv, OUT/sharedC_stot.csv.  Deterministic.
"""
import sys

import numpy as np
import pandas as pd

IN, OUT = sys.argv[1], sys.argv[2]
F = pd.read_csv(f"{IN}/referee2_joint_fits.csv")
F = F[F["var"].str.match(r"C\d") & (F.model == "COMP") & (F.data == "GBM+HE")].copy()
F["C"] = F["var"].str[1:].astype(float)
P = pd.read_csv(f"{IN}/referee2_joint_epprofile.csv")
P = P[P["var"].str.match(r"C\d") & (P.model == "COMP")].copy()
P["C"] = P["var"].str[1:].astype(float)
Cs = np.sort(F.C.unique())
segs = sorted({(r.t1, r.t2) for r in F.itertuples()})
S = np.array([[F[(F.t1 == a) & (F.t2 == b) & (F.C == c)].stat.iloc[0] for c in Cs] for a, b in segs])   # seg x C
stot = S.sum(0)
d = stot - stot.min()
m = d <= 15
a2, a1, a0 = np.polyfit(Cs[m], stot[m], 2)
c_hat = -a1 / (2 * a2)
c_sig = 1 / np.sqrt(a2)
pd.DataFrame(dict(C=Cs, S_tot=stot, dS=d)).to_csv(f"{OUT}/sharedC_stot.csv", index=False, float_format="%.4f")
print("S_tot - min:", dict(zip(Cs, np.round(d, 1))), f"\nC_hat = {c_hat:.3f} +- {c_sig:.3f} (parabola on {m.sum()} points)")


def cross(x, y, lev, side):
    """Walk from the minimum towards side (-1/+1) to the first grid point with
    y >= lev and interpolate linearly in log x; None if the profile stays below."""
    i = int(np.argmin(y))
    while 0 <= i + side < len(y):
        j = i + side
        if y[j] >= lev:
            return float(np.exp(np.log(x[i]) + (lev - y[i]) * (np.log(x[j]) - np.log(x[i])) / (y[j] - y[i])))
        i = j
    return None


rows = []
for k, (a, b) in enumerate(segs):
    others = S.sum(0) - S[k]                 # sum_{j != i} min S_j(C)
    sub = P[(P.t1 == a) & (P.t2 == b)]
    grid = np.sort(sub.Ep.unique())
    prof = np.full(grid.size, np.inf)
    for ic, c in enumerate(Cs):
        q = sub[sub.C == c].set_index("Ep").stat.reindex(grid).to_numpy()
        prof = np.minimum(prof, q + others[ic])
    pmin = prof.min()
    y = prof - pmin
    ep = float(grid[np.argmin(prof)])
    r = dict(t1=a, t2=b, Ep=ep, Ep_lo1=cross(grid, y, 1, -1), Ep_hi1=cross(grid, y, 1, +1),
             Ep_lo2=cross(grid, y, 4, -1), Ep_hi2=cross(grid, y, 4, +1), C_shared=c_hat, C_sig=c_sig,
             C_own=float(Cs[np.argmin(S[k])]), grid_max=float(grid.max()))
    rows.append(r)
    print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/sharedC_ep.csv", index=False, float_format="%.6g")
print("done")
