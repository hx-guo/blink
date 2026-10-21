"""Shared-C post-processing of fixed-C joint fits (p221009_gbm_joint.py
variants C<val>[~<variant>]): for each systematic variant, the shared
normalisation C (minimum of S_tot(C) = sum_i min S_i(C), parabola through the
three lowest grid points), the per-interval Ep profile
P_i(Ep) = min_C [S_i(Ep, C) + sum_{j != i} min S_j(C)], its best value and
1-sigma range, and the free-delta value at the shared C for the tiltp variant.

    python p221009_gbm_sharedC.py RUN_DIR [RUN_DIR ...]
(joint_fits.csv / joint_epprofile.csv with a 'var' column in each).
Output $GBM_OUT/sharedC/sharedC_ep.csv, sharedC_C.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import p221009_gbm_common as C

F = pd.concat([pd.read_csv(os.path.join(d, "joint_fits.csv")) for d in sys.argv[1:]])
P = pd.concat([pd.read_csv(os.path.join(d, "joint_epprofile.csv")) for d in sys.argv[1:]])
F = F[F["var"].astype(str).str.match(r"^C\d")]
P = P[P["var"].astype(str).str.match(r"^C\d")]
for D in (F, P):
    D["C"] = D["var"].str.extract(r"^C([\d.]+)")[0].astype(float)
    D["sys"] = D["var"].str.extract(r"^C[\d.]+~?(.*)$")[0].replace("", "nominal")
    D["lab"] = D.t1.round(3).astype(str) + "-" + D.t2.round(3).astype(str)
P = P[P.he == True]  # noqa: E712

ep_rows, c_rows = [], []
for sysv, Fs in F.groupby("sys"):
    Ps = P[P.sys == sysv]
    smin = np.minimum(Fs.groupby(["lab", "C"]).stat.min().unstack(), Ps.groupby(["lab", "C"]).stat.min().unstack())
    smin = smin.dropna(axis=1)
    tot = smin.sum(0)
    Cg = tot.index.values.astype(float)
    k = np.argsort(tot.values)[:3]
    a = np.polyfit(Cg[k], tot.values[k], 2)
    Cb = -a[1] / (2 * a[0]) if a[0] > 0 else Cg[k[0]]
    sC = 1 / np.sqrt(a[0]) if a[0] > 0 else np.nan
    split = float((smin.min(1)).sum())
    c_rows.append(dict(sys=sysv, C_shared=Cb, C_err=sC, S_tot_min=float(tot.min()),
                       S_split=split, dS_shared_vs_free=float(tot.min()) - split, n_int=len(smin),
                       C_grid=" ".join("%.2f" % c for c in Cg)))
    grid = np.sort(Ps.Ep.unique())
    for lab in smin.index:
        others = tot - smin.loc[lab]
        prof = []
        for ep in grid:
            v = [Ps[(Ps.lab == lab) & (Ps.C == c) & np.isclose(Ps.Ep, ep)].stat.min() + others[c] for c in Cg]
            prof.append(np.nanmin(v))
        prof = np.array(prof)
        y = prof - np.nanmin(prof); ok = np.where(y < 1)[0]
        lo = np.exp(np.interp(1, [y[ok[0]], y[ok[0] - 1]], np.log(grid[[ok[0], ok[0] - 1]]))) if ok[0] > 0 else np.nan
        hi = np.exp(np.interp(1, [y[ok[-1]], y[ok[-1] + 1]], np.log(grid[[ok[-1], ok[-1] + 1]]))) \
            if ok[-1] < len(grid) - 1 else np.nan
        # best Ep: parabola in log Ep around the grid minimum
        j = int(np.nanargmin(prof))
        if 0 < j < len(grid) - 1:
            xx = np.log(grid[j - 1:j + 2]); pp = np.polyfit(xx, prof[j - 1:j + 2], 2)
            epb = np.exp(-pp[1] / (2 * pp[0])) if pp[0] > 0 else grid[j]
        else:
            epb = grid[j]
        r = Fs[(Fs.lab == lab)]
        rc = r.iloc[np.argmin(abs(r.C - Cb))]
        ep_rows.append(dict(sys=sysv, lab=lab, t1=rc.t1, t2=rc.t2, C_shared=Cb, Ep=epb, Ep_lo1=lo, Ep_hi1=hi,
                            alpha_at_nearestC=rc.alpha, delta_at_nearestC=rc.get("delta", np.nan)))
E = pd.DataFrame(ep_rows); CC = pd.DataFrame(c_rows)
C.write_csv(E, "sharedC", "sharedC_ep.csv"); C.write_csv(CC, "sharedC", "sharedC_C.csv")
print(CC.to_string()); print(E.to_string())
