"""Referee-2 round-2 R1 post-processing of the fixed-C joint fits
(p221009_gbm_joint.py variants C<val>[~tilt<d>|~tlin2|~hech..|~g..|~b_..]).

1. C-delta grid: S_tot(C, delta) = sum_i min S_i(C, delta) over the seven
   precursor intervals (delta = global HE response tilt), the global minimum,
   the C profile with delta free / with priors N(0, 0.03), N(0, 0.06), and
   delta restricted to |delta| <= 0.03 / 0.06.
2. Per-interval S_i(C) at delta = 0 (preferred C, Delta S at the shared C).
3. 3-4.5 s: GBM-only PG-stat profile in Ep evaluated at the shared-C Ep,
   and the GBM / HE split of the statistic at the shared-C best fit.
4. Fixed-C (1.00, 1.12) Ep(t) with systematics variants.

    python p221009_gbm_r1.py JOINT3_DIR RUN_DIR [RUN_DIR ...]
Output $GBM_OUT/r1/{cdelta_grid.csv, cdelta_summary.txt, sC_interval.csv,
fixedC_ep.csv, i9_split.txt}.
"""
import os
import sys
import numpy as np
import pandas as pd
import p221009_gbm_common as C

J3 = sys.argv[1]
F = pd.concat([pd.read_csv(os.path.join(d, "joint_fits.csv")) for d in sys.argv[2:]])
P = pd.concat([pd.read_csv(os.path.join(d, "joint_epprofile.csv")) for d in sys.argv[2:]])
F = F[F["var"].astype(str).str.match(r"^C\d") & (F.model == "COMP")].copy()
P = P[P["var"].astype(str).str.match(r"^C\d") & (P.model == "COMP") & (P.he == True)].copy()  # noqa: E712
for D in (F, P):
    D["C"] = D["var"].str.extract(r"^C([\d.]+)")[0].astype(float).round(3)
    D["sys"] = D["var"].str.extract(r"^C[\d.]+~?(.*)$")[0].replace("", "nominal")
    D["lab"] = D.t1.round(3).astype(str) + "-" + D.t2.round(3).astype(str)
    D.drop_duplicates(subset=[c for c in ("var", "lab", "Ep") if c in D] if D is P else ["var", "lab"],
                      keep="last", inplace=True)
smin = np.minimum(F.groupby(["sys", "C", "lab"]).stat.min(), P.groupby(["sys", "C", "lab"]).stat.min())
smin = smin.dropna()
od = C.outdir("r1")
out = []

# ---------------------------------------------------------------- 1. C-delta
def dval(s):
    if s == "nominal":
        return 0.0
    if s.startswith("tilt") and not s.startswith("tiltp"):
        return float(s[4:])
    return np.nan


S = smin.reset_index()
S["delta"] = S["sys"].map(dval)
G = S[np.isfinite(S.delta)]
cnt = G.groupby(["delta", "C"]).lab.nunique()
tot = G.groupby(["delta", "C"]).stat.sum()[cnt == 7].unstack()       # rows delta, cols C
tot.to_csv(os.path.join(od, "cdelta_grid.csv"), float_format="%.3f")
m = np.nanmin(tot.values)
out.append("S_tot(C, delta) - min  (rows delta, cols C); min = %.2f" % m)
out.append((tot - m).round(2).to_string())


def cprof(T, prior=None, dmax=None):
    """min over delta of S_tot (+ prior) per C, delta interpolated by a parabola per C."""
    res = {}
    for c in T.columns:
        col = T[c].dropna()
        d, s = col.index.values.astype(float), col.values
        if dmax is not None:
            k = np.abs(d) <= dmax + 1e-9
            d, s = d[k], s[k]
        if len(d) < 3:
            continue
        a = np.polyfit(d, s, 2) if len(d) <= 5 else np.polyfit(d, s, 4)
        dd = np.linspace(d.min(), d.max(), 2001)
        ss = np.polyval(a, dd) + (0 if prior is None else (dd / prior) ** 2)
        j = np.argmin(ss)
        res[c] = (ss[j], dd[j])
    return pd.DataFrame(res, index=["S", "delta"]).T


def cbest(prof):
    c = prof.index.values.astype(float); s = prof.S.values
    k = np.argsort(s)[:3]
    a = np.polyfit(c[k], s[k], 2)
    cb = -a[1] / (2 * a[0]); return cb, 1 / np.sqrt(a[0]), np.polyval(a, cb)


for name, kw in [("delta = 0", None), ("delta free", {}), ("delta prior N(0,0.03)", dict(prior=0.03)),
                 ("delta prior N(0,0.06)", dict(prior=0.06)), ("|delta| <= 0.03", dict(dmax=0.03)),
                 ("|delta| <= 0.06", dict(dmax=0.06)), ("|delta| <= 0.1", dict(dmax=0.1))]:
    if kw is None:
        pr = pd.DataFrame({"S": tot.loc[0.0], "delta": 0.0}).dropna()
    else:
        pr = cprof(tot, **kw)
    cb, sc, sb = cbest(pr)
    sline = "  ".join("C=%.2f: %+.1f (d=%+.3f)" % (c, r.S - sb, r.delta) for c, r in pr.iterrows())
    # Delta S at C = 1.00 and 1.12 by interpolation of the profile
    cc = pr.index.values.astype(float)
    at = {c0: np.interp(c0, cc, pr.S.values) - sb for c0 in (1.0, 1.12)}
    out.append("\n[%s]  C_best = %.3f +- %.3f   dS(C=1.00) = %.1f   dS(C=1.12) = %.1f\n  %s"
               % (name, cb, sc, at[1.0], at[1.12], sline))
# global best delta (free) and its 1-sigma range from the delta profile (min over C)
dp = tot.min(axis=1)
d = dp.index.values.astype(float)
a = np.polyfit(d, dp.values, 4); dd = np.linspace(d.min(), d.max(), 4001); ss = np.polyval(a, dd)
j = np.argmin(ss); ok = dd[ss - ss[j] < 1]
out.append("\ndelta profile (min over C): best %.3f, 1-sigma [%.3f, %.3f]; dS(delta=0) = %.2f"
           % (dd[j], ok.min(), ok.max(), np.polyval(a, 0) - ss[j]))
out.append("  " + "  ".join("d=%+.2f: %+.2f" % (x, y - ss[j]) for x, y in zip(d, dp.values)))

# ---------------------------------------------------------------- 2. per interval S_i(C), delta = 0
rows = []
for sysv in sorted(S.sys.unique()):
    T = S[S.sys == sysv].pivot_table(index="lab", columns="C", values="stat")
    T = T.dropna(axis=1)
    if T.shape[1] < 3:
        continue
    tt = T.sum(0); c = tt.index.values.astype(float)
    k = np.argsort(tt.values)[:3]; a = np.polyfit(c[k], tt.values[k], 2)
    cs = -a[1] / (2 * a[0]) if a[0] > 0 else c[k[0]]
    for lab, r in T.iterrows():
        k = np.argsort(r.values)[:3]; b = np.polyfit(c[k], r.values[k], 2)
        ci = -b[1] / (2 * b[0]) if b[0] > 0 else c[k[0]]
        smn = np.polyval(b, ci) if b[0] > 0 else r.min()
        si = 1 / np.sqrt(b[0]) if b[0] > 0 else np.nan
        rows.append(dict(sys=sysv, lab=lab, C_pref=ci, C_pref_err=si, C_shared=cs,
                         dS_at_shared=np.interp(cs, c, r.values) - smn,
                         dS_at_1p00=np.interp(1.0, c, r.values) - smn if c.min() <= 1.0 else np.nan,
                         C_grid=" ".join("%.2f" % x for x in c)))
SC = pd.DataFrame(rows)
C.write_csv(SC, "r1", "sC_interval.csv")

# ---------------------------------------------------------------- 3. the 3-4.5 s interval
# the merged joint3 profile has no 'var' column (n7 GBM-only rows mixed in): use the n4 nominal file
G3 = pd.read_csv(os.path.join(J3, "joint_epprofile_9.csv"))
g = G3[(G3.t1 == 3.0) & (G3.t2 == 4.5) & (G3.model == "COMP") & (G3.he == False)]  # noqa: E712
g = g.groupby("Ep").stat.min()
F3 = pd.read_csv(os.path.join(J3, "joint_fits.csv"))
f0 = F3[(F3.t1 == 3.0) & (F3.t2 == 4.5) & (F3.model == "COMP") & (F3.data == "GBM") & (F3["var"] == "nominal")].iloc[0]
gmin = min(f0.stat, g.min())
lines = ["GBM-only 3-4.5 s: best Ep %.0f, PG %.2f" % (f0.Ep, gmin)]
for ep in (500, 600, 640, 700, 800, 890):
    lines.append("  GBM-only Delta PG at Ep = %4d keV: %.2f" % (ep, np.interp(np.log(ep), np.log(g.index.values), g.values) - gmin))
for sysv in ("nominal",):
    r = F[(F.sys == sysv) & (F.lab == "3.0-4.5")].sort_values("C")
    for _, x in r.iterrows():
        lines.append("  C=%.2f  joint stat %.2f  PG_gbm %.2f (GBM-only min %.2f -> +%.2f)  chi2_HE %.2f  Ep %.0f [%.0f, %.0f] alpha %.2f"
                     % (x.C, x.stat, x.pg_gbm, gmin, x.pg_gbm - gmin, x.chi2_he, x.Ep, x.Ep_lo1, x.Ep_hi1, x.alpha))
open(os.path.join(od, "i9_split.txt"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- 4. fixed C Ep(t)
FX = F[F.C.isin([1.0, 1.12])][["C", "sys", "lab", "t1", "t2", "Ep", "Ep_lo1", "Ep_hi1", "alpha", "stat", "chi2_he"]]
C.write_csv(FX.sort_values(["C", "sys", "t1"]), "r1", "fixedC_ep.csv")
open(os.path.join(od, "cdelta_summary.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out)); print(SC.to_string()); print("\n".join(lines))
print(FX[FX.sys == "nominal"].sort_values(["C", "t1"]).to_string())
