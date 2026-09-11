"""两件事：(1) 第 23 条那 28 个用新去重口径重算；(2) A 星 POSATT 坏行按物理半径带体检。

(1) 那 28 个的定义：2024-01-11，准入上界 + `bin >= 10 us` + `fa <= 7e-7`。
    它们本来就已经被限制在长窗，所以"短窗上 f3 零信息"那一条打不到它们；
    咬着它们的是去重口径。这里在新口径（死时间 / 一高一低 / 留低增益）上重算。
    偶然期望同时按 tau = q（严格同戳）与 tau = 150 ns 两套给，因为期望约正比于
    tau^2，两套数不能直接比。

(2) POSATT 的坏行不止"精确为零"那一类：B 星实测 36 行的 |r| 落在 1362–9060 km，
    能通过 |r| > 1e6 的门。正确判据是物理半径带。A 星高度实测 561–600 km，
    所以带取 [6850, 7050] km。
"""
import csv
import glob

import numpy as np
from scipy.special import gammaln

D = "/scratchfs2/gecam/guohx/gecam_a/"
Q_NS = 29.802322387695312
TAU_NS = 150.0


def chance(n, m):
    if n < 3 or m < 1:
        return 0.0
    p = 1.0 / m
    j = np.arange(0, min(n, 40) + 1)
    pmf = np.exp(gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
                 + j * np.log(p) + (n - j) * np.log1p(-p))
    return m * float((j[3:] * pmf[3:]).sum())


rows = list(csv.DictReader(open(D + "chain_20240111.csv")))
fa = np.array([float(r["fa"]) for r in rows])
binu = np.array([float(r["bin_us"]) for r in rows])
nkeep = np.array([int(r["n_keep"]) for r in rows])
nobs = np.array([int(r["n_obs"]) for r in rows])
f3 = np.array([float(r["f3"]) for r in rows])
f2 = np.array([float(r["f2"]) for r in rows])
ntrip = np.array([int(r["n_trip"]) for r in rows])
e_ntrip = np.array([float(r["e_n_trip"]) for r in rows])

sel = (nkeep >= 8) & (binu >= 10.0) & (fa <= 7e-7)
print(f"第 23 条的选法（准入上界 + bin >= 10 us + fa <= 7e-7），新去重口径："
      f"**{int(sel.sum())} 个**（老口径下是 28 个）\n")

i = np.where(sel)[0]
if i.size:
    print(f"  bin_us: p0 {binu[i].min():.1f}  p50 {np.median(binu[i]):.1f}  p100 {binu[i].max():.1f}")
    print(f"    10–50 us {int(((binu[i]>=10)&(binu[i]<50)).sum())} 个，"
          f">= 50 us {int((binu[i]>=50).sum())} 个")
    print(f"  实测 f3 > 0: {int((f3[i]>0).sum())}/{i.size} = {(f3[i]>0).mean()*100:.1f}%"
          f"（老口径报的是 28/28）")
    eq = e_ntrip[i]
    et = np.array([chance(nobs[k], max(int(round(binu[k] * 1000 / TAU_NS)), 1) + 1) for k in i])
    print(f"  偶然三重期望（tau = q = {Q_NS:.1f} ns）: 中位 {np.median(eq):.5f}  最大 {eq.max():.5f}")
    print(f"  偶然三重期望（tau = {TAU_NS:.0f} ns）    : 中位 {np.median(et):.5f}  最大 {et.max():.5f}")
    print(f"  实测三重计数合计 {int(ntrip[i].sum())}，"
          f"对 tau=q 的期望和 {eq.sum():.4f} => {ntrip[i].sum()/max(eq.sum(),1e-12):.0f} 倍；"
          f"对 tau=150ns 的期望和 {et.sum():.4f} => {ntrip[i].sum()/max(et.sum(),1e-12):.0f} 倍")
    print(f"  f2 中位 {np.median(f2[i]):.3f}，count 中位 {np.median(nobs[i]):.0f}")

print("\n同一天各 fa 档下新口径 f3 > 0 的占比（都先过准入 + bin >= 10 us）")
base = (nkeep >= 8) & (binu >= 10.0)
print(f"{'fa 阈':>10} {'n':>7} {'f3>0 占比':>10} {'bin 中位':>9}")
for t in (20, 1, 1e-2, 1e-4, 7e-7):
    s = base & (fa <= t)
    if s.any():
        print(f"{t:10.0e} {int(s.sum()):7d} {(f3[s]>0).mean()*100:9.1f}% {np.median(binu[s]):9.1f}")

print("\n" + "=" * 64)
print("POSATT 坏行体检（A 星，物理半径带 [6850, 7050] km）")
print("=" * 64)
from astropy.io import fits
tot = bad_zero = bad_band = bad_pass1e6 = 0
rs = []
for day in ("2024/01/11", "2023/06/01"):
    for p in sorted(glob.glob(f"/gecamfs/Archived-DATA/GSDC/LEVEL1/daily/{day}/GECAM_A/posatt/*_v*.fits")):
        with fits.open(p) as h:
            d = h["Orbit_Attitude"].data
            x = np.asarray(d["X_WGS84"], float)
            y = np.asarray(d["Y_WGS84"], float)
            z = np.asarray(d["Z_WGS84"], float)
        r = np.sqrt(x * x + y * y + z * z)
        tot += r.size
        bad_zero += int((r == 0).sum())
        ok = (r > 6.85e6) & (r < 7.05e6)
        bad_band += int((~ok).sum())
        bad_pass1e6 += int((~ok & (r > 1e6)).sum())
        rs.append(r[ok])
r = np.concatenate(rs)
print(f"  采样点 {tot}  精确为零 {bad_zero}  出半径带 {bad_band}  "
      f"（其中能通过 |r| > 1e6 的 {bad_pass1e6}）")
print(f"  带内半径 km: p0 {r.min()/1e3:.1f}  p50 {np.median(r)/1e3:.1f}  p100 {r.max()/1e3:.1f}")
