"""只用陆/近岸/远洋的**构成比**估 f——不用 LST，也不用台网。

起因是 SVOM 指出我一个前提错了：我拿"纯度不该依赖候选落在哪儿"当联合拟合
（三类共享一个 f）的依据，**这是错的**。纯度 = 该类 TGF 率 /（TGF 率 + 本底率），
**TGF 产生密度随下垫面变，而本底不跟着同一张地图变**，所以 f 本来就该在海上更低。

但这条恰好送来一个新的估计量：如果 f 在海上更低，那么一批混合样本的**类构成比**
就落在"纯 TGF 的构成比"和"纯本底的构成比"之间，**位置由混合比例决定**。
三个人群的构成比我都有：

- 正对照 611 个已认证真 TGF → 纯 TGF 的类构成
- 亚阈负对照 600 个        → 纯本底的类构成
- 多出来的显著 235 个      → 待测混合

**这个估计量不碰 LST、不碰 WWLLN、不用任何模板**，只用每个候选星下点落在哪一类。
它与 LST 那条腿共用的只有"分类规则"本身，是第三条独立的读数。

多项式似然：观测的 (n_land, n_coast, n_ocean) ~ Multinomial(N, f·P_tgf + (1−f)·P_bkg)。
"""
import csv
import os
import sys
from math import log

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("land", "coast", "ocean")


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def composition(rows, klass):
    c = {k: 0 for k in KINDS}
    for r in rows:
        c[klass(float(r["lon"]), float(r["lat"]))] += 1
    n = sum(c.values())
    return c, [c[k] / n for k in KINDS], n


def loglike(obs, p_tgf, p_bkg, f):
    ll = 0.0
    for o, t, b in zip(obs, p_tgf, p_bkg):
        p = f * t + (1 - f) * b
        if o:
            ll += o * log(max(p, 1e-12))
    return ll


def fit(obs, p_tgf, p_bkg):
    grid = [i / 1000.0 for i in range(1001)]
    lls = [loglike(obs, p_tgf, p_bkg, f) for f in grid]
    best = max(range(1001), key=lambda i: lls[i])
    lo = hi = grid[best]
    for i, f in enumerate(grid):
        if 2 * (lls[best] - lls[i]) <= 1.0:
            lo, hi = min(lo, f), max(hi, f)
    return grid[best], 2 * (lls[best] - lls[0]), lo, hi


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else HERE
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land, near = prep(g), prep(g.buffer(300.0 / 111.0))

    def klass(lon, lat):
        p = Point(wrap(lon), lat)
        return "land" if land.contains(p) else ("coast" if near.contains(p) else "ocean")

    pops = {}
    for label, name in (("正对照 611（纯 TGF 的类构成）", "matched_sig_2014b.csv"),
                        ("负对照 600（纯本底的类构成）", "weak_2014b.csv"),
                        ("多出来的显著 235（待测混合）", "extras_sig_2014b.csv")):
        rows = list(csv.DictReader(open(os.path.join(d, name))))
        pops[name] = composition(rows, klass)
        c, p, n = pops[name]
        print(f"{label:<32s} n={n:4d}   " +
              "  ".join(f"{k} {c[k]:4d}({100 * p[i]:4.1f}%)" for i, k in enumerate(KINDS)))
    print()

    _c, p_tgf, _n = pops["matched_sig_2014b.csv"]
    _c, p_bkg, _n = pops["weak_2014b.csv"]
    obs, _p, n = pops["extras_sig_2014b.csv"]
    obs = [obs[k] for k in KINDS]

    f, d2, lo, hi = fit(obs, p_tgf, p_bkg)
    print("=== 只用类构成比的两成分拟合 ===")
    print(f"  f = {f:.3f}   profile 68% [{lo:.3f}, {hi:.3f}]   Δχ² = {d2:.1f}（1 自由度）"
          f"   → {f * n:.0f} / {n} 个")
    print()
    print(f"  {'类':<7s} {'观测':>5s} {'纯本底期望':>10s} {'纯 TGF 期望':>11s} {'拟合期望':>9s}")
    for i, k in enumerate(KINDS):
        print(f"  {k:<7s} {obs[i]:5d} {n * p_bkg[i]:10.1f} {n * p_tgf[i]:11.1f} "
              f"{n * (f * p_tgf[i] + (1 - f) * p_bkg[i]):9.1f}")
    print()
    print("  自检：把正对照自己当待测（真值 f = 1）与把负对照当待测（真值 f = 0）")
    for label, name in (("正对照（应给 1）", "matched_sig_2014b.csv"),
                        ("负对照（应给 0）", "weak_2014b.csv")):
        c, _p, n2 = pops[name]
        o2 = [c[k] for k in KINDS]
        f2, d3, lo2, hi2 = fit(o2, p_tgf, p_bkg)
        print(f"    {label:<18s} f = {f2:.3f} [{lo2:.3f}, {hi2:.3f}]   Δχ² = {d3:.1f}")
    print()
    print("  注意：正/负对照当待测是**循环的**（它们定义了两个端点），")
    print("  所以上面两行只验算法不验数据——给不出 1 和 0 才说明代码错了。")


if __name__ == "__main__":
    main()
