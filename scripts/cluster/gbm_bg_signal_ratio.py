"""抬不抬阈，按「本底/信号」判，不按「本底掉了多少倍」判。

HXMT 实测：本底抽稀 20 倍，f 的 95% 半宽 0.0770 → 0.0760 几乎不动；
要到「本底/信号 < 0.3」才开始变差。所以掉多少倍本身不构成抬阈的理由。

维持发射阈 20 时，v2 留下的是旧 fa < 20/因子 的那些。冒烟实测因子
最小 115.8、中位 212、最大 9937 ⇒ 旧 fa 切点在 [20/9937, 20/115.8] = [0.0020, 0.173]，
中位对应 0.094。三个切点都算一遍，给出区间而不是单点。
"""
import csv
import gzip
import os

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
CUTS = [("最大因子 9937", 20 / 9937.0), ("中位因子 212", 20 / 212.0),
        ("最小因子 115.8", 20 / 115.8)]
# 实测的三个人群规模（陆地那一档是唯一有分辨力的，所以按陆地算最严）
SIG = {"多出来的显著 235（陆 84）": (235, 84),
       "正对照 611（陆 264）": (611, 264),
       "负对照 600（陆 131）": (600, 131)}


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def main():
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land = prep(g)

    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        rows = [(float(r["fa"]), float(r["lon"]), float(r["lat"])) for r in csv.DictReader(fh)]
    print(f"v1 2014 全池 {len(rows)} 个候选")
    isl = [land.contains(Point(wrap(lo), la)) for _f, lo, la in rows]
    nland = sum(isl)
    print(f"  其中判陆 {nland}")
    bg1 = sum(1 for (f, _lo, _la), l in zip(rows, isl) if f > 1.0 and l)
    print(f"  现用本底（fa > 1 且判陆）{bg1}\n")

    print("=== 维持发射阈 20 时，v2 还剩多少（按旧 fa 切点估）===")
    print(f"{'口径':<18s} {'旧 fa 切点':>11s} {'全池':>8s} {'判陆':>8s}   "
          f"{'新本底带判陆':>12s}")
    print("  新本底带 = 新 fa ∈ (1, 20] ⟺ 旧 fa ∈ (切点/20, 切点]")
    keep = {}
    for label, cut in CUTS:
        tot = sum(1 for f, _lo, _la in rows if f < cut)
        lnd = sum(1 for (f, _lo, _la), l in zip(rows, isl) if f < cut and l)
        band = sum(1 for (f, _lo, _la), l in zip(rows, isl)
                   if cut / 20.0 < f < cut and l)
        keep[label] = (tot, lnd, band)
        print(f"{label:<18s} {cut:11.4f} {tot:8d} {lnd:8d}   {band:12d}")
    print()

    print("=== 判据：本底/信号（只要 > 0.3 就不用抬）===")
    print(f"{'口径':<18s} {'本底(判陆)':>10s}   " +
          "  ".join(f"{k.split('（')[0]:>16s}" for k in SIG))
    for label, _cut in CUTS:
        tot, lnd, band = keep[label]
        # 保守起见用「新本底带」那一列当本底，不用整个留下的池
        cells = []
        for _k, (_n_all, n_land) in SIG.items():
            cells.append(f"{band / n_land:16.1f}")
        print(f"{label:<18s} {band:10d}   " + "  ".join(cells))
    print()
    print("  对照：现在（v1，fa > 1 判陆 %d）的本底/信号" % bg1)
    print("    " + "   ".join(f"{k.split('（')[0]} {bg1 / v[1]:.0f}" for k, v in SIG.items()))
    print()
    print("  注：本底与信号都按**判陆**算，因为陆地是唯一有分辨力的一档，")
    print("  其余两类的模板拟不出 f（第 17、25 条），本底再多也没用。")


if __name__ == "__main__":
    main()
