"""已发布的 GBM 目录把一个物理事件数了几次？对模板与完备性分母各影响多少。

HXMT 在自己的目录里实测 3.9% 的行落在 100 ms 内的多脉冲组里、最多一个事件被数 7 次、
全目录膨胀 1.72%。TGF 多脉冲是常见现象，Fermi 目录很可能同样，而我这边**模板
（1846 个陆地条目）、完备性分母（三年 1941 个）全都是从目录来的**，所以：

- 模板的 LST 直方会被多脉冲组重复加权 ⇒ 有效 N 比名义 N 小、A 被摊平；
- 完备性分母偏大。

按 MET 邻近度分组（同一探测器视场、毫秒量级），量几个尺度，再看去重对
陆地模板的 8 格与三年完备性分母的实际影响。
"""
import csv
import math
import os

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "gbm_tgf_catalog", "gbm_tgf_catalog_offline.csv")
NBIN = 8


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def hist(xs):
    h = [0] * NBIN
    for x in xs:
        h[min(NBIN - 1, max(0, int((x % 24) / 24 * NBIN)))] += 1
    return h


def normalise(h):
    s = sum(h)
    return [x / s for x in h]


def amplitude(p):
    return (max(p) - min(p)) / 2 / (sum(p) / len(p))


def group_by(mets, gap):
    """按 MET 邻近度切组，返回每组的下标列表。"""
    order = sorted(range(len(mets)), key=lambda i: mets[i])
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if mets[i] - mets[cur[-1]] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def main():
    rows = list(csv.reader(open(CAT)))
    hdr = [h.strip().lstrip("#") for h in rows[0]]
    cat = [dict(zip(hdr, [x.strip() for x in r])) for r in rows[1:]]
    mets = [float(r["MET"]) for r in cat]
    print(f"目录离线表 {len(cat)} 行\n")

    print("=== 多脉冲重复：按 MET 邻近度切组 ===")
    print(f"{'尺度':>8s} {'组数':>6s} {'多行组':>7s} {'落在多行组里的行':>16s} {'最大组':>6s} {'膨胀':>7s}")
    for gap, label in ((0.001, "1 ms"), (0.01, "10 ms"), (0.1, "100 ms"),
                       (1.0, "1 s"), (10.0, "10 s")):
        gs = group_by(mets, gap)
        multi = [g for g in gs if len(g) > 1]
        nrow = sum(len(g) for g in multi)
        infl = (len(cat) - len(gs)) / len(gs) * 100
        print(f"{label:>8s} {len(gs):6d} {len(multi):7d} {nrow:9d}({100 * nrow / len(cat):5.2f}%)"
              f" {max(len(g) for g in gs):6d} {infl:6.2f}%")
    print()

    # 100 ms 口径下去重：每组留 NAI+BGO 计数最大的那条
    gs = group_by(mets, 0.1)
    keep = []
    for g in gs:
        keep.append(max(g, key=lambda i: float(cat[i]["BGO_0_N"]) + float(cat[i]["BGO_1_N"])
                        + float(cat[i]["NAI_N"])))
    print(f"=== 100 ms 口径去重：{len(cat)} 行 → {len(keep)} 个物理事件 ===\n")

    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land = prep(g)

    def is_land(r):
        return land.contains(Point(wrap(float(r["Lon"])), float(r["Lat"])))

    for label, idx in (("去重前（现在供给全队的）", list(range(len(cat)))),
                       ("去重后", keep)):
        lst = [float(cat[i]["LST"]) * 24 for i in idx
               if cat[i]["LST"] != "NULL" and is_land(cat[i])]
        h = hist(lst)
        p = normalise(h)
        print(f"陆地模板 {label}: n = {len(lst)}")
        print(f"  整数计数 {h}")
        print(f"  归一化 {' '.join('%.4f' % x for x in p)}   A = {amplitude(p):.3f}")
        if label == "去重后":
            print(f"  与去重前的归一值最大差 "
                  f"{max(abs(a - b) for a, b in zip(p, p0)):.4f}")
        else:
            p0 = p
        print()

    # 完备性分母：三年
    y3 = [i for i in range(len(cat)) if cat[i]["Date"][:4] in ("2014", "2015", "2016")]
    k3 = [i for i in keep if cat[i]["Date"][:4] in ("2014", "2015", "2016")]
    print(f"=== 完备性分母（三年 2014+2015+2016H1）===")
    print(f"  去重前 {len(y3)} 行 → 去重后 {len(k3)} 个物理事件"
          f"（分母偏大 {100 * (len(y3) - len(k3)) / len(k3):.2f}%）")
    print()
    print("  注意：**去重会同时改分子**——被合并掉的那些行若各自都被我们找到，")
    print("  分子也要按同样的规则合并。所以完备性百分比的变化远小于分母的变化，")
    print("  不能只把分母换掉。")


if __name__ == "__main__":
    main()
