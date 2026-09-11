"""用 GBM 的分类器跑 HXMT 的 7890 个星下点，把分类器隔离成唯一变量。

点一样、纬度截断一样（|lat| < 26）⇒ 剩下的差别只能来自掩模与缓冲实现。
判读规则事先说好（he 定的）：结果接近 HXMT 圆盘版 ⇒ 差别是曝光地理，端点不能搬；
接近 GBM 自己的 24.8% ⇒ 差别是分类器，统一之后可以搬。
"""
import csv
import os

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))


def wrap(x):
    return x - 360 if x > 180 else x


def main():
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land, near = prep(g), prep(g.buffer(300.0 / 111.0))

    def klass(lon, lat):
        p = Point(wrap(lon), lat)
        return "land" if land.contains(p) else ("coast" if near.contains(p) else "ocean")

    rows = list(csv.DictReader(open(os.path.join(HERE, "subpoints_lat26.csv"))))
    acc = {}
    for r in rows:
        k = klass(float(r["longitude"]), float(r["latitude"]))
        acc.setdefault(r["group"], {"land": 0, "coast": 0, "ocean": 0})[k] += 1

    print("GBM 分类器（Natural Earth 50m + buffer(300/111 度)，星下点单点判陆）跑 HXMT 的点：")
    print(f"{'组':<6s} {'n':>5s}   {'陆地':>13s} {'近岸':>13s} {'远洋':>13s}")
    for gname in ("sig", "assoc", "bg"):
        c = acc[gname]
        n = sum(c.values())
        print(f"{gname:<6s} {n:5d}   " + "  ".join(
            f"{c[k]:5d}({100 * c[k] / n:5.1f}%)" for k in ("land", "coast", "ocean")))
    print()
    print("HXMT 自己的参考值：")
    print("  圆周 16 点  sig 45.8/31.1/23.1   assoc 30.8/34.8/34.5   bg 22.1/13.8/64.1")
    print("  圆盘 96 点  sig 45.8/33.1/21.1   assoc 30.8/37.9/31.4   bg 22.1/16.5/61.5")
    print("  GBM 自己的亚阈 600            bg 21.8/24.8/53.3")


if __name__ == "__main__":
    main()
