"""对比度损失（R 0.214 → 0.151）按陆/海拆开：损失集中在哪一类。

统筹要的：HXMT 量到夜间偏置只在陆地（+2.21 h）、远洋为零，问我这边的 R 损失
是不是也集中在陆地。在盲搜目录内部切关联/未关联，逐类算圆均值 R。
"""
import csv
import math
import os

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "gbm_tgf_catalog")


def wrap(x):
    return x - 360 if x > 180 else x


def ray(xs):
    n = len(xs)
    if n < 5:
        return 0.0, 0.0, n
    c = sum(math.cos(2 * math.pi * x / 24) for x in xs) / n
    s = sum(math.sin(2 * math.pi * x / 24) for x in xs) / n
    r = math.hypot(c, s)
    return r, (math.degrees(math.atan2(s, c)) / 15) % 24, n


def main():
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land, near = prep(g), prep(g.buffer(300.0 / 111.0))

    def klass(lon, lat):
        p = Point(wrap(lon), lat)
        return "land" if land.contains(p) else ("coast" if near.contains(p) else "ocean")

    rows = list(csv.reader(open(os.path.join(CAT, "gbm_tgf_catalog_offline.csv"))))
    hdr = [h.strip().lstrip("#") for h in rows[0]]
    cat = [dict(zip(hdr, [x.strip() for x in r])) for r in rows[1:]]
    ww = {r[0].strip() for r in
          csv.reader(open(os.path.join(CAT, "gbm_tgf_catalog_wwlln.csv")))
          if r and not r[0].startswith("#")}
    for r in cat:
        r["_k"] = klass(float(r["Lon"]), float(r["Lat"]))
        r["_w"] = r["OS_ID"] in ww
        r["_lst"] = float(r["LST"]) * 24 if r["LST"] != "NULL" else None

    print("对比度 R：关联 vs 未关联，逐类（盲搜目录内部切，圆均值口径）")
    print(f"{'类':<8s} {'关联 n':>7s} {'R':>7s} {'峰':>6s}   "
          f"{'未关联 n':>9s} {'R':>7s} {'峰':>6s}   {'ΔR':>7s} {'相对损失':>9s}")
    for k in ("land", "coast", "ocean", "ALL"):
        a = [r["_lst"] for r in cat if r["_w"] and r["_lst"] is not None
             and (k == "ALL" or r["_k"] == k)]
        b = [r["_lst"] for r in cat if not r["_w"] and r["_lst"] is not None
             and (k == "ALL" or r["_k"] == k)]
        ra, pa, na = ray(a)
        rb, pb, nb = ray(b)
        print(f"{k:<8s} {na:7d} {ra:7.3f} {pa:6.2f}   {nb:9d} {rb:7.3f} {pb:6.2f}   "
              f"{ra - rb:+7.3f} {100 * (ra - rb) / rb:+8.1f}%")


if __name__ == "__main__":
    main()
