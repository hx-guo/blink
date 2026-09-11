"""WWLLN 的偏置有没有空间轴：在**盲搜 gamma 目录**内部一分为二直接量。

HXMT 量到「关联样本比未关联样本海得多」，但它没有不经台网的真值，所以分不清
那是台网偏置还是两个人群本来就不同。GBM 能分：**第二版目录的离线表是盲搜
gamma 数据来的，选样完全不经 WWLLN**，其中哪些带 WWLLN 关联是目录另给的一列。
所以「关联 vs 未关联」这一刀切在**同一个盲搜人群内部**，两边的真实地理分布
本应相同，任何差别只能来自台网的探测效率。

这同时是对我自己那个构成法的把关：如果 611 个正对照的类构成带台网偏置，
构成法的信号端点就是坏的。
"""
import csv
import os
from datetime import datetime

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
_C = [os.path.join(_REPO, "crates/instruments/blink_fermi_gbm/evidence/gbm_tgf_catalog"),
      os.path.join(HERE, "gbm_tgf_catalog")]
CAT = next((p for p in _C if os.path.isdir(p)), _C[0])
KINDS = ("land", "coast", "ocean")


def wrap(lon):
    return lon - 360 if lon > 180 else lon


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
    ww = {r[0].strip() for r in csv.reader(open(os.path.join(CAT, "gbm_tgf_catalog_wwlln.csv")))
          if r and not r[0].startswith("#")}

    for r in cat:
        r["_k"] = klass(float(r["Lon"]), float(r["Lat"]))
        r["_w"] = r["OS_ID"] in ww
        r["_lst"] = float(r["LST"]) * 24 if r["LST"] != "NULL" else None

    a = [r for r in cat if r["_w"]]
    b = [r for r in cat if not r["_w"]]
    print("=== 盲搜 gamma 目录内部一分为二（选样不经 WWLLN，所以两边真实地理应当相同）===")
    print(f"{'人群':<22s} {'n':>5s}  " + "  ".join(f"{k:>13s}" for k in KINDS))
    comp = {}
    for label, rs in (("带 WWLLN 关联", a), ("不带 WWLLN 关联", b), ("全体（盲搜）", cat)):
        c = {k: sum(1 for r in rs if r["_k"] == k) for k in KINDS}
        n = len(rs)
        comp[label] = [c[k] / n for k in KINDS]
        print(f"{label:<22s} {n:5d}  " +
              "  ".join(f"{c[k]:5d}({100 * c[k] / n:4.1f}%)" for k in KINDS))
    print()

    # 远洋相对几率比：关联样本相对未关联样本，远洋比陆地被多收多少
    pa, pb = comp["带 WWLLN 关联"], comp["不带 WWLLN 关联"]
    ratio = (pa[2] / pb[2]) / (pa[0] / pb[0])
    print(f"  远洋 / 陆地 的相对探测几率比 = ({pa[2]:.3f}/{pb[2]:.3f}) / "
          f"({pa[0]:.3f}/{pb[0]:.3f}) = {ratio:.2f}")
    print("  （> 1 表示台网更容易收到远洋 TGF 的母闪电；HXMT 用它自己的两个人群估到 2.2）")
    print()

    # 2x2 卡方（陆地 vs 远洋，关联 vs 未关联）
    na_l = sum(1 for r in a if r["_k"] == "land")
    na_o = sum(1 for r in a if r["_k"] == "ocean")
    nb_l = sum(1 for r in b if r["_k"] == "land")
    nb_o = sum(1 for r in b if r["_k"] == "ocean")
    tot = na_l + na_o + nb_l + nb_o
    chi = 0.0
    for obs, e in ((na_l, (na_l + na_o) * (na_l + nb_l) / tot),
                   (na_o, (na_l + na_o) * (na_o + nb_o) / tot),
                   (nb_l, (nb_l + nb_o) * (na_l + nb_l) / tot),
                   (nb_o, (nb_l + nb_o) * (na_o + nb_o) / tot)):
        chi += (obs - e) ** 2 / e
    print(f"  只看陆地 vs 远洋的 2x2：关联 {na_l}/{na_o}，未关联 {nb_l}/{nb_o}，"
          f"chi2 = {chi:.1f}/1")
    print()

    # 时间轴：同一刀切下的 LST 峰
    import math

    def rayleigh(xs):
        n = len(xs)
        c = sum(math.cos(2 * math.pi * x / 24) for x in xs) / n
        s = sum(math.sin(2 * math.pi * x / 24) for x in xs) / n
        r = math.hypot(c, s)
        return r, (math.degrees(math.atan2(s, c)) / 15) % 24
    print("=== 同一刀切下的时间轴（对照：这是已知的 VLF 夜间偏置）===")
    for label, rs in (("带 WWLLN 关联", a), ("不带 WWLLN 关联", b)):
        xs = [r["_lst"] for r in rs if r["_lst"] is not None]
        r_, ph = rayleigh(xs)
        print(f"  {label:<20s} n={len(xs):5d}  峰 {ph:5.2f} LT   R={r_:.3f}")
    print()
    print("=== 把两轴分开：只在陆地类内部比 LST，只在夜间比构成 ===")
    for k in KINDS:
        xs = [r["_lst"] for r in a if r["_k"] == k and r["_lst"] is not None]
        ys = [r["_lst"] for r in b if r["_k"] == k and r["_lst"] is not None]
        if len(xs) < 30 or len(ys) < 30:
            continue
        ra, pa_ = rayleigh(xs)
        rb, pb_ = rayleigh(ys)
        print(f"  {k:<7s} 关联 n={len(xs):5d} 峰 {pa_:5.2f}   "
              f"未关联 n={len(ys):5d} 峰 {pb_:5.2f}   差 {(pa_ - pb_) % 24:5.2f} h")


if __name__ == "__main__":
    main()
