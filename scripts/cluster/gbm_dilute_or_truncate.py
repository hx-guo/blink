"""维持阈 20 时，GBM 的本底是被**抽稀**还是被**截断**？看形状，不看条数。

HXMT 的余量表只对抽稀有效，它特意标了 GBM 不能照搬——因子 212、切点掉到 0.094，
可能是截断、形状也变。本底唯一的用途是它的 **LST 直方形状**，所以判据就是：

    维持阈 20 之后留下的那段本底，LST 直方与现用本底（fa > 1）是否同形。

同形 ⇒ 抽稀，放心；不同形 ⇒ 截断，本底模型跟着变，要么抬阈要么换本底定义。

顺带出 Fano 那件事的第一步：候选的 k 中位与 λ 中位（决定该报测量还是外推）。
"""
import csv
import gzip
import math
import os

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
NBIN = 8
# 冒烟实测因子 115.8 / 212 / 9937 ⇒ 维持阈 20 的旧 fa 切点
CUT = 20 / 212.0


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def lst_of(r):
    t = datetime.strptime(r["start"].rstrip("Z").partition(".")[0], "%Y-%m-%dT%H:%M:%S")
    return (t.hour + t.minute / 60 + t.second / 3600 + wrap(float(r["lon"])) / 15) % 24


def hist(xs):
    h = [0] * NBIN
    for x in xs:
        h[min(NBIN - 1, max(0, int((x % 24) / 24 * NBIN)))] += 1
    return h


def norm(h):
    s = sum(h)
    return [x / s for x in h] if s else [0] * NBIN


def amp(p):
    return (max(p) - min(p)) / 2 / (sum(p) / len(p)) if sum(p) else 0.0


def chi2(o, e):
    """两个直方是否同形：把 e 归一到 o 的总数后算 χ²。"""
    n = sum(o)
    pe = norm(e)
    return sum((o[i] - n * pe[i]) ** 2 / max(n * pe[i], 1e-9) for i in range(NBIN))


def main():
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land = prep(g)
    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_fa"] = float(r["fa"])
        r["_land"] = land.contains(Point(wrap(float(r["lon"])), float(r["lat"])))

    print("=== Fano 那件事的第一步：候选的 k 与 λ ===")
    k = sorted(int(r["count"]) for r in rows)
    lam = sorted(float(r["mean"]) for r in rows)
    print(f"  count（k）中位 {k[len(k)//2]}，p10 {k[len(k)//10]}，p90 {k[9*len(k)//10]}")
    print(f"  mean（λ，两组之和）中位 {lam[len(lam)//2]:.3f}")
    print(f"  **注意 λ 是两组之和**，逐组约一半 ⇒ 逐组 λ 中位约 {lam[len(lam)//2]/2:.3f}\n")

    bands = [
        ("现用本底 fa > 1", lambda f: f > 1.0),
        (f"v2 新本底带 ⟺ 旧 fa ∈ ({CUT/20:.4f}, {CUT:.4f})", lambda f: CUT / 20 < f < CUT),
        (f"v2 留下的全池 ⟺ 旧 fa < {CUT:.4f}", lambda f: f < CUT),
        ("对照：fa ∈ (0.1, 1)", lambda f: 0.1 < f < 1.0),
    ]
    print("=== 判陆本底的 LST 直方：形状变没变 ===")
    ref = None
    print(f"{'档':<42s} {'n':>6s} {'A':>6s}  归一化 8 格")
    for label, sel in bands:
        sub = [lst_of(r) for r in rows if r["_land"] and sel(r["_fa"])]
        h = hist(sub)
        p = norm(h)
        if ref is None:
            ref = h
        print(f"{label:<42s} {sum(h):6d} {amp(p):6.3f}  " +
              " ".join(f"{x:.4f}" for x in p))
    print()
    print("=== 与现用本底同形吗（χ²/7，把现用本底当期望形状）===")
    for label, sel in bands[1:]:
        sub = [lst_of(r) for r in rows if r["_land"] and sel(r["_fa"])]
        h = hist(sub)
        c = chi2(h, ref)
        print(f"  {label:<42s} n={sum(h):6d}  χ² = {c:7.1f}/7"
              f"   {'同形' if c < 14 else '形状变了'}")
    print()
    print("  判读：χ²/7 在 14 以内（p > 0.05）算同形 ⇒ **抽稀**，本底模型不变，不用抬阈；")
    print("  显著超出 ⇒ **截断**，本底形状跟着变，要抬阈或改本底定义。")


if __name__ == "__main__":
    main()
