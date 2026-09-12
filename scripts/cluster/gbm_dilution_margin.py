"""直接量：把本底换成 v2 维持阈 20 之后剩下的那 565 个，f 与误差棒变多少。

比"算比值"硬——HXMT 就是这么做的（本底抽稀 20 倍，半宽 0.0770 → 0.0760）。
误差棒用与第 25 条同一套零分布（从本底抽 n 个去拟剩下的本底），不用似然区间。
"""
import csv
import gzip
import math
import os
import random
from datetime import datetime

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "gbm_tgf_catalog", "gbm_tgf_catalog_offline.csv")
NBIN = 8
CUT = 20 / 212.0
NTRIAL = 400


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


def norm(h, floor=1e-6):
    s = sum(h)
    return [max(x / s, floor) for x in h]


def fit(c, t, b):
    bf, bl = 0.0, -1e300
    for i in range(1001):
        f = i / 1000.0
        ll = sum(n * math.log(f * x + (1 - f) * y) for n, x, y in zip(c, t, b) if n)
        if ll > bl:
            bl, bf = ll, f
    return bf


def main():
    rng = random.Random(20260911)
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land = prep(g)

    rows = list(csv.reader(open(CAT)))
    hdr = [h.strip().lstrip("#") for h in rows[0]]
    cat = [dict(zip(hdr, [x.strip() for x in r])) for r in rows[1:]]
    src = [r for r in cat if r["LST"] != "NULL" and r["Date"][:4] != "2014"
           and "2010" <= r["Date"][:4] <= "2016"
           and land.contains(Point(wrap(float(r["Lon"])), float(r["Lat"])))]
    tmpl = norm(hist([float(r["LST"]) * 24 for r in src]))

    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        pool = list(csv.DictReader(fh))
    bl = [r for r in pool
          if land.contains(Point(wrap(float(r["lon"])), float(r["lat"])))]

    variants = [
        ("现用本底 fa > 1", [lst_of(r) for r in bl if float(r["fa"]) > 1.0]),
        ("v2 维持阈 20 的新本底带", [lst_of(r) for r in bl
                                if CUT / 20 < float(r["fa"]) < CUT]),
    ]
    ex = list(csv.DictReader(open(os.path.join(HERE, "extras_sig_2014b.csv"))))
    sig = [lst_of(r) for r in ex
           if land.contains(Point(wrap(float(r["lon"])), float(r["lat"])))]
    print(f"信号（多出来的显著，判陆）n = {len(sig)}\n")
    print(f"{'本底口径':<26s} {'本底 n':>7s} {'比值':>6s} {'f':>7s} {'零 σ':>7s} "
          f"{'偏离':>7s} {'95% 半宽':>9s}")
    for label, bgl in variants:
        bg = norm(hist(bgl))
        f = fit(hist(sig), tmpl, bg)
        nulls = []
        for _ in range(NTRIAL):
            idx = set(rng.sample(range(len(bgl)), len(sig)))
            sub = [bgl[i] for i in idx]
            rest = [bgl[i] for i in range(len(bgl)) if i not in idx]
            nulls.append(fit(hist(sub), tmpl, norm(hist(rest))))
        mu = sum(nulls) / len(nulls)
        sd = (sum((x - mu) ** 2 for x in nulls) / (len(nulls) - 1)) ** 0.5
        # 95% 半宽用自助重抽信号
        bs = []
        for _ in range(NTRIAL):
            s = [rng.choice(sig) for _ in sig]
            bs.append(fit(hist(s), tmpl, bg))
        bs.sort()
        half = (bs[int(0.975 * NTRIAL) - 1] - bs[int(0.025 * NTRIAL)]) / 2
        print(f"{label:<26s} {len(bgl):7d} {len(bgl)/len(sig):6.1f} {f:7.3f} "
              f"{sd:7.3f} {(f-mu)/sd:6.1f}σ {half:9.4f}")
    print()
    print("  判读：若 f、零 σ、95% 半宽三者都基本不动 ⇒ 抽稀无害，维持阈 20 成立。")


if __name__ == "__main__":
    main()
