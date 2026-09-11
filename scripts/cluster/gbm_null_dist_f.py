"""似然区间不能当显著性：用重抽样建零分布，重判三类 f。

统筹的更正：profile likelihood 的 Δχ²(1 dof) 假定了渐近正态，在 f 被截在 [0,1]、
模板与本底几乎正交的小 N 情形下**区间偏窄**。SVOM 已经因此撤回过一条结论。

正确做法（SVOM 的）：**从本底里抽同样大小的子样本，拿它去拟剩下的本底**，
重复多次建零分布——这样零假设（纯本底）是由数据自己实现的，不靠渐近。
拟合时用的本底模板必须是**剩下那部分**，否则被检验样本自己进了本底模板，是循环的。
"""
import csv
import gzip
import math
import os
import random

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "gbm_tgf_catalog", "gbm_tgf_catalog_offline.csv")
NBIN = 8
KINDS = ("land", "coast", "ocean")
NTRIAL = 500


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def lst_of(row):
    head = row["start"].rstrip("Z").partition(".")[0]
    from datetime import datetime
    t = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    return (t.hour + t.minute / 60 + t.second / 3600 + wrap(float(row["lon"])) / 15) % 24


def hist(xs):
    h = [0] * NBIN
    for x in xs:
        h[min(NBIN - 1, max(0, int((x % 24) / 24 * NBIN)))] += 1
    return h


def normalise(h, floor=1e-6):
    s = sum(h)
    return [max(x / s, floor) for x in h]


def fit(counts, tgf, bkg):
    best_f, best_ll = 0.0, -1e300
    for i in range(1001):
        f = i / 1000.0
        ll = sum(n * math.log(f * t + (1 - f) * b)
                 for n, t, b in zip(counts, tgf, bkg) if n)
        if ll > best_ll:
            best_ll, best_f = ll, f
    return best_f


def main():
    rng = random.Random(20260911)
    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land, near = prep(g), prep(g.buffer(300.0 / 111.0))

    def klass(lon, lat):
        p = Point(wrap(lon), lat)
        return "land" if land.contains(p) else ("coast" if near.contains(p) else "ocean")

    rows = list(csv.reader(open(CAT)))
    hdr = [h.strip().lstrip("#") for h in rows[0]]
    cat = [dict(zip(hdr, [x.strip() for x in r])) for r in rows[1:]]
    src = [r for r in cat if r["LST"] != "NULL" and r["Date"][:4] != "2014"
           and "2010" <= r["Date"][:4] <= "2016"]
    tmpl = {k: normalise(hist([float(r["LST"]) * 24 for r in src
                               if klass(float(r["Lon"]), float(r["Lat"])) == k]))
            for k in KINDS}

    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        pool = list(csv.DictReader(fh))
    bg = {k: [] for k in KINDS}
    for r in pool:
        if float(r["fa"]) > 1.0:
            bg[klass(float(r["lon"]), float(r["lat"]))].append(lst_of(r))

    obs = {}
    for name, label in (("extras_sig_2014b.csv", "多出来的显著 235"),
                        ("matched_sig_2014b.csv", "正对照 611")):
        rs = list(csv.DictReader(open(os.path.join(HERE, name))))
        d = {k: [] for k in KINDS}
        for r in rs:
            d[klass(float(r["lon"]), float(r["lat"]))].append(lst_of(r))
        obs[label] = d

    print("=== 零分布（从本底抽 n 个去拟剩下的本底），%d 次 ===" % NTRIAL)
    print(f"{'人群':<16s} {'类':<7s} {'n':>4s} {'实测 f':>7s} {'零均值':>7s} {'零 σ':>7s} "
          f"{'偏离':>7s}  {'零分布 ≥ 实测的占比':>10s}")
    for label, d in obs.items():
        for k in KINDS:
            n = len(d[k])
            if n < 20:
                continue
            f_obs = fit(hist(d[k]), tmpl[k], normalise(hist(bg[k])))
            nulls = []
            for _ in range(NTRIAL):
                idx = set(rng.sample(range(len(bg[k])), n))
                sub = [bg[k][i] for i in idx]
                rest = [bg[k][i] for i in range(len(bg[k])) if i not in idx]
                nulls.append(fit(hist(sub), tmpl[k], normalise(hist(rest))))
            mu = sum(nulls) / len(nulls)
            sd = (sum((x - mu) ** 2 for x in nulls) / (len(nulls) - 1)) ** 0.5
            p = sum(1 for x in nulls if x >= f_obs) / len(nulls)
            print(f"{label:<16s} {k:<7s} {n:4d} {f_obs:7.3f} {mu:7.3f} {sd:7.3f} "
                  f"{(f_obs - mu) / sd if sd else float('nan'):6.1f}σ  "
                  f"{p:10.4f}{'  (0/%d)' % NTRIAL if p == 0 else ''}")
        print()


if __name__ == "__main__":
    main()
