"""三类 f 的误差棒、显著性，以及远洋那一档到底靠什么撑着。

三件事，按重要性排：

1. **拿同类的正对照刻度**。611 个目录匹配真 TGF 的真值 f = 1，所以「用 k 类模板
   拟 k 类正对照」给出的就是该类模板的**回收率**。远洋模板在 f = 1 的人群上只
   回收 0.396——这一档的刻度掉了 2.5 倍，绝对值不能直接读。
2. **f ≠ 0 到底显不显著**：profile likelihood 的 Δχ²（1 自由度），不是靠误差棒目测。
3. **远洋的 f 由哪几格撑着**：逐格拆 Δlnℒ = ln[f·T+(1−f)·B] − ln B。
   正贡献来自模板 TGF 高于本底的格子才算「模板峰把它拟起来的」；
   若正贡献集中在模板的**低谷**格（观测在那儿缺口），那是非峰位结构带着走。
"""
import csv
import gzip
import math
import os
import random
import zlib
from datetime import datetime

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
# 目录表：仓库里在 evidence/ 下；脚本被拷到集群跑数目录时就放在脚本旁边。
# 两处都找，找不到再报错，省得每次改常量。
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CANDIDATES = [
    os.path.join(_REPO, "crates/instruments/blink_fermi_gbm/evidence",
                 "gbm_tgf_catalog/gbm_tgf_catalog_offline.csv"),
    os.path.join(HERE, "gbm_tgf_catalog/gbm_tgf_catalog_offline.csv"),
    os.path.join(HERE, "gbm_tgf_catalog_offline.csv"),
]
CAT = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
NBIN = 8
KINDS = ("land", "coast", "ocean")


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def lst_of(row):
    head = row["start"].rstrip("Z").partition(".")[0]
    t = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    return (t.hour + t.minute / 60 + t.second / 3600 + wrap(float(row["lon"])) / 15) % 24


def hist(lsts):
    h = [0] * NBIN
    for x in lsts:
        h[min(NBIN - 1, max(0, int((x % 24) / 24 * NBIN)))] += 1
    return h


def normalise(h, floor=1e-6):
    total = sum(h)
    return [max(x / total, floor) for x in h]


def amplitude(p):
    """全队统一刻度 A = (max − min)/2/均值；8 格归一化时 = (max − min) × 4。"""
    return (max(p) - min(p)) / 2 / (sum(p) / len(p))


def rayleigh(lsts):
    n = len(lsts)
    if n == 0:
        return 0.0, 0.0, 1.0
    c = sum(math.cos(2 * math.pi * x / 24) for x in lsts) / n
    s = sum(math.sin(2 * math.pi * x / 24) for x in lsts) / n
    r = math.hypot(c, s)
    return r, (math.degrees(math.atan2(s, c)) / 15) % 24, math.exp(-n * r * r)


def loglike(counts, tgf, bkg, f):
    return sum(n * math.log(f * t + (1 - f) * b)
               for n, t, b in zip(counts, tgf, bkg) if n)


def fit(counts, tgf, bkg):
    best_f, best_ll = 0.0, -1e300
    for i in range(1001):
        f = i / 1000.0
        ll = loglike(counts, tgf, bkg, f)
        if ll > best_ll:
            best_ll, best_f = ll, f
    return best_f, best_ll


def profile(counts, tgf, bkg):
    """Δχ² = 2(lnℒ_max − lnℒ_{f=0})，1 自由度。并给 profile 的 68% 区间。"""
    f, ll = fit(counts, tgf, bkg)
    d = 2 * (ll - loglike(counts, tgf, bkg, 0.0))
    lo, hi = f, f
    for i in range(1001):
        x = i / 1000.0
        if 2 * (ll - loglike(counts, tgf, bkg, x)) <= 1.0:
            lo, hi = min(lo, x), max(hi, x)
    return f, d, lo, hi


def boot_ci(lsts, tgf, bkg, n=500, seed=0):
    rng = random.Random(seed)
    out = [fit(hist([rng.choice(lsts) for _ in lsts]), tgf, bkg)[0] for _ in range(n)]
    out.sort()
    return out[int(0.16 * n)], out[int(0.84 * n)]


def main():
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
    tmpl, tmpl_raw, tmpl_stat = {}, {}, {}
    for k in KINDS:
        lst = [float(r["LST"]) * 24 for r in src if klass(float(r["Lon"]), float(r["Lat"])) == k]
        tmpl_raw[k] = hist(lst)
        tmpl[k] = normalise(tmpl_raw[k])
        tmpl_stat[k] = (len(lst),) + rayleigh(lst)

    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        pool = list(csv.DictReader(fh))
    bkg_rows = [r for r in pool if float(r["fa"]) > 1.0]
    bkg, bkg_raw = {}, {}
    for k in KINDS:
        sub = [lst_of(r) for r in bkg_rows if klass(float(r["lon"]), float(r["lat"])) == k]
        bkg_raw[k] = hist(sub)
        bkg[k] = normalise(bkg_raw[k])

    print("=== 模板与本底的刻度（A = (max−min)/2/均值，全队统一）===")
    print(f"{'类':<7s} {'模板 n':>7s} {'A':>6s} {'R':>6s} {'峰 LT':>7s} {'p':>9s}   "
          f"{'本底 n':>8s} {'A':>6s}")
    for k in KINDS:
        n, r, ph, p = tmpl_stat[k]
        print(f"{k:<7s} {n:7d} {amplitude(tmpl[k]):6.3f} {r:6.3f} {ph:7.2f} {p:9.1e}   "
              f"{sum(bkg_raw[k]):8d} {amplitude(bkg[k]):6.3f}")
    print()

    pops = (("正对照 611（真值 f = 1，给的是模板回收率）", "matched_sig_2014b.csv"),
            ("多出来的显著 235", "extras_sig_2014b.csv"),
            ("负对照 600（真值 f ≈ 0）", "weak_2014b.csv"))
    res = {}
    for label, name in pops:
        rs = list(csv.DictReader(open(os.path.join(HERE, name))))
        by = {k: [] for k in KINDS}
        for r in rs:
            by[klass(float(r["lon"]), float(r["lat"]))].append(lst_of(r))
        print(f"--- {label} ---")
        print(f"{'类':<7s} {'n':>4s} {'f':>6s} {'profile 68%':>16s} {'bootstrap 68%':>16s} "
              f"{'Δχ²(1dof)':>10s}  {'自身峰 LT':>9s} {'自身 R':>7s} {'自身 p':>9s}")
        res[name] = {}
        for k in KINDS:
            lst = by[k]
            if len(lst) < 20:
                continue
            f, d, plo, phi = profile(hist(lst), tmpl[k], bkg[k])
            blo, bhi = boot_ci(lst, tmpl[k], bkg[k], seed=zlib.crc32((name + k).encode()))
            r, ph, p = rayleigh(lst)
            res[name][k] = (len(lst), f, d, plo, phi, blo, bhi)
            print(f"{k:<7s} {len(lst):4d} {f:6.3f} [{plo:5.3f},{phi:5.3f}]  "
                  f"[{blo:5.3f},{bhi:5.3f}]  {d:10.1f}  {ph:9.2f} {r:7.3f} {p:9.1e}")
        print()

    print("=== 按同类正对照的回收率刻度归一（f_cal = f_extras / f_matched）===")
    print("  正对照真值 f = 1，所以它拟出来的就是该类模板的回收率；除掉它才是可比的纯度。")
    for k in KINDS:
        ne, fe, _, _, _, _, _ = res["extras_sig_2014b.csv"][k]
        nm, fm, _, _, _, _, _ = res["matched_sig_2014b.csv"][k]
        print(f"  {k:<7s} f_extras={fe:.3f} (n={ne})  ÷  回收率 {fm:.3f} (n={nm})  "
              f"=  {fe / fm if fm else float('nan'):.3f}")
    print()

    print("=== 远洋那一档：f 由哪几格撑着（Δlnℒ 逐格拆）===")
    ex = list(csv.DictReader(open(os.path.join(HERE, "extras_sig_2014b.csv"))))
    oc = [lst_of(r) for r in ex if klass(float(r["lon"]), float(r["lat"])) == "ocean"]
    obs = hist(oc)
    n = sum(obs)
    f, d, plo, phi = profile(obs, tmpl["ocean"], bkg["ocean"])
    print(f"  n={n}  f={f:.3f}  Δχ²={d:.2f}（1 自由度）  profile 68% [{plo:.3f},{phi:.3f}]")
    print(f"  {'格':<8s} {'观测':>5s} {'本底期望':>9s} {'拟合期望':>9s} {'T/B':>6s} {'Δlnℒ':>8s}  说明")
    tot = 0.0
    for i in range(NBIN):
        t, b = tmpl["ocean"][i], bkg["ocean"][i]
        dll = obs[i] * (math.log(f * t + (1 - f) * b) - math.log(b)) if obs[i] else 0.0
        tot += dll
        tag = "模板峰位" if t / b > 1.1 else ("模板低谷" if t / b < 0.9 else "模板持平")
        print(f"  {3 * i:02d}-{3 * i + 3:02d}h  {obs[i]:5d} {n * b:9.1f} "
              f"{n * (f * t + (1 - f) * b):9.1f} {t / b:6.2f} {dll:+8.3f}  {tag}")
    print(f"  合计 Δlnℒ = {tot:+.3f}  → Δχ² = {2 * tot:.2f}")


if __name__ == "__main__":
    main()
