"""给 HXMT 的陆地模板配套包：原始计数 + 同口径本底 + 成串邻居数。

三样都按 he 要的口径出：
1. 陆地模板那批的**本底 LST 直方**，与模板同口径（同样的星下点单点判陆），
   并额外给**按模板经度分布重加权**的版本——LST = UT + 经度/15，本底在它自己的
   经度分布下平不等于在被检验样本的经度分布下也平。
2. **成串**：模板来自已发表目录，我们没有、也不能对它做 is_train 摘除
   （那是我们自己候选池上的判据）。改给逐条的 ±10 min 目录内邻居数，he 自己摘。
3. **每格原始整数计数**，模板与本底都给。

顺带把「一个 f 跨三类」的联合拟合也算了：陆/近岸/远洋是已知协变量，纯度不该
依赖候选落在哪儿，所以正确的做法是三类共享一个 f、各配各的模板与本底，
而不是三个 f 取加权平均——后者会让没有分辨力的那两档按 n 摊进来。
"""
import csv
import gzip
import math
import os
from datetime import datetime, timedelta

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
NLON = 12


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def lst_of(row):
    head = row["start"].rstrip("Z").partition(".")[0]
    t = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    return (t.hour + t.minute / 60 + t.second / 3600 + wrap(float(row["lon"])) / 15) % 24


def hist(xs, n=NBIN):
    h = [0] * n
    for x in xs:
        h[min(n - 1, max(0, int((x % 24) / 24 * n)))] += 1
    return h


def lon_bin(lon):
    return min(NLON - 1, int((wrap(lon) + 180) / 360 * NLON))


def normalise(h, floor=1e-6):
    s = sum(h)
    return [max(x / s, floor) for x in h]


def amplitude(p):
    return (max(p) - min(p)) / 2 / (sum(p) / len(p))


def loglike(counts, tgf, bkg, f):
    return sum(n * math.log(f * t + (1 - f) * b)
               for n, t, b in zip(counts, tgf, bkg) if n)


def joint_fit(parts):
    """parts = [(counts, tgf, bkg), ...]，三类共享一个 f。"""
    best_f, best_ll = 0.0, -1e300
    for i in range(1001):
        f = i / 1000.0
        ll = sum(loglike(c, t, b, f) for c, t, b in parts)
        if ll > best_ll:
            best_ll, best_f = ll, f
    zero = sum(loglike(c, t, b, 0.0) for c, t, b in parts)
    lo, hi = best_f, best_f
    for i in range(1001):
        x = i / 1000.0
        if 2 * (best_ll - sum(loglike(c, t, b, x) for c, t, b in parts)) <= 1.0:
            lo, hi = min(lo, x), max(hi, x)
    return best_f, 2 * (best_ll - zero), lo, hi


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
    cat = [r for r in cat if r["LST"] != "NULL"]
    for r in cat:
        r["_k"] = klass(float(r["Lon"]), float(r["Lat"]))
        r["_lst"] = float(r["LST"]) * 24
        r["_t"] = datetime.strptime(r["Date"] + " " + r["UTC"][:15], "%Y-%m-%d %H:%M:%S.%f")

    print("=== 给 HXMT 的三样 ===\n")
    print("【口径】星下点单点判陆：把目录给的 (Lon, Lat) 当一个点，落在 Natural Earth")
    print("  50m physical/land 多边形内算陆，否则按到陆地的距离 < 300 km 算近岸、")
    print("  >= 300 km 算远洋。300 km 缓冲的实现是把陆地多边形整体 buffer(300/111 度)")
    print("  ——在**经纬度平面**上的等角缓冲，不是真正的大圆距离；纬度越高越窄")
    print("  （60 度处东西方向实际只有约 150 km）。含岛屿：Natural Earth 的 land 图层")
    print("  本身就含全部岛屿，unary_union 之后是一个整体。判的是「到最近陆地的距离」，")
    print("  不是到海岸线，两者对内陆点才有区别（内陆点距离为 0，直接判陆）。\n")

    lnd = [r for r in cat if r["_k"] == "land"]
    print(f"【1】陆地模板：目录全期（2008-07..2016-07）LST 非空且判陆的 n = {len(lnd)}")
    h_t = hist([r["_lst"] for r in lnd])
    print(f"  原始整数计数（格心 1.5 4.5 ... 22.5 h）: {h_t}   合计 {sum(h_t)}")
    print(f"  归一化: {' '.join('%.4f' % x for x in normalise(h_t))}   A = {amplitude(normalise(h_t)):.3f}")
    print()

    # 本底：我们自己 2014 池 fa>1 的候选，同一条判陆规则
    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        pool = list(csv.DictReader(fh))
    bkg_rows = [r for r in pool if float(r["fa"]) > 1.0]
    for r in bkg_rows:
        r["_k"] = klass(float(r["lon"]), float(r["lat"]))
        r["_lst"] = lst_of(r)
    bl = [r for r in bkg_rows if r["_k"] == "land"]
    h_b = hist([r["_lst"] for r in bl])
    print(f"【2】同口径本底：我们自己 2014 全年搜索池里 fa > 1 的候选，判陆的 n = {len(bl)}")
    print("  留出档是 **fa > 1**（每天每仪器期望假警报 > 1 个，本底彻底主导）；")
    print("  显著档是 fa <= 1e-5，两者差 5 个量级、不重叠。")
    print(f"  原始整数计数: {h_b}   合计 {sum(h_b)}")
    print(f"  归一化: {' '.join('%.4f' % x for x in normalise(h_b))}   A = {amplitude(normalise(h_b)):.3f}")
    print()

    # 经度重加权：把本底重加权到模板的经度分布上
    wt, wb = hist([float(r["Lon"]) for r in lnd], NLON), hist([float(r["lon"]) for r in bl], NLON)
    wt = [x / sum(wt) for x in wt]
    wb = [x / sum(wb) for x in wb]
    h_bw = [0.0] * NBIN
    for r in bl:
        j = lon_bin(float(r["lon"]))
        w = wt[j] / wb[j] if wb[j] else 0.0
        h_bw[min(NBIN - 1, int((r["_lst"] % 24) / 24 * NBIN))] += w
    scale = sum(h_b) / sum(h_bw)
    h_bw = [x * scale for x in h_bw]
    print("  **经度重加权版**（12 个 30 度经度扇区，把本底的经度分布压到模板的经度分布上，")
    print("  再缩回同一总数，所以是有效计数不是整数；泊松权重仍用上面那组整数）:")
    print("  " + " ".join("%.1f" % x for x in h_bw))
    print(f"  归一化: {' '.join('%.4f' % x for x in normalise(h_bw))}   A = {amplitude(normalise(h_bw)):.3f}")
    print(f"  重加权前后归一值最大改变 "
          f"{max(abs(a - b) for a, b in zip(normalise(h_b), normalise(h_bw))):.4f}")
    print()

    # 时间口径对齐版：只用 2014 的目录条目（与本底同一年）
    l14 = [r for r in lnd if r["Date"][:4] == "2014"]
    h14 = hist([r["_lst"] for r in l14])
    print(f"  时间口径对齐的对照版（只取目录 2014 年的陆地条目，与本底同一年）n = {len(l14)}")
    print(f"  原始整数计数: {h14}")
    print(f"  归一化: {' '.join('%.4f' % x for x in normalise(h14))}   A = {amplitude(normalise(h14)):.3f}")
    print()

    print("【3】成串：目录条目没有做过 is_train 摘除，我们也不该替它做")
    print("  ——is_train 是我们**自己候选池**上的判据（neighbors_10min 超阈），")
    print("  而这些是已发表目录的条目，不是我们搜出来的。改给逐条的目录内")
    print("  +-10 min 邻居数，he 自己按需摘：")
    times = sorted(r["_t"] for r in lnd)
    nb = []
    for r in lnd:
        t = r["_t"]
        lo, hi = t - timedelta(minutes=10), t + timedelta(minutes=10)
        nb.append(sum(1 for x in times if lo <= x <= hi) - 1)
    dist = {}
    for x in nb:
        dist[x] = dist.get(x, 0) + 1
    print("  邻居数分布: " + ", ".join(f"{k}个邻居:{v}条" for k, v in sorted(dist.items())))
    print(f"  有至少一个邻居的: {sum(1 for x in nb if x > 0)}/{len(nb)} = "
          f"{100 * sum(1 for x in nb if x > 0) / len(nb):.1f}%")
    solo = [r["_lst"] for r, x in zip(lnd, nb) if x == 0]
    h_s = hist(solo)
    print(f"  摘掉所有有邻居的之后 n = {len(solo)}，原始整数计数: {h_s}")
    print(f"  归一化: {' '.join('%.4f' % x for x in normalise(h_s))}   A = {amplitude(normalise(h_s)):.3f}")
    print(f"  -> 摘与不摘，归一值最大改变 "
          f"{max(abs(a - b) for a, b in zip(normalise(h_t), normalise(h_s))):.4f}")
    print()

    print("=== 顺带：一个 f 跨三类的联合拟合（三类共享 f，各配各的模板与本底）===")
    src = [r for r in cat if r["Date"][:4] != "2014" and "2010" <= r["Date"][:4] <= "2016"]
    tmpl = {k: normalise(hist([r["_lst"] for r in src if r["_k"] == k])) for k in KINDS}
    bkg = {k: normalise(hist([r["_lst"] for r in bkg_rows if r["_k"] == k])) for k in KINDS}
    for label, name in (("正对照 611（真值 f = 1）", "matched_sig_2014b.csv"),
                        ("多出来的显著 235", "extras_sig_2014b.csv"),
                        ("负对照 600（真值 f ~ 0）", "weak_2014b.csv")):
        rs = list(csv.DictReader(open(os.path.join(HERE, name))))
        by = {k: [] for k in KINDS}
        for r in rs:
            by[klass(float(r["lon"]), float(r["lat"]))].append(lst_of(r))
        parts = [(hist(by[k]), tmpl[k], bkg[k]) for k in KINDS if by[k]]
        f, d, lo, hi = joint_fit(parts)
        print(f"  {label:<24s} n={len(rs):4d}  f={f:.3f}  profile 68% [{lo:.3f},{hi:.3f}]  "
              f"dchi2={d:.1f}（1 自由度）  -> {f * len(rs):.0f} 个")
    print()
    print("  只用陆地那一档（唯一有分辨力的）：")
    for label, name in (("正对照 611", "matched_sig_2014b.csv"),
                        ("多出来的显著 235", "extras_sig_2014b.csv")):
        rs = list(csv.DictReader(open(os.path.join(HERE, name))))
        lst = [lst_of(r) for r in rs if klass(float(r["lon"]), float(r["lat"])) == "land"]
        f, d, lo, hi = joint_fit([(hist(lst), tmpl["land"], bkg["land"])])
        print(f"    {label:<16s} 陆地 n={len(lst):4d}  f={f:.3f} [{lo:.3f},{hi:.3f}]  dchi2={d:.1f}")


if __name__ == "__main__":
    main()
