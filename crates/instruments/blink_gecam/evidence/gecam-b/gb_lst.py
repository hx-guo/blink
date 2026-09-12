"""GECAM-B 的 LST 腿：前置 0a（模板对比度）与形状预言检验。

**0a 要同时给两个数**：实测的瑞利 p，和"若真分布就是某个已知模板、在这个 N 下
期望的 p 是多少"。SVOM 83 个实测 p = 0.82 看似无结构，而期望 p 就是 0.115 ——
**那是统计量不足，不是没有日变化**。

**形状预言检验比任何拟合都该先做**：峰位对不对得上外部模板，不经 f / A / 本底
模型。SVOM 322 个陆地候选的峰 18.9 h 对上 GBM 陆地模板的 18.9 h，p = 4.5e-12。

**注意这 147 个是触发式亮样本**，完备性高只证明亮端不漏；而且它们**不是闪电
关联选出来的**，所以没有"午夜峰是 VLF 传播相位"那个偏置（那条对 WWLLN 关联
样本才成立）。

用法: python3 gb_lst.py
"""

import csv
import datetime as dt
import os

import numpy as np
from scipy import stats

CATALOG = os.environ.get("GB_CATALOG", "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")

# 全队共用的干净模板（8 格，格心 LST 1.5, 4.5 … 22.5 h，归一化）
GBM_ALL = np.array([0.1331, 0.1385, 0.0867, 0.0533, 0.1002, 0.2026, 0.1649, 0.1208])
GBM_LAND = np.array([0.1089, 0.1018, 0.0406, 0.0330, 0.1067, 0.2801, 0.2031, 0.1257])
GBM_COAST = np.array([0.1516, 0.1638, 0.1125, 0.0648, 0.0990, 0.1510, 0.1375, 0.1198])
HXMT_ALL = np.array([0.1321, 0.1237, 0.0793, 0.0571, 0.1087, 0.1863, 0.1716, 0.1410])
CENTRES = np.arange(8) * 3.0 + 1.5
# 模板是从别人报的 4 位小数抄来的，和不精确等于 1；抽样前必须归一化
for _t in (GBM_ALL, GBM_LAND, GBM_COAST, HXMT_ALL):
    _t /= _t.sum()


def rayleigh(lst, n=None):
    n = lst.size if n is None else n
    th = lst / 24.0 * 2 * np.pi
    c, s = np.cos(th).mean(), np.sin(th).mean()
    r = float(np.hypot(c, s))
    z = n * r * r
    p = float(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n)))
    return r, float((np.arctan2(s, c) / (2 * np.pi) * 24) % 24), z, p


def expected_p(template, n, nsim=4000, seed=7):
    """若真分布就是 `template`，在这个 N 下瑞利 p 的期望（中位）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nsim):
        k = rng.choice(8, size=n, p=template)
        x = CENTRES[k] + rng.uniform(-1.5, 1.5, n)
        out.append(rayleigh(x % 24.0)[3])
    return float(np.median(out))


def main():
    rows = list(csv.DictReader(open(CATALOG)))
    ut = np.array([dt.datetime.strptime(r["UT"][:19], "%Y-%m-%dT%H:%M:%S") for r in rows])
    hours = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in ut])
    lon = np.array([float(r["Longitude_deg"]) for r in rows])
    lat = np.array([float(r["Latitude_deg"]) for r in rows])
    lst = (hours + lon / 15.0) % 24.0
    n = lst.size

    print(f"{n} 个已发表 TGF；|纬度| 中位 {np.median(np.abs(lat)):.1f}°，"
          f"范围 {lat.min():.1f} .. {lat.max():.1f}°")
    r, peak, z, p = rayleigh(lst)
    print(f"瑞利检验：R = {r:.4f}，峰 LST = {peak:.2f} h，Z = {z:.3f}，p = {p:.3g}")
    h8, _ = np.histogram(lst, bins=8, range=(0, 24))
    f8 = h8 / h8.sum()
    print("8 格归一化：" + " ".join(f"{x:.4f}" for x in f8))
    amp = (f8.max() - f8.min()) / f8.mean() / 2
    print(f"幅度 A = (max−min)/2/均值 = {amp:.3f}")
    print()
    print("若真分布就是某个已知模板，n = %d 时期望的瑞利 p（中位）：" % n)
    for name, t in (("GBM 全体", GBM_ALL), ("GBM 陆地", GBM_LAND),
                    ("GBM 近岸", GBM_COAST), ("HXMT 全体", HXMT_ALL)):
        print(f"  {name:9s} 期望 p = {expected_p(t, n):.3g}")
    print()
    print("形状预言检验（不经 f / A / 本底模型）：")
    print(f"  实测峰 {peak:.2f} h  vs  GBM 陆地 18.88 h、GBM 全体 20.18 h、HXMT 全体 16.5 h")
    for name, t in (("GBM 全体", GBM_ALL), ("GBM 陆地", GBM_LAND),
                    ("GBM 近岸", GBM_COAST), ("HXMT 全体", HXMT_ALL)):
        e = t * n
        c2 = float(((h8 - e) ** 2 / np.maximum(e, 1e-9)).sum())
        print(f"  与 {name:9s} 的 χ² = {c2:6.1f} / 7   p = {stats.chi2.sf(c2, 7):.3g}")
    print()
    print("经度与 UT 的边缘分布（LST = UT + 经度/15，两者直接耦合，0c 要用）：")
    hl, _ = np.histogram(lon, bins=8, range=(-180, 180))
    hu, _ = np.histogram(hours, bins=8, range=(0, 24))
    print("  经度 8 格 (−180..180)：" + " ".join(f"{x:4d}" for x in hl))
    print("  UT   8 格 (0..24 h)  ：" + " ".join(f"{x:4d}" for x in hu))
    print()
    print("=== 模板要能把真值拟回来：拆半自拟（模板一半、被拟另一半，真值 f = 1）===")
    rng = np.random.default_rng(11)
    bg = np.full(8, 1 / 8)          # 占位本底：均匀（B 星进动 48.5 天会扫匀 LST）
    rec = []
    for _ in range(400):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        ta = np.histogram(lst[a], bins=8, range=(0, 24))[0].astype(float)
        ta /= ta.sum()
        hb = np.histogram(lst[b], bins=8, range=(0, 24))[0].astype(float)
        # 两成分最小二乘：hb ≈ N_b * (f*ta + (1-f)*bg)
        y = hb / hb.sum() - bg
        x = ta - bg
        rec.append(float((x @ y) / (x @ x)))
    rec = np.array(rec)
    print(f"  400 次随机拆半：回收 f 中位 {np.median(rec):.3f}，"
          f"5–95% {np.percentile(rec, 5):.3f} .. {np.percentile(rec, 95):.3f}")
    print(f"  ⇒ 模板 N = {n // 2} 时的衰减因子实测 = {np.median(rec):.3f}"
          f"（蒙卡估的是 0.58，模板 N = {n}）")
    print("  注意：拆半模板只有 73 个，比真用的 147 个更噪 ⇒ 这个回收率是下界，")
    print("  真正 N = 147 的衰减因子应当更接近 1。要外推得看回收率随模板 N 的走向。")
    for frac in (0.25, 0.5, 0.75):
        r2 = []
        for _ in range(300):
            idx = rng.permutation(n)
            k = int(n * frac)
            a, b = idx[:k], idx[k:]
            ta = np.histogram(lst[a], bins=8, range=(0, 24))[0].astype(float)
            ta /= ta.sum()
            hb = np.histogram(lst[b], bins=8, range=(0, 24))[0].astype(float)
            y = hb / hb.sum() - bg
            x = ta - bg
            r2.append(float((x @ y) / (x @ x)))
        print(f"    模板用 {int(n * frac):3d} 个、被拟 {n - int(n * frac):3d} 个："
              f"回收 f 中位 {np.median(r2):.3f}")


if __name__ == "__main__":
    main()
