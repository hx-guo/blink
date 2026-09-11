"""两件事：差一个的跨阈进/出，以及 A 星的定量探测限。

(1) 差一个（OPEN-QUESTIONS 第 36 条）。`P(X >= count) > P(X > count)` 恒成立，
    所以改对之后 `fa` **只会变大**——**跨阈只有出、没有进，而且这是结构上必然
    的，不是这批数据的偶然**。这里先逐条验明单向性，再数各阈上出去几个。
    外部真值这一项在 A 星上判不了：已发表目录（Zhao et al. 2023，147 个）
    **只用 GECAM-B、2020-12-10 .. 2022-08-31**，而 A 星第一批事例是
    2022-10-21——**与 A 星零重叠**（第 10 条）。这里用目录自己的日期区间
    对 A 星的曝光再验一遍，确认不是"碰巧没交集"而是区间不相交。

(2) 探测限：统筹要的"把 A 星的暗端探测不到量出来"。做法是把候选存活曲线
    和**真 TGF 的亮度分布**放在同一张图上。已发表目录的 `NetCounts` 就是
    "一台 25 路 GRD 的 GECAM 看到的 TGF 亮度分布"，A 与 B 探头配置相同，
    所以这条分布可以直接搬。对每个计数阈 c：
      候选存活率 R_cand(c)  = 链末端里 count >= c 的个数 / 曝光
      真 TGF 存活率 R_tgf(c) = 6.20e-6 /s × 目录里 NetCounts >= c 的占比
      污染倍数 = R_cand / R_tgf
    污染倍数降到 1 的那个 c，就是 A 星"这条链能到的最暗处"。
"""
import csv
import datetime as dt
import glob
import json

import numpy as np
from scipy.stats import poisson

D = "/scratchfs2/gecam/guohx/gecam_a/"
CAT = "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv"
YEAR = 3600 * 24 * 365.25
TARGET = 6.20e-6
SKIP = {"2026-06-01"}

days, exp = {}, 0.0
for p in sorted(glob.glob(D + "chain_2???????.csv")):
    tag = p.rsplit("_", 1)[-1].split(".")[0]
    day = f"{tag[:4]}-{tag[4:6]}-{tag[6:]}"
    if day in SKIP:
        continue
    rows = list(csv.DictReader(open(p)))
    days[day] = rows
    exp += float(json.load(open(f"{D}data/GECAM-A/{tag[:4]}/{tag[4:6]}/{tag}_hours.json"))["searched_seconds"])

col = lambda k, t: np.concatenate([np.array([t(r[k]) for r in rs]) for rs in days.values()])
cnt = col("count", int)
mean = col("mean", float)
binu = col("bin_us", float)
fa = col("fa", float)
f2 = col("f2", float)
f3 = col("f3", float)
nkeep = col("n_keep", int)
fa_corr = poisson.sf(cnt - 1, mean) * YEAR / (binu * 1e-6)
N = cnt.size
print(f"A 星 {len(days)} 天，曝光 {exp:.0f} s，候选 {N}\n")

print("=" * 70)
print("(1) 差一个的跨阈进/出")
print("=" * 70)
grew = fa_corr >= fa
print(f"改对后 fa 变大或不变的占比 {grew.mean()*100:.2f}%（应当 100%，"
      f"P(X>=k) > P(X>k) 恒成立）")
print(f"  最大的反向偏离 {max(0.0, float((fa - fa_corr).max())):.3e}（浮点噪声）")
print(f"\n{'阈':>8} {'报出口径内':>10} {'改对口径内':>10} {'出':>7} {'进':>5} {'净':>8}")
for t in (20, 1, 1e-2, 1e-5, 1e-7):
    a = fa <= t
    b = fa_corr <= t
    out = int((a & ~b).sum())
    inn = int((~a & b).sum())
    print(f"{t:8.0e} {int(a.sum()):10d} {int(b.sum()):10d} {out:7d} {inn:5d} {int(b.sum())-int(a.sum()):8d}")
print("\n**进的一栏结构上恒为 0**：改对只会让 fa 变大，阈内集合是严格套嵌的。")
print("所以要问的不是进出各多少，是有没有已发表 TGF 掉出阈。")

print("\n外部真值在 A 星上判不了：")
cat = list(csv.DictReader(open(CAT)))
ut = [dt.datetime.fromisoformat(r["UT"]) for r in cat]
print(f"  已发表目录 {len(cat)} 个，UT 区间 {min(ut).date()} .. {max(ut).date()}（只用 GECAM-B）")
print(f"  A 星本批曝光的日期 {sorted(days)}")
print(f"  A 星事例产品起于 2022-10-21，晚于目录末尾 {max(ut).date()} "
      f"{(dt.date(2022,10,21)-max(ut).date()).days} 天 => **区间不相交，交集恒为空**")
print("  => 第 2 项只能在 GECAM-B 上答；A 星这边只能说变了多少，判不了好坏。")

print("\n" + "=" * 70)
print("(2) 探测限：候选存活 vs 真 TGF 亮度分布（目录 NetCounts，A/B 探头配置相同）")
print("=" * 70)
net = np.array([float(r["NetCounts"]) for r in cat])
print(f"目录 NetCounts: 最小 {net.min():.1f}  p5 {np.percentile(net,5):.1f}  "
      f"中位 {np.median(net):.1f}  p95 {np.percentile(net,95):.1f}")
print(f"A 星搜索的 min_number = 8，**低于目录最暗的 {net.min():.1f}** "
      f"=> 灵敏度不是瓶颈，污染才是\n")

final = (nkeep >= 8) & (binu >= 50.0) & (f3 == 0) & (f2 <= 0.3)
print(f"{'计数阈 c':>9} {'链末端幸存':>10} {'候选率 /s':>11} {'目录 >=c 占比':>12} "
      f"{'真 TGF 率 /s':>12} {'污染倍数':>10}")
for c in (8, 10, 12, 15, 18, 21, 25, 30, 40, 50, 64, 80, 100, 130, 185):
    k = int((final & (cnt >= c)).sum())
    frac = float((net >= c).mean())
    r_tgf = TARGET * frac
    r_cand = k / exp
    ratio = r_cand / r_tgf if r_tgf > 0 else float("inf")
    print(f"{c:9d} {k:10d} {r_cand:11.3e} {frac*100:11.1f}% {r_tgf:12.3e} {ratio:10.0f}")

print("\n同一条曲线，链末端再叠上改对口径的 fa <= 1e-2：")
f2sel = final & (fa_corr <= 1e-2)
print(f"{'计数阈 c':>9} {'幸存':>6} {'候选率 /s':>11} {'真 TGF 率 /s':>12} {'污染倍数':>10}")
for c in (8, 15, 21, 30, 50, 64, 100):
    k = int((f2sel & (cnt >= c)).sum())
    frac = float((net >= c).mean())
    r_tgf = TARGET * frac
    ratio = (k / exp) / r_tgf if r_tgf > 0 else float("inf")
    print(f"{c:9d} {k:6d} {k/exp:11.3e} {r_tgf:12.3e} {ratio:10.0f}")
print(f"\n（期望个数口径：曝光 {exp:.0f} s 里按已发表率该有 "
      f"{exp*TARGET:.2f} 个 NetCounts >= {net.min():.1f} 的真 TGF）")

print("\n" + "=" * 70)
print("(3) 改对之后排序被搅动多少——固定阈是套嵌的，固定名额不是")
print("=" * 70)
o1 = np.argsort(fa, kind="stable")
o2 = np.argsort(fa_corr, kind="stable")
print(f"{'取最显著的前 N 个':>18} {'两套口径的重合':>14} {'换掉的比例':>10}")
for n in (10, 100, 1000, 10000, 100000):
    if n > N:
        break
    a, b = set(o1[:n].tolist()), set(o2[:n].tolist())
    ov = len(a & b)
    print(f"{n:18d} {ov:14d} {(1-ov/n)*100:9.1f}%")
rho = np.corrcoef(np.argsort(o1).astype(float), np.argsort(o2).astype(float))[0, 1]
print(f"两套 fa 的秩相关 {rho:.4f}")
print("=> **按固定阈选：只出不进，集合严格套嵌**（改正因子逐点 >= 1）；")
print("   **按固定名额选：会换人**，因为改正因子跨两个数量级、不是常数平移。")
print("   两句话不矛盾，报的时候要说清楚用的是哪一种选法。")
