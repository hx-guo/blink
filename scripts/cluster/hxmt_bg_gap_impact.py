#!/usr/bin/env python3
"""本底窗里的数据缺口：直接量出来之后，目录到底受多大影响。

第 7 条用的是间接判据 r =（搜索的 ±0.5 s 本底率）/（审计的 ±1 s 本底率），
本底平滑时应为 1。**它有一个盲区：两个窗共享同一个缺口时 r 仍然 ≈ 1。**
`bg_max_gap` 是直接量——本底窗内最大的事例间隔，缺口会直接顶上去。

修正：缺口占掉 gap 秒 ⇒ 真实本底率 = n_bg / (1.98 - gap) 而不是 n_bg / 1.98
⇒ lambda 被低估了 1.98/(1.98-gap) 倍。拿 sig_all_v6.csv 的 mean 做底重算 fa。
"""
import csv, sys
import numpy as np
from scipy.stats import poisson

T_BG = 1.98

gap = {}
for r in csv.DictReader(open(sys.argv[1])):
    gap[r['start'][:23]] = (float(r['bg_max_gap']), r['is_train'] == '1',
                            float(r['false_positive_per_year']), r['assoc'] == '1')

sig = list(csv.DictReader(open(sys.argv[2])))
print("gap 表 %d 行，sig 表 %d 行" % (len(gap), len(sig)))

rec = []
miss = 0
for r in sig:
    k = r['start'][:23]
    if k not in gap:
        miss += 1
        continue
    g, tr, fa_g, ass = gap[k]
    cnt = int(float(r['count'])); mean = float(r['mean']); sf = float(r['sf'])
    fa = float(r['false_positive_per_year'])
    rec.append((k, g, tr, cnt, mean, sf, fa, ass))
print("匹配上 %d，对不上 %d" % (len(rec), miss))

g = np.array([x[1] for x in rec]); tr = np.array([x[2] for x in rec])
cnt = np.array([x[3] for x in rec]); mean = np.array([x[4] for x in rec])
sf = np.array([x[5] for x in rec]); fa = np.array([x[6] for x in rec])

scale = T_BG / np.maximum(T_BG - g, 1e-6)
mean_c = mean * scale
sf_c = poisson.sf(cnt, mean_c)
fa_c = fa * np.where(sf > 0, sf_c / np.maximum(sf, 1e-320), scale ** cnt)

clean = ~tr
print("")
print("=== 池级清洁后的显著候选 %d 个 ===" % int(clean.sum()))
for thr in (0.01, 0.05, 0.2, 0.5, 0.98):
    m = clean & (g > thr)
    print("  gap > %.2f s: %4d (%.2f%%)  修正后 fa > 1e-5 的 %d，fa > 1 的 %d"
          % (thr, int(m.sum()), 100 * m.sum() / clean.sum(),
             int((m & (fa_c > 1e-5)).sum()), int((m & (fa_c > 1.0)).sum())))
out = clean & (fa_c > 1e-5)
print("")
print("**修正后跨出 fa <= 1e-5 的共 %d 个（占清洁显著候选的 %.2f%%）**"
      % (int(out.sum()), 100 * out.sum() / clean.sum()))
idx = [i for i in np.argsort(-fa_c) if clean[i]][:15]
print("")
print("  %-24s %-8s %-5s %-9s %-9s %-10s %s" % ("候选", "gap(s)", "count", "mean", "mean 修正", "fa", "fa 修正"))
for i in idx:
    if not clean[i]:
        continue
    print("  %-24s %-8.4f %-5d %-9.4g %-9.4g %-10.2e %.2e%s"
          % (rec[i][0], g[i], cnt[i], mean[i], mean_c[i], fa[i], fa_c[i],
             "  <== 跨出" if fa_c[i] > 1e-5 else ""))
print("")
print("对照第 7 条的间接判据：它当时只找出 29 个 r < 0.8（其中 25 个 is_train），")
print("清洁池里只剩 4 个、判'目录影响 <= 3 行'。直接量给出的是 %d 个。"
      % int((clean & (g > 0.2)).sum()))
