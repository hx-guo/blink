#!/usr/bin/env python3
"""用 sig_all_v6.csv 精确算：它带 count / mean / sf 三列，不用猜也不用反解。

第一步仍然是对账：sf 列必须逐行等于 Poisson(mean).sf(count)。
对不上就说明这份文件不是搜索直出的，后面的数一概不能要。
"""
import csv, sys
import numpy as np
from scipy.stats import poisson

YEAR = 3600.0 * 24.0 * 365.25

rows = list(csv.DictReader(open(sys.argv[1])))
cnt = np.array([int(float(r['count'])) for r in rows])
mean = np.array([float(r['mean']) for r in rows])
sf = np.array([float(r['sf']) for r in rows])
fa = np.array([float(r['false_positive_per_year']) for r in rows])
lat = np.array([float(r['lat']) for r in rows])
lon = np.array([float(r['lon']) for r in rows])
train = np.array([r['is_train'] == '1' for r in rows])
day = np.array([r['date'] for r in rows])

print("sig_all_v6.csv: %d 行" % len(rows))
print("")
print("=== 0 对账：sf 列 == Poisson(mean).sf(count)? ===")
calc = poisson.sf(cnt, mean)
good = (sf > 0) & (calc > 0)
rel = np.abs(calc[good] / sf[good] - 1.0)
print("  可比 %d 行，相对偏差中位 %.2e，最大 %.2e，超过 1e-9 的 %d 行"
      % (int(good.sum()), np.median(rel), rel.max(), int((rel > 1e-9).sum())))
if np.median(rel) > 1e-6:
    print("  !! 对不上，后面不用看了")
    sys.exit(1)
print("  对上了 —— mean 就是判据用的 lambda，count 就是判据用的 count。")

bin_best = (sf / np.maximum(fa, 1e-300)) * YEAR
print("  反推 bin_size_best = sf/fa*year：中位 %.4g s，5–95 [%.4g, %.4g]，最大 %.4g"
      % (np.median(bin_best), np.percentile(bin_best, 5), np.percentile(bin_best, 95), bin_best.max()))

sf_true = poisson.sf(cnt - 1, mean)
ratio = sf_true / np.where(sf > 0, sf, np.nan)
approx = (cnt + 1.0) / mean

print("")
print("=== 3 (count+1)/lambda 的分布 ===")
print("  解析 (count+1)/mean：中位 %.4g，5–95 [%.4g, %.4g]，跨 %.2f dex"
      % (np.median(approx), np.percentile(approx, 5), np.percentile(approx, 95),
         np.log10(np.percentile(approx, 95) / np.percentile(approx, 5))))
r = ratio[np.isfinite(ratio)]
print("  实算 P(X>=k)/P(X>k)：中位 %.4g，5–95 [%.4g, %.4g]" % (np.median(r), np.percentile(r, 5), np.percentile(r, 95)))
print("  lambda = mean：中位 %.4g，5–95 [%.4g, %.4g]" % (np.median(mean), np.percentile(mean, 5), np.percentile(mean, 95)))
print("  按 count 分档：")
for a, b in ((0, 10), (10, 20), (20, 40), (40, 100000)):
    m = (cnt >= a) & (cnt < b)
    if m.sum() < 5:
        continue
    print("    count %3d–%-6d N=%-6d mean 中位 %.4g  比值中位 %.4g" % (a, b, int(m.sum()), np.median(mean[m]), np.median(approx[m])))

fa_true = np.where(sf > 0, fa * ratio, fa * approx)
print("")
print("=== 1 改对之后谁跨过 fa <= 1e-5（池级清洁前的全体显著候选）===")
for lab, sel in (("全体（含 is_train）", np.ones(len(rows), bool)), ("摘掉 is_train 之后", ~train)):
    cur = sel & (fa <= 1e-5)
    new = sel & (fa_true <= 1e-5)
    print("  %-20s 现在 fa <= 1e-5 的 %-6d 行；改对后 %-6d 行；**掉出去 %d（%.1f%%）**"
          % (lab, int(cur.sum()), int((cur & new).sum()), int((cur & ~new).sum()),
             100 * (cur & ~new).sum() / max(cur.sum(), 1)))
    m = cur & ~new
    if m.sum():
        print("      掉出去的 fa_true 分布：中位 %.2e，5–95 [%.2e, %.2e]；其中 <= 1 的 %d 行（还能靠闪电关联留下）"
              % (np.median(fa_true[m]), np.percentile(fa_true[m], 5), np.percentile(fa_true[m], 95),
                 int((fa_true[m] <= 1).sum())))

print("")
print("=== fa 分布的形状会不会变（幂律外推那条）===")
for lab, x in (("现在的 fa", fa), ("改对的 fa", fa_true)):
    for thr in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0):
        pass
    cs = [int((x <= t).sum()) for t in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0)]
    print("  %-10s 累计计数 @ fa <= [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1] = %s" % (lab, cs))
