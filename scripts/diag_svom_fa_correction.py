"""把 fa 从 P(X > count) 改成 P(X >= count) 重算，看 SVOM 这批候选被移动多少。

fa = sf x (一年 / bin_size)，sf 现在是 statrs 的 DiscreteCDF::sf(count) = P(X > count)。
正确的单边 p 是 P(X >= count) = sf(count - 1)。改正因子 = P(X>=c)/P(X>c)。

输入 evidence/coverage/features_top899.csv：有 n_core（count）、dur_ms、rate_bkg，
所以 lambda = rate_bkg * dur_ms/1000 可以现算。**注意**：分组搜索的候选其 sf 走的是
逐组尾概率 + Bonferroni，用合并的 count/mean 重算不等价（见 candidate.rs 的注释），
所以下面是**单组近似**，给的是量级与方向，不是逐候选的精确新值。
"""
import csv
import sys

import numpy as np
from scipy.stats import poisson

rows = list(csv.DictReader(open(sys.argv[1])))
c = np.array([float(r["n_core"]) for r in rows])
dur = np.array([float(r["dur_ms"]) for r in rows]) / 1000.0
rate = np.array([float(r["rate_bkg"]) for r in rows])
fa = np.array([float(r["fa"]) for r in rows])
lam = rate * dur
ok = (lam > 0) & (c > 0)
c, lam, fa, dur = c[ok], lam[ok], fa[ok], dur[ok]
print("候选 %d 个（features_top899.csv，v6 的显著池）。" % len(c))
print("count 中位 %.0f（%.0f–%.0f），窗长中位 %.2f ms，本底率中位 %.0f c/s，"
      "λ 中位 %.3f（%.4f–%.3f）"
      % (np.median(c), c.min(), c.max(), 1000 * np.median(dur), np.median(rate),
         np.median(lam), lam.min(), lam.max()))

p_gt = poisson.sf(c, lam)          # P(X > c)
p_ge = poisson.sf(c - 1, lam)      # P(X >= c)
ratio = p_ge / np.maximum(p_gt, 1e-300)
approx = (c + 1) / lam
print("\n改正因子 P(X>=c)/P(X>c)：中位 %.3g，10–90%% %.3g–%.3g，跨度 %.3g–%.3g"
      % (np.median(ratio), *np.percentile(ratio, [10, 90]), ratio.min(), ratio.max()))
print("统筹要的 (count+1)/λ：中位 %.3g，10–90%% %.3g–%.3g，跨度 %.3g–%.3g"
      % (np.median(approx), *np.percentile(approx, [10, 90]), approx.min(), approx.max()))
print("（两者在 λ 很小时相等；实测中位相差 %.1f%%）"
      % (100 * abs(np.median(ratio) / np.median(approx) - 1)))

fa_new = fa * ratio
for thr in (1e-5, 1e-3, 1.0):
    a, b = fa <= thr, fa_new <= thr
    print("\n阈 fa <= %g：原 %d 个 → 改正后 %d 个（出去 %d，进来 %d）"
          % (thr, int(a.sum()), int(b.sum()), int((a & ~b).sum()), int((~a & b).sum())))

# 改正因子随 fa 变不变——变了幂律形状就会变，不只是平移
print("\n改正因子随 fa 的走向（幂律形状会不会变就看这个）：")
edges = [0, 1e-30, 1e-20, 1e-10, 1e-7, 1e-5, 1e-3, 1e-1, 1e9]
for i in range(len(edges) - 1):
    k = (fa > edges[i]) & (fa <= edges[i + 1])
    if k.sum() < 5:
        continue
    print("  fa %8.0e .. %8.0e  N=%4d  改正因子中位 %.3g  log10 中位 %+.2f"
          % (edges[i], edges[i + 1], int(k.sum()), np.median(ratio[k]),
             np.log10(np.median(ratio[k]))))
pos = (fa > 0) & (ratio > 0)
lf, lr = np.log10(fa[pos]), np.log10(ratio[pos])
m = np.isfinite(lf) & np.isfinite(lr)
sl = np.polyfit(lf[m], lr[m], 1)[0]
print("  log10(改正因子) 对 log10(fa) 的斜率 = %+.4f" % sl)
print("  幂律 N(>fa) ∝ fa^b 的指数会被乘上 1/(1+斜率) = %.4f" % (1.0 / (1.0 + sl)))
