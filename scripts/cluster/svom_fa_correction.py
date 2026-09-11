"""fa 单边 p 差一个：用搜索自己的 count / mean / sf / bin_size_best 精确重算。

hxmt 的警告是对的：目录/特征表里的 duration 是 merge() 之后的包络、n_bg 是另一个
本底窗，拿它们重建 lambda 会错得离谱。正确的输入是 tgfs.json 里逐候选的
count / mean / sf / bin_size_best。

改正因子 = P(X>=c)/P(X>c) = 1 + pmf(c; mean) / sf，其中 sf 就是存下来的那一列，
所以不需要再算一次尾概率，也顺便让 sf 列自己参与对账。
"""
import json
import math
import sys


def pmf(c, lam):
    if lam <= 0:
        return 1.0 if c == 0 else 0.0
    return math.exp(-lam + c * math.log(lam) - math.lgamma(c + 1.0))


def sf_gt(c, lam):
    """P(X > c)，从上尾直接累加，用来独立复算 sf 列。"""
    if lam <= 0:
        return 0.0
    t = pmf(c + 1, lam)
    s = 0.0
    k = c + 1
    while t > 0 and (s == 0 or t > s * 1e-17) and k < c + 1 + 50000:
        s += t
        k += 1
        t *= lam / k
    return s


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    i = min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))
    return v[i]


YEAR = 3600.0 * 24.0 * 365.25
recs = json.load(open(sys.argv[1]))
rows = []
for r in recs:
    s = r["signal"]
    tr = r.get("train") or {}
    if tr.get("is_train"):
        continue
    rows.append((int(s["count"]), float(s["mean"]), float(s["sf"]),
                 float(s["bin_size_best"]), float(s["false_positive_per_year"])))
print("候选 %d 个（tgfs.json，已去掉 is_train）。" % len(rows))

# ---- 对账一：sf 列是不是就是 Poisson(mean).sf(count) -------------------------
d = []
for c, lam, sf, b, fa in rows[:4000]:
    if sf > 0:
        d.append(abs(sf_gt(c, lam) / sf - 1.0))
print("对账一 sf 列 vs 自己复算的 P(X>count)：相对偏差中位 %.2e，最大 %.2e（抽 %d 个）"
      % (pct(d, 50), max(d), len(d)))

# ---- 对账二：sf/fa*年 应当等于 bin_size_best，且 <= 1 ms --------------------
d2 = []
bs = []
for c, lam, sf, b, fa in rows:
    if fa > 0 and sf > 0:
        d2.append(abs(sf / fa * YEAR / b - 1.0))
        bs.append(sf / fa * YEAR)
print("对账二 sf/fa*年 vs bin_size_best：相对偏差中位 %.2e，最大 %.2e" % (pct(d2, 50), max(d2)))
print("       反推 bin_size_best：最大 %.6g s（搜索上限 1e-3 s），最小 %.6g s"
      % (max(bs), min(bs)))

# ---- 改正因子 --------------------------------------------------------------
fac, appr, lams, fas, fas_new = [], [], [], [], []
for c, lam, sf, b, fa in rows:
    if sf <= 0 or fa <= 0:
        continue
    f = 1.0 + pmf(c, lam) / sf
    fac.append(f)
    appr.append((c + 1.0) / lam if lam > 0 else float("inf"))
    lams.append(lam)
    fas.append(fa)
    fas_new.append(fa * f)
print("\nλ：中位 %.3f，5–95%% [%.3f, %.3f]" % (pct(lams, 50), pct(lams, 5), pct(lams, 95)))
print("改正因子 P(X>=c)/P(X>c)：中位 %.3f，5–95%% [%.3f, %.3f]，跨度 %.3f–%.3f"
      % (pct(fac, 50), pct(fac, 5), pct(fac, 95), min(fac), max(fac)))
print("近似式 (count+1)/λ：中位 %.3f，5–95%% [%.3f, %.3f]"
      % (pct(appr, 50), pct(appr, 5), pct(appr, 95)))

for thr in (1e-5, 1e-3, 1.0):
    a = sum(1 for x in fas if x <= thr)
    bnew = sum(1 for x in fas_new if x <= thr)
    out = sum(1 for x, y in zip(fas, fas_new) if x <= thr < y)
    inn = sum(1 for x, y in zip(fas, fas_new) if y <= thr < x)
    print("阈 fa <= %g：原 %d → 改正后 %d（出去 %d，进来 %d）" % (thr, a, bnew, out, inn))

# ---- 幂律形状：N(<=fa) ∝ fa^b 的指数变不变 ----------------------------------
def slope(vals, lo, hi):
    v = sorted(x for x in vals if lo <= x <= hi)
    if len(v) < 50:
        return float("nan")
    import math as m
    xs = [m.log10(x) for x in v]
    ys = [m.log10(i + 1.0) for i in range(len(v))]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    return sxy / sxx


for lo, hi in ((1e-5, 1.0), (1e-3, 1.0), (1e-5, 1e-2)):
    print("幂律 dlogN/dlog(fa) 在 [%g, %g]：原 %.4f → 改正后 %.4f（变 %+.2f%%）"
          % (lo, hi, slope(fas, lo, hi), slope(fas_new, lo, hi),
             100 * (slope(fas_new, lo, hi) / slope(fas, lo, hi) - 1)))

# 改正因子随 fa 漂不漂
print("\n改正因子按 fa 分档：")
edges = [0, 1e-30, 1e-20, 1e-10, 1e-5, 1e-3, 1e-1, 1.0, 5.0, 20.0, 1e9]
for i in range(len(edges) - 1):
    k = [f for f, x in zip(fac, fas) if edges[i] < x <= edges[i + 1]]
    if len(k) < 20:
        continue
    print("  fa %8.0e .. %8.0e  N=%6d  改正因子中位 %7.3f" % (edges[i], edges[i + 1], len(k), pct(k, 50)))
