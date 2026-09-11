"""`fa` 到底怎么从 `count` / `mean` 算出来的——别再反标定一个"试验数 T"。

先前 an5/an7 把 `fa / gammainc(count, mean)` 的中位当作"反标定试验数 T"，
量出 1.906e10。那是错的：JSON 里本来就有 `sf`，而 `fa = sf × 一年秒数 /
bin_size_best`（实测逐条精确成立），T 不是拟出来的常数、它随窗长变。
另外 `sf` 也不等于 `P(X >= count | mean)`——这里把两者的关系量清楚。
"""
import json
import numpy as np
from scipy.stats import poisson

YEAR = 3600 * 24 * 365.25
p = "/scratchfs2/gecam/guohx/gecam_a/data/GECAM-A/2024/01/20240111_signals.json"
sig = json.load(open(p))
rng = np.random.default_rng(0)
pick = rng.choice(len(sig), size=min(4000, len(sig)), replace=False)

fa = np.array([sig[i]["false_positive_per_year"] for i in pick])
sf = np.array([sig[i]["sf"] for i in pick])
cnt = np.array([sig[i]["count"] for i in pick])
mean = np.array([sig[i]["mean"] for i in pick])
binw = np.array([sig[i]["bin_size_best"] for i in pick])

r1 = fa / (sf * YEAR / binw)
print(f"fa / (sf · 一年秒数 / bin)：中位 {np.median(r1):.10f}  "
      f"最大偏离 {np.max(np.abs(r1-1)):.2e}  => fa = sf · YEAR / bin 精确成立")

tail = poisson.sf(cnt - 1, mean)
good = (tail > 0) & (sf > 0)
ratio = sf[good] / tail[good]
print(f"\nsf / P(X>=count | mean)：中位 {np.median(ratio):.3e}  "
      f"四分位 {np.percentile(ratio,25):.2e} .. {np.percentile(ratio,75):.2e}")
print("=> sf 不是拿 mean 当泊松均值的尾概率；mean 与 sf 用的 lambda 不是同一个量。")
# 反解 sf 对应的 λ（固定 count，解 P(X>=count|λ)=sf），看它与 mean 的比
from scipy.optimize import brentq
sub = np.where(good)[0][:300]
lam = []
for k in sub:
    c, s = int(cnt[k]), float(sf[k])
    try:
        lam.append(brentq(lambda L: poisson.sf(c - 1, L) - s, 1e-12, 1e3))
    except ValueError:
        pass
lam = np.array(lam)
mm = mean[sub][: lam.size]
print(f"\n反解的 λ / 报出来的 mean：中位 {np.median(lam/mm):.4f}  "
      f"四分位 {np.percentile(lam/mm,25):.4f} .. {np.percentile(lam/mm,75):.4f}")
