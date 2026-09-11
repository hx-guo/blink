"""搜索报的 `sf` 是 P(X > count)，不是 P(X >= count)——差一个，而且差得很大。

`snapshot_stepping.rs` 里 `Poisson::new(lambda).sf(numbers[group] as u64)`，
statrs 的 `DiscreteCDF::sf(x)` 是 `1 - cdf(x) = P(X > x)`。观测到 count 个计数
时的 p 值应当是 `P(X >= count)`，用 `P(X > count)` 等于把"恰好观测到这么多"
那一项整个扔掉。而在这种极端尾巴上**恰好那一项就是尾巴本身**：
`P(X > k) / P(X >= k) ≈ lambda / (k + 1)`，lambda ~ 1e-4、k = 9 时是 3.7e-5。

先逐条验明这个关系（拿 JSON 里的 sf/count/mean 直接对），再量"改成 P(X >= count)
之后池子还剩多少"。groupnumber = 1，所以 `mean` 就是那个 lambda，没有分组歧义。
"""
import glob
import json

import numpy as np
from scipy.stats import poisson

YEAR = 3600 * 24 * 365.25
FA_MAX = 20.0
SKIP = {"20260601"}

print(f"{'day':10} {'池':>8} {'sf==P(X>c) 精确':>15} {'改成 P(X>=c) 后 fa<=20':>22} {'留存%':>8}")
tot = keep = 0
for p in sorted(glob.glob("/scratchfs2/gecam/guohx/gecam_a/data/GECAM-A/*/*/*_signals.json")):
    tag = p.split("/")[-1][:8]
    if tag in SKIP:
        continue
    sig = json.load(open(p))
    cnt = np.array([s["count"] for s in sig], float)
    mean = np.array([s["mean"] for s in sig])
    sf = np.array([s["sf"] for s in sig])
    binw = np.array([s["bin_size_best"] for s in sig])
    gt = poisson.sf(cnt, mean)          # P(X > count)，应当 == 报出来的 sf
    ge = poisson.sf(cnt - 1, mean)      # P(X >= count)，正确的 p 值
    ok = np.isclose(gt, sf, rtol=1e-6) | ((gt == 0) & (sf < 1e-300))
    fa2 = ge * YEAR / binw
    k = int((fa2 <= FA_MAX).sum())
    tot += len(sig)
    keep += k
    print(f"{tag:10} {len(sig):8d} {ok.mean()*100:14.2f}% {k:22d} {k/len(sig)*100:7.2f}%")
print(f"{'合计':10} {tot:8d} {'':15} {keep:22d} {keep/tot*100:7.2f}%")
print(f"\n=> 差一个把池子放大了 {tot/max(keep,1):.1f} 倍。")

# 差多少：逐候选的 fa 比
p = "/scratchfs2/gecam/guohx/gecam_a/data/GECAM-A/2024/01/20240111_signals.json"
sig = json.load(open(p))
cnt = np.array([s["count"] for s in sig], float)
mean = np.array([s["mean"] for s in sig])
r = poisson.sf(cnt - 1, mean) / np.maximum(poisson.sf(cnt, mean), 1e-300)
print(f"\n2024-01-11 逐候选 P(X>=c)/P(X>c)：p10 {np.percentile(r,10):.3g}  "
      f"p50 {np.median(r):.3g}  p90 {np.percentile(r,90):.3g}")
print(f"  解析预期 (count+1)/lambda：p50 {np.median((cnt+1)/mean):.3g}")
