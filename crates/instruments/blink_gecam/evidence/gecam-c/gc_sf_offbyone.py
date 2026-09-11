"""GECAM-C：`Poisson::sf` 是 `P(X > count)` 而不是 `P(X >= count)`，差多少。

搜索里报的 `sf` 与 `false_positive_per_year` 用的是 `P(X > count)`。正确的
单侧尾概率是 `P(X >= count)`——**观测到的那个计数本身要算进尾巴里**。两者的
比值就是 `fa` 被低估的倍数：

    P(X >= k) / P(X > k) = 1 + pmf(k) / sf(k)

在 `k >> λ` 这一端（我们的候选全在这一端）它约等于 **(k + 1) / λ**，这就是
统筹要的那个量。这里两个都算：解析近似给直觉，精确值给结论。

**不改代码**，只在候选表上离线重算。用法:

    gc_sf_offbyone.py <data 根目录> [fa 阈值=20]
"""

import glob
import json
import sys

import numpy as np
from scipy import stats


def main():
    root = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    count, mean, fa, best, sfv = [], [], [], [], []
    files = sorted(glob.glob(f"{root}/**/*_signals.json", recursive=True))
    for path in files:
        for s in json.load(open(path)):
            count.append(s["count"])
            mean.append(s["mean"])
            fa.append(s["false_positive_per_year"])
            best.append(s["bin_size_best"])
            sfv.append(s["sf"])
    k = np.array(count, float)
    lam = np.array(mean, float)
    fa = np.array(fa, float)
    sf = np.array(sfv, float)
    width = np.array(best, float)
    print(f"{len(files)} 天，候选 {k.size}")

    # 解析近似与精确比值。
    #
    # **精确比值必须在对数空间算**：`P(X > k)` 在我们这批候选上小到 1e-300
    # 以下，直接相除会得到 0/0 = nan，而 nan 会一路传到分位数里，**看着像
    # "算不出来"，其实是把最显著的那批候选整段丢了**。
    approx = (k + 1.0) / lam
    log_sf = stats.poisson.logsf(k, lam)      # log P(X > k)
    log_pmf = stats.poisson.logpmf(k, lam)
    exact = 1.0 + np.exp(log_pmf - log_sf)

    for name, v in (("(count+1)/λ（解析近似）", approx), ("P(X≥k)/P(X>k)（精确）", exact)):
        q = np.percentile(v, [10, 25, 50, 75, 90, 99])
        print(f"{name}：p10 {q[0]:.3g}  p25 {q[1]:.3g}  p50 {q[2]:.3g}  "
              f"p75 {q[3]:.3g}  p90 {q[4]:.3g}  p99 {q[5]:.3g}  "
              f"min {v.min():.3g}  max {v.max():.3g}")
    # 两者对得上吗
    ratio = approx / exact
    print(f"解析/精确：中位 {np.median(ratio):.4f}，"
          f"5–95% {np.percentile(ratio, 5):.4f}–{np.percentile(ratio, 95):.4f}")

    # 跨阈：正确的尾概率只会更大，所以只有出、没有进。
    #
    # **不要用 `fa * 比值` 去算**：最显著的那批候选 `sf` 已经下溢成 0，
    # `0 * inf = nan`，**它们会从统计里静默消失**——而那正是最该保住的一批。
    # 改成从 `fa / sf` 反解出每年的试验数（搜索里它是个常数，这里核对一遍），
    # 再用对数空间算出来的 `P(X >= k)` 重新乘回去。
    # **每候选的试验数不是常数**（它随窗宽走：窗越窄，一小时里能试的格子越多），
    # 所以不能反解出一个全局常数再乘回去——只能逐候选用 `fa × 比值`。
    trials = np.where(sf > 0, fa / np.where(sf > 0, sf, 1.0), np.nan)
    print(f"\n每候选每年试验数（由 fa/sf 反解，随窗宽变，**不是常数**）："
          f"中位 {np.nanmedian(trials):.4g}，"
          f"5–95% {np.nanpercentile(trials, 5):.4g}–{np.nanpercentile(trials, 95):.4g}")
    # `sf` 下溢成 0 的那几个是最显著的一批：正确的尾概率同样远小于任何阈值，
    # 直接当作留下。**不这么写就会 0 × inf = nan，把它们从统计里静默抹掉。**
    underflowed = sf <= 0
    fa_fixed = np.where(underflowed, 0.0, fa * np.where(underflowed, 1.0, exact))
    if underflowed.any():
        print(f"  其中 {int(underflowed.sum())} 个候选的 sf 已下溢成 0"
              f"（比 float64 能表达的还显著），按「留下」处理")
    out = fa_fixed > threshold
    print(f"\n按 fa ≤ {threshold:g} 判：改对后掉出去 {int(out.sum())} / {k.size} "
          f"= {out.sum() / k.size * 100:.2f}%")
    print("  **只有出、没有进**：正确的尾概率一律更大，`fa` 只会上升；"
          "而原来 fa 就 > 阈值的候选根本没进过产物，从候选表上看不见。")
    for cut in (1e-5, 1e-3, 1.0):
        keep_old = fa <= cut
        keep_new = fa_fixed <= cut
        print(f"  fa ≤ {cut:g}：{int(keep_old.sum())} → {int(keep_new.sum())}"
              f"（掉 {int((keep_old & ~keep_new).sum())}）")

    # **"改正因子跨度大 ⇒ 排序一定被搅动"是错的**（A 星实测证伪）。要比的是
    # 改正因子的散布 **÷ 被改量自身的散布**：改正因子跨 1.6 个数量级、而 `fa`
    # 自己跨 108 个数量级时，排序几乎不动。所以三个量一起报。
    finite = fa > 0
    span_fa = np.log10(fa[finite].max()) - np.log10(fa[finite].min())
    span_corr = np.log10(np.nanmax(exact[np.isfinite(exact)])) - np.log10(np.nanmin(exact))
    order_old = np.argsort(np.argsort(fa))
    order_new = np.argsort(np.argsort(fa_fixed))
    rho = np.corrcoef(order_old, order_new)[0, 1]
    print(f"\n排序动没动，三个量一起看：")
    print(f"  改正因子自己的跨度       {span_corr:.2f} 个数量级")
    print(f"  log10(fa) 自己的跨度     {span_fa:.2f} 个数量级")
    print(f"  两套 fa 的秩相关         {rho:.6f}")
    # 固定阈是严格套嵌、只出不进；**固定名额才会换人**，两种口径要分开报
    rank_old = np.argsort(fa)
    rank_new = np.argsort(fa_fixed)
    print("  固定名额下的换人率（阈是套嵌的，只有名额制才换人）：")
    for n in (10, 100, 1000, 10000):
        if n > fa.size:
            break
        a = set(rank_old[:n].tolist())
        b = set(rank_new[:n].tolist())
        print(f"    前 {n:6d} 名换掉 {n - len(a & b):5d} = {(n - len(a & b)) / n * 100:5.2f}%")

    # 窗长分档：λ 随窗长走，而亚微秒窗的 λ 最小，偏得最狠
    print("\n按窗长分档（λ 随窗长走，窗越短 λ 越小、偏得越狠）")
    print("窗长档            n      λ 中位     (k+1)/λ 中位    精确比值中位   掉出 fa≤20 的")
    edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, np.inf]
    labels = ["≤ 1 µs", "1–10 µs", "10–100 µs", "0.1–1 ms", "> 1 ms"]
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        sel = (width > lo) & (width <= hi)
        if sel.sum() < 10:
            continue
        print(f"{label:12s} {int(sel.sum()):8d}  {np.median(lam[sel]):9.3g}  "
              f"{np.median(approx[sel]):13.3g}  {np.median(exact[sel]):13.3g}  "
              f"{out[sel].sum() / sel.sum() * 100:9.2f}%")


if __name__ == "__main__":
    main()
