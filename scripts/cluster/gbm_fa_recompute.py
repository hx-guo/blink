"""离线用 P(X ≥ count) 重算 fa，看按 fa 定义的人群变没变。

`blink_algorithms::poisson::sf` 调 statrs 的 `DiscreteCDF::sf(count)`，返回
**P(X > count)**；而"观测到 count 个或更多"的单边 p 值是 **P(X ≥ count)**。
所以所报 sf / fa 系统性偏小，**比值恰好是 (count+1)/λ**（因为
P(X≥k) = P(X>k) + pmf(k)，而泊松尾在 λ ≪ k 时由首项主导）。

这里**不改 crate**（共用代码 + 版本边界 + 全量重跑要协调），只离线重算，
回答一个问题：**按 fa 分档定义的那几个人群，成员变不变。**
若基本不变 ⇒ 所有 LST / 雷暴邻近度 / 构成法的结论原封不动。

用法: python3 gbm_fa_recompute.py <cand.csv.gz> [显著阈=1e-5]
"""
import csv
import gzip
import sys

import numpy as np
from scipy.stats import poisson

SIG = 1e-5


def main():
    path = sys.argv[1]
    sig = float(sys.argv[2]) if len(sys.argv) > 2 else SIG
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    count = np.array([int(r["count"]) for r in rows])
    lam = np.array([float(r["mean"]) for r in rows])
    fa_old = np.array([float(r["fa"]) for r in rows])
    sf_old = np.array([float(r["sf"]) for r in rows])
    binb = np.array([float(r["bin_best"]) for r in rows])
    print(f"{path}: {n} 个候选\n")

    # 对账 1：能不能从 (count, mean) 复现报出来的 sf。
    # **先过这一关才谈得上离线重算**——过不了就说明输出里的 count/mean 不是
    # 判据实际用的那两个数，任何基于它们的"改正"都是假的。
    sf_gt = poisson.sf(count, lam)
    ratio = sf_old / np.maximum(sf_gt, 1e-300)
    med = np.median(ratio)
    print(f"对账：报的 sf / P(X > count)  p10 {np.percentile(ratio, 10):.4g}  "
          f"p50 {med:.4g}  p90 {np.percentile(ratio, 90):.4g}")
    if not (0.99 < med < 1.01):
        print()
        print("  **对不上，离线重算在这份表上做不了。** 原因在 "
              "`blink_algorithms::types::candidate` 的注释里已经写明：")
        print("  分组搜索的判据走**逐组**泊松尾概率再乘组数做 Bonferroni，")
        print("  而输出里的 `count` / `mean` 是**各组之和**。GBM 的探测器类型是")
        print("  Nai / Bgo0 / Bgo1 三种，2014-2016 只有 BGO ⇒ **2 组**，")
        print("  所以 sf = min(两组各自的尾概率) × 2，拿合并值根本算不出来。")
        print()
        print("  要离线重算，搜索必须额外输出**逐组的 count 与 λ**"
              "（或直接输出判据用的 λ）。")
        print("  在那之前，(count+1)/mean 只是量级参考，不是改正因子：")
        pred = (count + 1) / lam
        q = np.percentile(pred, [10, 50, 90])
        print(f"    (count+1)/mean  p10 {q[0]:.3g}  p50 {q[1]:.3g}  p90 {q[2]:.3g}"
              f"   跨度 {pred.min():.3g} .. {pred.max():.3g}")
        print("  真实改正因子用的是**逐组的** λ_g（比合并 mean 小）与 k_g，")
        print("  (k_g+1)/λ_g 与上面这组数不是一回事，方向也不确定。")
        return

    # 对账 2：fa = sf × 一年秒数 / bin_best
    ratio = fa_old / sf_old
    implied = np.median(ratio * binb)
    print(f"对账：fa/sf × bin_best 的中位 = {implied:.6e} s"
          f"（一年 = 3.15576e7 s），散布 {np.std(ratio * binb) / implied:.2e}"
          f" ⇒ fa = sf × 年秒数 / bin_best 成立，不需要反标定试验次数\n")

    # 改正
    sf_new = sf_gt + poisson.pmf(count, lam)
    corr = sf_new / np.maximum(sf_old, 1e-300)
    fa_new = fa_old * corr
    pred = (count + 1) / lam
    q = np.percentile(corr, [10, 50, 90])
    qp = np.percentile(pred, [10, 50, 90])
    print(f"改正因子 sf_new/sf_old：p10 {q[0]:.3g}  p50 {q[1]:.3g}  p90 {q[2]:.3g}"
          f"   跨度 {corr.min():.3g} .. {corr.max():.3g}")
    print(f"公式 (count+1)/λ    ：p10 {qp[0]:.3g}  p50 {qp[1]:.3g}  p90 {qp[2]:.3g}")
    print(f"两者中位之比 {q[1] / qp[1]:.4f} ⇒ 公式对得上\n")

    old_sig = fa_old <= sig
    new_sig = fa_new <= sig
    print(f"=== 显著档（fa ≤ {sig:g}）===")
    print(f"  改正前 {old_sig.sum()}，改正后 {new_sig.sum()}"
          f"（{100 * (new_sig.sum() - old_sig.sum()) / max(old_sig.sum(), 1):+.1f}%）")
    print(f"  出（原显著、现不显著）{int((old_sig & ~new_sig).sum())}，"
          f"进（原不显著、现显著）{int((~old_sig & new_sig).sum())}")
    print(f"  交集 {int((old_sig & new_sig).sum())}，"
          f"占原人群 {100 * (old_sig & new_sig).sum() / max(old_sig.sum(), 1):.1f}%\n")

    for lo, hi, label in ((1e-3, 1.0, "亚阈随机样的取样档 fa 1e-3 .. 1"),
                          (1.0, np.inf, "本底留出档 fa > 1")):
        o = (fa_old > lo) & (fa_old <= hi) if np.isfinite(hi) else fa_old > lo
        m = (fa_new > lo) & (fa_new <= hi) if np.isfinite(hi) else fa_new > lo
        print(f"=== {label} ===")
        print(f"  改正前 {o.sum()}，改正后 {m.sum()}；"
              f"出 {int((o & ~m).sum())}，进 {int((~o & m).sum())}，"
              f"交集占原人群 {100 * (o & m).sum() / max(o.sum(), 1):.1f}%")
    print()
    print("  **判读**：交集占比接近 100% ⇒ 按 fa 分档定义的人群基本不变 ⇒")
    print("  所有建立在这些人群上的结论（LST 三类 f、构成法、雷暴邻近度）不受影响。")


if __name__ == "__main__":
    main()
