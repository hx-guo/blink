"""v1 与 v2 逐候选对比：改正因子的分布，以及 argmin 换位到底影响多大。

这是全队要的探针数。重跑昂贵的仪器（GECAM-C 上百小时机时）可以先看这个数，
再决定自己要不要重跑，而不是各自猜。

两件事分开量：

1. **改正因子的分布**——同一个候选（按 `(start, count)` 配对）新旧 `fa` 之比。
   理论值是 `1 + pmf(k_g)/sf(k_g)`，逐组算，所以合并表上算不出来；这里直接实测。
2. **最佳格换位的比例**——配不上对的候选。`sf` 参与最佳格的选择，
   所以改了 `sf` 之后最佳格可能挪，候选的 `(start, count)` 跟着变。
   **这正是"新 argmin 落在旧前二名之外"那个问题的经验答案**：
   若换位比例很小，说明按旧最佳格离线重算得到的上界是紧的。

用法: python3 gbm_v1_v2_compare.py <v1_data_dir> <v2_data_dir> [年份 ...]
"""
import glob
import json
import os
import sys

import numpy as np

SIG = 1e-5


def load(root, years):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "Fermi_GBM", "*", "*", "*_signals.json"))):
        day = os.path.basename(path)[:8]
        if years and day[:4] not in years:
            continue
        for s in json.load(open(path)):
            out[(s["start"], s["count"])] = s
    return out


def q(a, ps=(0, 10, 50, 90, 100)):
    return [np.percentile(a, p) for p in ps]


def main():
    v1_dir, v2_dir = sys.argv[1], sys.argv[2]
    years = tuple(sys.argv[3:]) or None
    a, b = load(v1_dir, years), load(v2_dir, years)
    print(f"v1 候选 {len(a)}   v2 候选 {len(b)}   "
          f"v2/v1 = {len(b) / len(a):.4f}\n")

    common = set(a) & set(b)
    only1, only2 = set(a) - set(b), set(b) - set(a)
    print("=== 最佳格换位（sf 参与最佳格选择的直接后果）===")
    print(f"  同 (start, count) 配上对的 {len(common)}")
    print(f"  只在 v1 {len(only1)}（{100 * len(only1) / len(a):.2f}% of v1）")
    print(f"  只在 v2 {len(only2)}（{100 * len(only2) / len(b):.2f}% of v2）")
    print(f"  **只在 v2 的那批是换了键的，不是新增**——fa 逐点只增，新池不可能有新成员。")
    print(f"  所以最佳格换位率的下界 = {100 * len(only2) / len(b):.2f}%（v2 侧）\n")

    if common:
        r = np.array([b[k]["false_positive_per_year"] / a[k]["false_positive_per_year"]
                      for k in common if a[k]["false_positive_per_year"] > 0])
        p = q(r)
        print("=== 改正因子（配上对的候选，新 fa / 旧 fa）===")
        print(f"  最小 {p[0]:.4g}  p10 {p[1]:.4g}  中位 {p[2]:.4g}  "
              f"p90 {p[3]:.4g}  最大 {p[4]:.4g}")
        print(f"  跨度 {np.log10(p[4] / p[0]):.2f} 个数量级")
        print(f"  **比值 < 1 的（不该有，fa 逐点只增）：{int((r < 0.999).sum())}**")
        lo, hi = np.log10(r).std(), np.log10(
            [a[k]["false_positive_per_year"] for k in common
             if a[k]["false_positive_per_year"] > 0]).std()
        print(f"  因子散布 {lo:.3f} dex ÷ fa 自身散布 {hi:.3f} dex = {lo / hi:.4f}")
        print("  （GECAM 立的规矩：这个比值小 ⇒ 秩几乎不动；看的是比值不是因子多大）\n")

    for name, d in (("v1", a), ("v2", b)):
        fa = np.array([s["false_positive_per_year"] for s in d.values()])
        print(f"{name}: 显著（fa ≤ {SIG:g}）{int((fa <= SIG).sum())}，"
              f"fa 范围 {fa.min():.3g} .. {fa.max():.3g}")
    s1 = {k for k in a if a[k]["false_positive_per_year"] <= SIG}
    s2 = {k for k in b if b[k]["false_positive_per_year"] <= SIG}
    print(f"\n=== 显著集 ===")
    print(f"  v1 {len(s1)}  v2 {len(s2)}  交集 {len(s1 & s2)}")
    print(f"  只在 v1 {len(s1 - s2)}；只在 v2 {len(s2 - s1)}"
          f"（后者应当只来自最佳格换键）")


if __name__ == "__main__":
    main()
