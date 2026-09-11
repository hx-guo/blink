"""GECAM-C：逐路基线里有没有零格 —— 探头级停机 GTI 表达不了，只有逐路基线看得见。

OPEN-QUESTIONS 第 4c 条。GTI 是**全仪器**概念：12 路里掉一路，仪器照样"活着"，
只是有效面积在变，而 `mean` 的分母不会跟着改、`fa` 就偏低。HXMT 那边查明的机制是
整个机箱静默好几秒，逐路基线向量给出的 `111111000000000000` 一字不差对上时序。

我们已经把逐路基线向量存进 `Signal::detectors` 了，查这个几乎零成本——不回事例流。
上一轮只查了一天（2023-06-15，0/3385），**"未观察到"不等于"不存在"**，这一版把
30 天的池子全过一遍。

同时报三个量，缺一不可：
* **有零格的候选数**——绝对数与占比（小样本带 Clopper–Pearson 区间）；
* **逐路的零格次数**——是集中在某一路（真停机）还是散在各路（统计涨落）；
* **基线计数的低分位**——离零多远。基线窗只有 2 s，一路约 460 c/s 时期望 920 个，
  泊松涨落到零的概率是 e^-920，**所以任何一个零格都不可能是涨落**。

用法: gc_detzero.py <signals.json glob>
"""

import collections
import glob
import json
import sys

import numpy as np


def clopper_pearson(k, n, alpha=0.05):
    """二项比例的精确区间。0/n 的上限不是 0。"""
    from scipy.stats import beta  # noqa: PLC0415

    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def main():
    signals = []
    for path in sorted(glob.glob(sys.argv[1])):
        signals.extend(json.load(open(path)))
    print(f"候选 {len(signals)} 个")

    n_with = 0
    n_no_field = 0
    per_detector_zero = collections.Counter()
    baselines = []
    zero_examples = []
    for signal in signals:
        detectors = signal.get("detectors")
        if not detectors:
            n_no_field += 1
            continue
        baseline = detectors.get("baseline") if isinstance(detectors, dict) else None
        if baseline is None:
            n_no_field += 1
            continue
        arr = np.asarray(baseline, float)
        baselines.append(arr)
        zeros = np.flatnonzero(arr == 0)
        if zeros.size:
            n_with += 1
            for i in zeros:
                per_detector_zero[int(i)] += 1
            if len(zero_examples) < 10:
                zero_examples.append((signal["start"], baseline))

    n = len(baselines)
    print(f"带逐路基线向量的候选 {n}；没有这个字段的 {n_no_field}")
    if n == 0:
        return
    lo, hi = clopper_pearson(n_with, n)
    print(f"\n**有零格的候选 {n_with}/{n} = {n_with/n*100:.4f}%"
          f"（95% Clopper–Pearson {lo*100:.4f}–{hi*100:.4f}%）**")

    stack = np.vstack(baselines)
    print(f"基线向量形状 {stack.shape}（候选 × 路）")
    print("\n逐路：零格次数 / 基线计数分位 1/5/50/95")
    for d in range(stack.shape[1]):
        column = stack[:, d]
        q = np.percentile(column, [1, 5, 50, 95])
        print(f"  路 {d+1:2d}  零格 {per_detector_zero.get(d, 0):6d}   "
              + " / ".join(f"{v:8.0f}" for v in q))
    flat = stack.ravel()
    q = np.percentile(flat, [1, 5, 50, 95])
    print("\n全部格子分位 1/5/50/95 = " + " / ".join(f"{v:.0f}" for v in q))
    print(f"最小非零格 {flat[flat > 0].min():.0f}；恰好为零的格子 {int((flat == 0).sum())}"
          f" / {flat.size}")
    print("\n基线窗 2 s、单路约 460 c/s 时期望约 920 个计数，"
          "泊松涨落到零的概率是 e^-920——**任何一个零格都不可能是涨落**。")
    for start, baseline in zero_examples:
        print(f"  {start}  {baseline}")


if __name__ == "__main__":
    main()
