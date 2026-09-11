"""GECAM-C 判据链：逐步的池子大小、成串尺度扫描，以及每一步的闪电关联与偶然期望。

**每一步都必须并排给出偶然期望。** 关联数本身没有意义——偶然本底正比于池子大小，
而判据链的作用就是缩小池子，所以"切完之后关联率上升"是自明的，不是证据。
偶然期望取 `blink wwlln` 逐候选给出的 `coincidence_probability` 之和（它已经把
该时刻该位置的雷暴活跃度算进去了），不用全局平均率。

**不用 CPD 当真伪标签。** gecamB 拿 140 个已发表真 TGF 实测真 TGF 自己的 ±10 µs
CPD obs/exp 就是 7.50（命中 72.1%），「CPD 紧符合 ⇒ 带电粒子」这个标签不成立。
CPD 在这里只当**环境量**（`cpd_rate_per_det`，按活时间与路数折），环境率是环境率，
不是这个候选是什么。

判据链里现在只有两件不依赖 CPD 标签的：

* **`f3`** —— 落在 ≥3 重同戳簇里的计数占比。**只在窗 ≥ 10 µs 时用**：更短的窗上
  三重同戳的偶然期望不可忽略（亚微秒窗上能到 0.5），那一档没有分辨力。逐候选的
  偶然期望 `exp_trip` 已经在特征表里，用它筛而不是用一个写死的窗长阈。
* **成串** —— 按人群扫尺度，阈用「泊松期望 + 5σ」由池率自己定，不看幸存者，无循环风险。

用法: gc_chain.py <feat/*.csv glob> <tgfs.json> [输出前缀]
"""

import csv
import datetime as dt
import glob
import json
import math
import sys

import numpy as np

SCALES = (0.5, 5.0, 30.0, 60.0, 600.0)
EPOCH = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)


def met_ns(iso):
    """ISO → 整数纳秒 MET。三个时间量走整数纳秒，不过 f64。"""
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    whole = int(round((stamp - EPOCH).total_seconds())) * 1_000_000_000
    if frac:
        whole += int(round(float("0." + frac) * 1e9))
    return whole


def load_features(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def load_lightning(path):
    """tgfs.json → {起始时刻字符串: (是否关联, 偶然概率)}。"""
    out = {}
    for tgf in json.load(open(path)):
        key = tgf.get("start") or tgf["signal"]["start"]
        out[key[:23]] = (
            bool(tgf.get("associated", False)),
            float(tgf.get("coincidence_probability", 0.0) or 0.0),
        )
    return out


def neighbours(times_ns, half_seconds):
    """每个候选 ±half 内的邻居数（不含自己）。全量池、排序 + 二分。"""
    half = int(half_seconds * 1e9)
    lo = np.searchsorted(times_ns, times_ns - half, "left")
    hi = np.searchsorted(times_ns, times_ns + half, "right")
    return hi - lo - 1


def report(tag, mask, light, keys):
    n = int(mask.sum())
    if n == 0:
        print(f"{tag:44s}  n=0")
        return
    obs = sum(1 for i in np.flatnonzero(mask) if light.get(keys[i], (False, 0.0))[0])
    exp = sum(light.get(keys[i], (False, 0.0))[1] for i in np.flatnonzero(mask))
    # 观测到 obs 个或更多的泊松概率
    if exp > 0:
        p = 1.0 - sum(math.exp(-exp) * exp**k / math.factorial(k) for k in range(obs))
        sigma = (obs - exp) / math.sqrt(exp)
    else:
        p, sigma = float("nan"), float("nan")
    print(f"{tag:44s}  n={n:7d}  关联 {obs:3d}  偶然期望 {exp:7.2f}  "
          f"比 {obs / exp if exp > 0 else float('nan'):5.2f}  {sigma:+5.2f}σ  P(≥obs)={p:.4f}")


def main():
    rows = load_features(sys.argv[1])
    light = load_lightning(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(f"候选 {len(rows)} 个；闪电表 {len(light)} 条")

    keys = [r["start"] for r in rows]
    matched = sum(1 for k in keys if k in light)
    print(f"特征表与闪电表对上的 {matched}/{len(rows)}（{matched/len(rows)*100:.2f}%）")
    if matched < len(rows) * 0.99:
        print("**两张表对不上，先修 key 口径再看下面的数。**")

    times_ns = np.array([met_ns(r["start"]) for r in rows], dtype=np.int64)
    order = np.argsort(times_ns)
    times_ns = times_ns[order]
    rows = [rows[i] for i in order]
    keys = [keys[i] for i in order]

    fa = np.array([float(r["fa"]) for r in rows])
    bin_us = np.array([float(r["bin_us"]) for r in rows])
    f3 = np.array([float(r["f3"]) for r in rows])
    exp_trip = np.array([float(r["exp_trip"]) for r in rows])
    n_trip = np.array([int(r["n_trip"]) for r in rows])
    cpd_rate = np.array([float(r["cpd_rate_per_det"]) if r["cpd_rate_per_det"] else np.nan
                         for r in rows])
    hard = np.array([float(r["hard200"]) for r in rows])

    span_s = (times_ns[-1] - times_ns[0]) / 1e9
    pool_rate = len(rows) / span_s
    print(f"\n池率 {pool_rate:.5f} /s（跨度 {span_s/86400:.2f} 天）")

    print("\n=== 成串尺度扫描（阈 = 泊松期望 + 5σ，由池率定，不看幸存者）===")
    print("尺度      泊松期望   阈    摘掉        留下")
    counts = {}
    for half in SCALES:
        k = neighbours(times_ns, half)
        counts[half] = k
        mu = pool_rate * 2 * half
        threshold = math.ceil(mu + 5 * math.sqrt(mu))
        cut = int((k > threshold).sum())
        print(f"±{half:6.1f} s  {mu:9.2f}  {threshold:4d}  {cut:7d} ({cut/len(rows)*100:5.2f}%)  "
              f"{len(rows)-cut:7d}")

    # f3 只在「三重同戳的偶然期望可以当零」的候选上用
    usable = exp_trip <= 1e-4
    clean_f3 = ~((n_trip > 0) & usable)
    print(f"\n=== f3 ===")
    print(f"偶然期望 exp_trip ≤ 1e-4 的候选 {int(usable.sum())}"
          f"（{usable.mean()*100:.1f}%）；其中 n_trip > 0 的 {int((usable & (n_trip>0)).sum())}")
    print(f"窗 < 10 µs 的候选 {int((bin_us < 10).sum())}（{(bin_us<10).mean()*100:.1f}%）"
          f"——那一档 f3 不能用")

    half = 30.0
    mu = pool_rate * 2 * half
    train_threshold = math.ceil(mu + 5 * math.sqrt(mu))
    not_train = counts[half] <= train_threshold

    print("\n=== 判据链，每一步都带偶然期望 ===")
    all_mask = np.ones(len(rows), bool)
    report("全部候选", all_mask, light, keys)
    report("+ f3 判定（窗够长且 n_trip>0 的摘掉）", clean_f3, light, keys)
    report(f"+ 成串 ±30 s > {train_threshold}", clean_f3 & not_train, light, keys)
    chain = clean_f3 & not_train
    for cut in (1.0, 0.1, 0.01):
        report(f"+ fa ≤ {cut}", chain & (fa <= cut), light, keys)

    print("\n=== 对照：只用单条判据 ===")
    report("只 f3", clean_f3, light, keys)
    report(f"只成串 ±30 s", not_train, light, keys)
    report("只 fa ≤ 1", fa <= 1.0, light, keys)

    print("\n=== 判据链幸存者的性质（不含 CPD 标签）===")
    for tag, mask in (("全部", all_mask), ("判据链", chain)):
        m = mask
        print(f"{tag:8s} n={int(m.sum()):7d}  窗长中位 {np.median(bin_us[m]):8.2f} µs  "
              f"hard200 中位 {np.median(hard[m]):.3f}  "
              f"CPD 环境率中位 {np.nanmedian(cpd_rate[m]):7.1f} c/s/路")


if __name__ == "__main__":
    main()
