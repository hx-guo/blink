"""暴内的排队份额 q：用 B 角候选自己的**背靠背帧占比**去卡。

## 为什么它是关键路径

折扣因子对 q 极度敏感（`grid-inject` 的包络：q = 0 → 0.023、0.21 → 0.142、1.0 → 0.629，
**跨度 28 倍**），而对 latch 窗 w 几乎不敏感（w 改 2.4 倍只动 7%）。q 在**本底率**下已有
两份独立测量（约 0.19），**缺的是暴内值**——暴内队列会变深，浅缓冲与深缓冲在本底上
分不开。

## 判别量

**背靠背帧占比**：帧间隔 ≤ `120 + 37(m−1) + 2` tick 的那一部分。排队越多、背靠背帧
越多，而且这个量**在暴窗里直接可测**（`burst_queue.py` 已经量过 GRID-04 的 21 个
显著候选：暴窗中位 0.18、全距 0.023–0.293，本底窗 0.008）。

做法：对每个候选，用**它自己的**输出计数与 T90 造合成暴，过实测读出（帧成本
`dead(s) = 120 + 37(s−1)`、latch 窗 w、排队份额 q），扫 q，看哪一个 q 复现它实测的
背靠背占比。**每个候选各自给一个 q，再看这 21 个 q 的分布**——比只对中位数拟合稳。

**限定**：合成暴假定暴内光子均匀铺开。真暴是成团的，成团会让局部占用率更高、
背靠背更多 ⇒ **由均匀假设反解出的 q 偏高**。所以给出的是 q 的**上限**。

用法: python3 queue_fraction.py <burst_queue.csv> <t90_v15.csv>
"""

import csv
import sys

import numpy as np

sys.path.insert(0, ".")
from burst_sim import readout, run_stats                          # noqa: E402

RNG = np.random.default_rng(17)
QGRID = (0.0, 0.05, 0.10, 0.19, 0.30, 0.45, 0.60, 0.80, 1.0)
W_US = 2.91          # GRID-04 实测 latch 窗（grid02 的回归法，2.91 ± 0.20 µs）
NTRIAL = 120


def predict(n_out, dur_s, bkg_cps, q, span=0.06):
    """给定目标输出计数与本征时长，返回背靠背占比中位。入射数用二分调到目标输出。"""
    lo, hi = max(n_out, 4), max(int(n_out * 12), 40)
    for _ in range(12):
        mid = (lo + hi) // 2
        outs = []
        for _ in range(24):
            nb = RNG.poisson(bkg_cps * span)
            t = np.concatenate([RNG.random(nb) * span,
                                span / 2 - dur_s / 2 + RNG.random(mid) * dur_s])
            det = RNG.integers(0, 4, t.size)
            o = np.argsort(t)
            ft, fs = readout(t[o], det[o], W_US * 1e-6, q)
            lo_, hi_ = span / 2 - dur_s / 2 - 0.003, span / 2 + dur_s / 2 + 0.003
            ev = np.repeat(ft, fs)
            outs.append(int(((ev >= lo_) & (ev <= hi_)).sum()))
        if np.median(outs) < n_out:
            lo = mid
        else:
            hi = mid
    n_in = (lo + hi) // 2
    bbs = []
    for _ in range(NTRIAL):
        nb = RNG.poisson(bkg_cps * span)
        t = np.concatenate([RNG.random(nb) * span,
                            span / 2 - dur_s / 2 + RNG.random(n_in) * dur_s])
        det = RNG.integers(0, 4, t.size)
        o = np.argsort(t)
        ft, fs = readout(t[o], det[o], W_US * 1e-6, q)
        lo_, hi_ = span / 2 - dur_s / 2 - 0.003, span / 2 + dur_s / 2 + 0.003
        m = (ft >= lo_) & (ft <= hi_)
        if m.sum() < 3:
            continue
        bb, _, _ = run_stats(ft[m], fs[m])
        bbs.append(bb)
    return (float(np.median(bbs)) if bbs else np.nan), n_in


def main():
    bq = {r["start"][:23]: r for r in csv.DictReader(open(sys.argv[1]))}
    t90 = {r["start"][:23]: r for r in csv.DictReader(open(sys.argv[2]))}
    print("W = %.2f µs；每档 %d 次" % (W_US, NTRIAL))
    print("%-20s %6s %8s %8s %9s %9s" % ("候选", "计数", "T90 µs", "本底c/s", "实测 bb", "**解出 q**"))
    qs = []
    for k, r in sorted(bq.items()):
        t = t90.get(k)
        if t is None or not t.get("t90_us"):
            continue
        try:
            n_out = int(float(t["n_t90"]))
            dur = float(t["t90_us"]) * 1e-6
            bkg = float(t["rate_bkg"])
            obs = float(r["bb_burst"])
        except (ValueError, KeyError):
            continue
        if n_out < 8 or dur <= 0:
            continue
        curve = []
        for q in QGRID:
            p, _ = predict(n_out, dur, bkg, q)
            curve.append(p)
        curve = np.array(curve)
        ok = np.isfinite(curve)
        if ok.sum() < 3:
            continue
        # 单调插值反解
        qq = np.interp(obs, curve[ok], np.array(QGRID)[ok])
        qs.append(qq)
        print("%-20s %6d %8.0f %8.0f %9.3f %9.2f"
              % (k[:19], n_out, dur * 1e6, bkg, obs, qq))
    qs = np.array(qs)
    if qs.size:
        print()
        print("**解出的 q：中位 %.2f，四分位 %.2f–%.2f，全距 %.2f–%.2f（n = %d）**"
              % (np.median(qs), np.percentile(qs, 25), np.percentile(qs, 75),
                 qs.min(), qs.max(), qs.size))
        print("（合成暴按均匀铺开，真暴成团会抬高背靠背 ⇒ 这是 q 的上限）")


if __name__ == "__main__":
    main()
