"""成串（列车密度）判据在 A 星 25 路上的独立复现：按人群扫尺度，不只报一个尺度。

HXMT 用的是 `neighbors_10min`（±600 s 内全量初始候选池的邻居数，超阈就摘），
GECAM-C 上实测双向失效：绝对阈 34 坐在分布第 8 百分位、摘掉 92.2%，按池中位缩放
又高过分布最大值、一个都摘不掉；而且对检验样本零分辨力，它携带的是轨道位置信息
（|lat| 随邻居数单调）不是真伪信息。第 14 条的处方是**按人群扫尺度**——天格把窗
从 ±600 s 换到 ±30 s 结论完全相反。

A 星这里有一个 C 星没有的便利：**合并判据本身就是一个与成串无关的真伪标签**。
被 τ=150 ns 合并杀掉的候选是同戳簇（粒子穿越），活下来的不是。拿这个标签当真值
去扫成串尺度，两个量在构造上互不相关（一个只看窗内 150 ns 的结构，一个只看
±Δ 秒内有几个邻居）。

任何命中率都并排给偶然期望：均匀池下 ±Δ 内的邻居期望 = 池率 × 2Δ。

用法: python3 ga_train_scan.py <jit_*_cand.csv> <jit_*_report.json> <hours.json>
"""

import csv
import datetime as dt
import json
import sys

import numpy as np
from scipy.special import gammainc

TAUS = [0, 29, 59, 100, 150, 200, 300, 500, 1000]
SCALES = (0.5, 5.0, 30.0, 60.0, 600.0)
EPOCH = (2019, 1, 1)
MIN_NUMBER = 8


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    rep = json.load(open(sys.argv[2]))
    hours = json.load(open(sys.argv[3]))
    hlist = hours["hours"] if isinstance(hours, dict) else hours
    exposure = sum(h["searched_seconds"] for h in hlist)

    t = np.array([met(r["start"]) for r in rows])
    order = np.argsort(t)
    rows = [rows[i] for i in order]
    t = t[order]

    hour = np.array([r["hour"] for r in rows])
    fa = np.array([float(r["fa"]) for r in rows])
    cnt = np.array([int(r["count"]) for r in rows], float)
    mean = np.array([float(r["mean"]) for r in rows])
    bin_us = np.array([float(r["bin_us"]) for r in rows])
    n150 = np.array([int(r["n_t150"]) for r in rows], float)

    Tm = float(np.median(fa / gammainc(cnt, mean)))
    k = TAUS.index(150)
    sh = np.array([1 - rep["hours"][h]["clusters"][k]["merged_frac"] for h in hour])
    fa2 = gammainc(np.maximum(n150, 1), mean * sh) * Tm
    keep = n150 >= MIN_NUMBER            # 合并后仍有 8 个独立事例 = 不是同戳簇
    surv = keep & (fa2 <= 20)

    rate = len(rows) / exposure
    print(f"池 {len(rows)} 个 / 曝光 {exposure:.0f} s = {rate:.3f} /s")
    print(f"合并后 n>=8 的 {int(keep.sum())} 个，其中 fa<=20 的 {int(surv.sum())} 个\n")

    print(f"{'±Δ(s)':>8} {'偶然期望':>9} | {'被合并杀掉(粒子)':>22} | {'合并后 n>=8':>20}")
    print(f"{'':>8} {'':>9} | {'中位':>7} {'p10':>6} {'p90':>6} | {'中位':>7} {'p10':>6} {'p90':>6}")
    for d in SCALES:
        lo = np.searchsorted(t, t - d, "left")
        hi = np.searchsorted(t, t + d, "right")
        nb = (hi - lo - 1).astype(float)          # 不算自己
        exp = rate * 2 * d
        a, b = nb[~keep], nb[keep]
        print(f"{d:8.1f} {exp:9.1f} | {np.median(a):7.1f} {np.percentile(a,10):6.1f} "
              f"{np.percentile(a,90):6.1f} | {np.median(b):7.1f} {np.percentile(b,10):6.1f} "
              f"{np.percentile(b,90):6.1f}")

    print("\n分离度（粒子中位 / 非粒子中位；接近 1 就是没有分辨力）")
    for d in SCALES:
        lo = np.searchsorted(t, t - d, "left")
        hi = np.searchsorted(t, t + d, "right")
        nb = (hi - lo - 1).astype(float)
        ma, mb = np.median(nb[~keep]), np.median(nb[keep])
        print(f"  ±{d:6.1f} s : {ma:8.1f} / {mb:8.1f} = {ma/max(mb,1e-9):6.3f}")

    print("\n候选自己的时间聚簇：相邻候选间隔的分布（s）")
    gap = np.diff(t)
    gap = gap[gap > 0]
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{p:3d} = {np.percentile(gap, p):10.5f}   （均匀池期望中位 "
              f"{np.log(2)/rate:.5f}）")

    print("\n幸存者彼此的时间关系")
    ts = t[surv]
    for i in np.where(surv)[0]:
        print(f"  {rows[i]['start']:26} bin={bin_us[i]:8.2f}us cnt={int(cnt[i]):3d} "
              f"n150={int(n150[i]):3d} fa合={fa2[i]:9.3g}")
    if ts.size > 1:
        print("  两两间隔 (s):", ", ".join(f"{x:.3f}" for x in np.diff(ts)))


if __name__ == "__main__":
    main()
