"""读出占用度 saturation = count / n_max(W)：候选把 GRID-03B 的读出占满了多少。

非瘫痪型死时间 τ 下，一路探头在跨度 W 的窗里最多给 ⌊W/τ⌋ + 1 个事例，四路合计
    n_max(W) = 4 · (⌊W/τ⌋ + 1)，τ = 4.7684 µs（20 tick，实测 788 天没变过）
saturation = count / n_max(W) ∈ (0, 1]。**= 1 意味着这个候选坐在读出的硬件上限上。**

**这是诊断量，不是判据。** saturation = 1 有两种完全不同的成因，形状上分不开：
  (a) 读出伪信号——本来就只能出这么多事例，「暴发」是上限本身；
  (b) 足够亮的真 TGF——它确实把读出占满了，计数是真的、只是被截断了。
所以它只能用来**标记需要单独看的候选**，不能用来砍。真要用，得配合别的量
（f₃ 的粒子签名、闪电关联、谱硬度）一起判。

窗长取哪个也要说清：`bin_size_best` 是搜索选出的最佳格，本身就是「使显著性最大」的
那个宽度，**天然偏向把计数挤进短窗**，所以按它算的 saturation 是**上界**。同时给按
T90 算的一版当对照。

用法：
    python3 saturation.py --index <burst_events>/index.csv [--pool f3_pool.csv]
        [--t90 t90_v15.csv]
"""

import argparse
import csv
import math

import numpy as np

TAU_US = 20 * (2.0**-22) * 1e6  # 4.76837158203125 µs
N_CHAIN = 4


def n_max(w_us, tau=TAU_US):
    if w_us <= 0:
        return N_CHAIN
    return N_CHAIN * (math.floor(w_us / tau) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--pool", help="f3_pool.py 的全池产物，用来给分布做背景")
    ap.add_argument("--t90", help="t90_v15.csv，用 T90 当窗长的对照")
    args = ap.parse_args()

    t90 = {}
    if args.t90:
        for r in csv.DictReader(open(args.t90)):
            if r["sat"] == "GRID-03B":
                try:
                    t90[r["start"][:23]] = float(r["t90_us"])
                except ValueError:
                    pass

    rows = []
    for r in csv.DictReader(open(args.index)):
        w = float(r["bin_size_best_us"])
        n = int(r["count"])
        nm = n_max(w)
        rec = dict(start=r["start"][:23], lightning=r["lightning"], n=n, W_us=w,
                   n_max=nm, sat=n / nm)
        t9 = t90.get(r["start"][:23])
        if t9 and t9 > 0:
            rec["T90_us"] = t9
            rec["sat_t90"] = float(r["count"]) / n_max(t9)
        rows.append(rec)

    print("τ = %.6f µs（20 tick），n_max(W) = 4·(⌊W/τ⌋+1)" % TAU_US)
    print("38 个显著候选（GRID-03B，v10）")
    print()
    print("%-22s %3s %9s %6s %7s %9s %8s %s" %
          ("start", "n", "W µs", "n_max", "sat", "T90 µs", "sat(T90)", "闪电"))
    for r in sorted(rows, key=lambda x: -x["sat"]):
        print("%-22s %3d %9.2f %6d %7.3f %9s %8s %s" %
              (r["start"][:19], r["n"], r["W_us"], r["n_max"], r["sat"],
               "%.0f" % r["T90_us"] if "T90_us" in r else "-",
               "%.3f" % r["sat_t90"] if "sat_t90" in r else "-",
               "★" if r["lightning"] == "1" else ""))

    sat = np.array([r["sat"] for r in rows])
    lit = np.array([r["lightning"] == "1" for r in rows])
    print()
    print("38 个全体：中位 %.3f，四分位 %.3f–%.3f，最大 %.3f，= 1.000 的 %d 个"
          % (np.median(sat), *np.percentile(sat, [25, 75]), sat.max(), (sat >= 0.999).sum()))
    print("7 个闪电认证：中位 **%.3f**，全距 %.3f–%.3f，= 1.000 的 %d 个"
          % (np.median(sat[lit]), sat[lit].min(), sat[lit].max(), (sat[lit] >= 0.999).sum()))
    print("其余 31 个：中位 %.3f，全距 %.3f–%.3f，= 1.000 的 %d 个"
          % (np.median(sat[~lit]), sat[~lit].min(), sat[~lit].max(), (sat[~lit] >= 0.999).sum()))

    if args.pool:
        pn, pw = [], []
        for r in csv.DictReader(open(args.pool)):
            pn.append(int(r["n"]))
            pw.append(float(r["T_us"]))
        ps = np.array([pn[i] / n_max(pw[i]) for i in range(len(pn))])
        print()
        print("全池 %d 个候选（含不显著的）：中位 %.3f，四分位 %.3f–%.3f，"
              "= 1.000 的 %d 个（%.2f%%），≥ 0.5 的 %d 个（%.1f%%）"
              % (len(ps), np.median(ps), *np.percentile(ps, [25, 75]),
                 (ps >= 0.999).sum(), 100 * (ps >= 0.999).mean(),
                 (ps >= 0.5).sum(), 100 * (ps >= 0.5).mean()))


if __name__ == "__main__":
    main()
