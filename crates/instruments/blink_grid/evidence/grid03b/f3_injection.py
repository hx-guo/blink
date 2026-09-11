"""亮而短的模拟 TGF 注入：量 f₃ 判据的误杀率。

**为什么必须做这一条。** 现有 7 个闪电认证的 TGF 都不亮不短（计数 8–21、窗长 41–173 µs），
它们的 f₃ 全为 0 只说明"在它们那个密度下本来就不该有三重同戳"（各自的 λ₃ 合计只有
0.0112），**说明不了判据在更亮更短的暴上安不安全**。而更亮更短正是要探的方向：偶然三重
的期望随 n³/T² 涨，固定阈的 f₃ 判据在密度够高的地方会成片误杀。

**但 03B 的死时间自己把这个危险区堵住了大半**，这是本脚本量出来的第一条：四路各自
4.7684 µs（20 tick）的非瘫痪型死时间，使长 T 的窗里**最多只能记下 4·(⌊T/τ⌋+1) 个事例**
——T = 30 µs 时上限是 28 个。往 30 µs 里注 50 个光子，实际留下的中位只有 18 个，λ₃ 中位
2.2×10⁻²，不是按"50 个事例散在 30 µs"算出来的 0.5。**估 λ₃ 时不施加死时间会高估一个多
数量级**，这一条以前算错过。

做法：把 n 个光子按给定轮廓铺进长 T 的窗，探头四路等概率（实测真暴 det_frac 中位 0.36
≈ 均分），时刻量化到 2⁻²² s 栅格，按探头施加 20 tick（4.7684 µs）非瘫痪型死时间，再叠
GRID-03B 的**真实**本底事例流（取自 38 个候选过境的暴外 ±5–50 ms 段），最后对留下的
事例算 f₃、k₃、λ₃ 与 Poisson 检验，统计各判据的误杀率。

两个判据并排：
    甲  f₃ > 0（窗内有任何 ≥3 重同戳簇就否决）
    甲' f₃ > 0.5（v11 现行；对"一次穿越"恒不触发，见 f3_pool.py 的说明）
    乙  P(K₃ ≥ k₃ | λ₃) < p，λ₃ 逐候选由窗内逐探头计数算出

误杀率只在**还够得着候选门**（留下的计数 ≥ min_number = 8）的试验里统计——够不着的
根本不会进目录，不算判据的账。

用法：
    python3 f3_injection.py <burst_events 目录> [-o out.csv] [--trials 20000]
        [--profile uniform|peaked]
"""

import argparse
import csv
import math
import os
from itertools import combinations

import numpy as np

Q_US = (2.0**-22) * 1e6
DEADTIME_TICKS = 20
N_DET = 4
TRIPLE = 3
MIN_NUMBER = 8
BKG_INNER_US = 5000.0  # 暴外多远才算本底
GRID_N = (20, 30, 50)
GRID_T_US = (30.0, 100.0, 300.0)
P_THRESHOLDS = (1e-2, 3e-3, 1e-3, 1e-4, 1e-5)
# λ₃ 的模型安全系数：`f3_structure.py` 实测"保结构"的 λ₃ 与均匀解析式之比，在 k₃ = 0
# 的干净子集上落在 0.60–1.9（保结构尺度越细越高）。均匀式**可能低估**，而低估 λ₃ 会把
# P 压小、判据偏向误杀，是不安全的方向，所以判据里给 λ₃ 乘一个安全系数。扫 1/2/3 看
# 结论稳不稳。
SAFETY = (1.0, 2.0, 3.0)


def lambda3_uniform(counts, m):
    if m <= 0:
        return float("nan")
    a = [min(c / m, 1.0) for c in counts]
    dets = range(len(a))
    lam = 0.0
    for size in range(TRIPLE, len(a) + 1):
        for s in combinations(dets, size):
            pr = 1.0
            for d in dets:
                pr *= a[d] if d in s else (1.0 - a[d])
            lam += pr
    return m * lam


def pois_ge(k, lam):
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    s = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, 1.0 - s)


def dead_time(tick, det, gap=DEADTIME_TICKS):
    """逐探头非瘫痪型死时间：死期只由**被记录下来**的事例起算。tick 已升序。"""
    last = np.full(N_DET, -(10**18), dtype=np.int64)
    keep = np.zeros(tick.size, dtype=bool)
    for i in range(tick.size):
        d = det[i]
        if tick[i] - last[d] >= gap:
            keep[i] = True
            last[d] = tick[i]
    return keep


def clusters(tick):
    if tick.size == 0:
        return 0, 0
    edge = np.flatnonzero(np.diff(tick) != 0)
    st = np.concatenate(([0], edge + 1))
    en = np.concatenate((edge + 1, [tick.size]))
    size = en - st
    big = size >= TRIPLE
    return int(big.sum()), int(size[big].sum())


def load_background(burst_dir):
    """从 38 个候选的 ±50 ms 事例流里取暴外段当本底池。"""
    pool = []
    for name in sorted(os.listdir(burst_dir)):
        if not name.endswith(".csv") or name == "index.csv":
            continue
        t, d = [], []
        for r in csv.DictReader(open(os.path.join(burst_dir, name))):
            x = float(r["t_us_rel_best_start"])
            if abs(x) > BKG_INNER_US:
                t.append(x)
                d.append(int(r["det"]))
        if len(t) < 20:
            continue
        o = np.argsort(t)
        pool.append((np.array(t)[o], np.array(d, dtype=np.int8)[o]))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("burst_dir")
    ap.add_argument("-o", "--out", default="f3_injection.csv")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--profile", choices=["uniform", "peaked"], default="uniform")
    ap.add_argument("--n", type=int, nargs="*", default=list(GRID_N))
    ap.add_argument("--t", type=float, nargs="*", default=list(GRID_T_US))
    args = ap.parse_args()

    rng = np.random.default_rng(20260911)
    pool = load_background(args.burst_dir)
    span = np.median([p[0].max() - p[0].min() for p in pool])
    rate = np.median([p[0].size / ((p[0].max() - p[0].min()) * 1e-6) for p in pool])
    print("本底池 %d 段，跨度中位 %.0f µs，速率中位 %.0f c/s（四路合计，keep 后）"
          % (len(pool), span, rate))
    print("注入轮廓 %s，死时间 %d tick，每格 %d 次" % (args.profile, DEADTIME_TICKS, args.trials))
    print()

    rows = []
    hdr = "%-6s %-8s %8s %8s %8s %8s %8s" % ("n", "T µs", "留计数", "达标率", "甲 f₃>0", "甲' >0.5", "λ₃ 中位")
    hdr += "".join("%10s" % ("乙 p<%.0e" % p) for p in P_THRESHOLDS)
    print(hdr)
    for n_emit in args.n:
        for t_us in args.t:
            m = t_us / Q_US
            t_ticks = int(round(m))
            rej_a = rej_a5 = 0
            rej_b = [[0] * len(P_THRESHOLDS) for _ in SAFETY]
            ok = 0
            n_keep, lam_keep, f3_keep = [], [], []
            for _ in range(args.trials):
                if args.profile == "uniform":
                    u = rng.random(n_emit)
                else:
                    # 前三分之一集中：Beta(1.5, 4) 轮廓，模拟真暴的"上升快、拖尾"
                    u = rng.beta(1.5, 4.0, n_emit)
                tb = np.rint(u * t_ticks).astype(np.int64)
                db = rng.integers(0, N_DET, n_emit).astype(np.int8)
                # 真实本底：随机取一段同长的窗
                seg, segd = pool[rng.integers(0, len(pool))]
                lo = rng.uniform(seg[0], seg[-1] - t_us)
                i0 = np.searchsorted(seg, lo)
                i1 = np.searchsorted(seg, lo + t_us)
                tb = np.concatenate([tb, np.rint((seg[i0:i1] - lo) / Q_US).astype(np.int64)])
                db = np.concatenate([db, segd[i0:i1]])
                o = np.argsort(tb, kind="stable")
                tb, db = tb[o], db[o]
                keep = dead_time(tb, db)
                tk, dk = tb[keep], db[keep]
                n = tk.size
                if n < MIN_NUMBER:
                    continue
                ok += 1
                counts = np.bincount(dk, minlength=N_DET)
                k3, ev3 = clusters(tk)
                f3 = ev3 / n
                lam = lambda3_uniform(counts, m)
                n_keep.append(n)
                lam_keep.append(lam)
                f3_keep.append(f3)
                if f3 > 0:
                    rej_a += 1
                if f3 > 0.5:
                    rej_a5 += 1
                for si, sf in enumerate(SAFETY):
                    p = pois_ge(k3, sf * lam)
                    for j, th in enumerate(P_THRESHOLDS):
                        if p < th:
                            rej_b[si][j] += 1
            if ok == 0:
                continue
            rec = dict(n_emit=n_emit, T_us=t_us, trials=args.trials, reach_min=ok,
                       reach_frac="%.3f" % (ok / args.trials),
                       n_kept_med="%.1f" % np.median(n_keep),
                       lambda3_med="%.3e" % np.median(lam_keep),
                       f3_med="%.3f" % np.median(f3_keep),
                       rej_f3_gt0="%.4f" % (rej_a / ok),
                       rej_f3_gt05="%.4f" % (rej_a5 / ok))
            for si, sf in enumerate(SAFETY):
                for j, th in enumerate(P_THRESHOLDS):
                    rec["rej_p%.0e_sf%g" % (th, sf)] = "%.4f" % (rej_b[si][j] / ok)
            rows.append(rec)
            line = "%-6d %-8.0f %8.1f %8.3f %8.4f %8.4f %8.2e" % (
                n_emit, t_us, np.median(n_keep), ok / args.trials, rej_a / ok, rej_a5 / ok,
                np.median(lam_keep))
            line += "".join("%10.4f" % (rej_b[0][j] / ok) for j in range(len(P_THRESHOLDS)))
            print(line)
            for si, sf in enumerate(SAFETY[1:], start=1):
                print("%-6s %-8s %8s %8s %8s %8s %8s" % ("", "λ₃×%g" % sf, "", "", "", "", "")
                      + "".join("%10.4f" % (rej_b[si][j] / ok) for j in range(len(P_THRESHOLDS))))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print("读法：每一格的数字是**误杀率**——注入的是纯光子暴，没有任何带电粒子，")
    print("      任何一次否决都是误杀。留计数是死时间之后真正留下的事例数中位；")
    print("      达标率是留下计数 ≥ %d（够得着候选门）的试验占比。" % MIN_NUMBER)


if __name__ == "__main__":
    main()
