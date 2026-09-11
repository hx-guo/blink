"""含死时间约束的 λ₃ 闭式解——不用 MC，也就不必给 λ₃ 加安全系数。

先前用拒绝抽样量"含死时间的偶然期望"，在高占用度的窗上有效抽样只剩几百次，噪声比
要测的效应还大。其实这个量可以**精确算**。

零模型：探头 d 的 n_d 个事例等概率地落在窗内 m 个时戳格上，**同探头两两间隔 ≥ g 格**
（g = 20，实测死时间）；四路彼此独立。

**逐格占据概率的闭式。** 位置 x₁ < … < x_{n} 满足 x_{i+1} − x_i ≥ g，令
y_i = x_i − (i−1)(g−1)，则 y 是 [0, M) 上的严格递增序列，M = m − (n−1)(g−1)。
所以合法组态与 C(M, n) 个子集一一对应。落在格 t 上的组态数是
    Σ_i C(y_i, i−1) · C(M−1−y_i, n−i)，y_i = t − (i−1)(g−1)
于是
    P_d(t) = Σ_i C(y_i, i−1)·C(M−1−y_i, n_d−i) / C(M, n_d)

**λ₃ 由期望的线性性精确给出**（逐格求和，四路独立）：
    λ₃ = Σ_t Σ_{|S|≥3} Π_{d∈S} P_d(t) · Π_{d∉S} (1 − P_d(t))

g = 1 时退回"只禁重格"，再令 P_d(t) = n_d/m 就退回原来的均匀式。

代价是 O(m · Σn_d) 次组合数，m ≤ 4000、Σn_d ≤ 30，在 Rust 里可忽略。

用法：
    python3 f3_deadtime_exact.py            # 扫 (n, W) 网格，报死时间修正的走向
    python3 f3_deadtime_exact.py --index <burst_events>/index.csv   # 逐候选重算
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
_LG = [0.0]


def _lgam(k):
    while len(_LG) <= k:
        _LG.append(_LG[-1] + math.log(len(_LG)))
    return _LG[k]


def _logc(a, b):
    if b < 0 or a < 0 or b > a:
        return None
    return _lgam(a) - _lgam(b) - _lgam(a - b)


def occupancy(n, m, g):
    """P_d(t)，t = 0..m-1。n 个事例落 m 格、两两间隔 ≥ g。"""
    p = np.zeros(m)
    if n <= 0:
        return p
    M = m - (n - 1) * (g - 1)
    if M < n:
        raise ValueError("窗放不下 %d 个间隔 ≥ %d 的事例（m = %d）" % (n, g, m))
    denom = _logc(M, n)
    for t in range(m):
        s = 0.0
        for i in range(1, n + 1):
            y = t - (i - 1) * (g - 1)
            if y < 0 or y > M - 1:
                continue
            a = _logc(y, i - 1)
            b = _logc(M - 1 - y, n - i)
            if a is None or b is None:
                continue
            s += math.exp(a + b - denom)
        p[t] = min(s, 1.0)
    return p


def lambda3_exact(counts, m, g=DEADTIME_TICKS):
    """含死时间约束的 λ₃，逐格精确求和。"""
    m = int(round(m))
    occ = [occupancy(int(c), m, g) for c in counts]
    lam = 0.0
    dets = range(len(counts))
    for size in range(TRIPLE, len(counts) + 1):
        for s in combinations(dets, size):
            term = np.ones(m)
            for d in dets:
                term = term * (occ[d] if d in s else (1.0 - occ[d]))
            lam += term.sum()
    return lam


def lambda3_uniform(counts, m):
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


def n_max(w_us):
    return N_DET * (math.floor(w_us / (DEADTIME_TICKS * Q_US)) + 1)


def grid():
    print("τ = %.4f µs（20 tick），n_max(W) = 4·(⌊W/τ⌋+1)；λ₃ 两版都是精确值，无抽样噪声" %
          (DEADTIME_TICKS * Q_US))
    print()
    print("%5s %9s %7s %6s %13s %13s %8s" %
          ("n", "W µs", "n_max", "sat", "λ₃ 均匀", "λ₃ 含死时间", "比值"))
    rows = []
    for w_us in (8.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1000.0):
        m = int(round(w_us / Q_US))
        for n in (8, 10, 12, 16, 20, 28):
            nm = n_max(w_us)
            if n > nm:
                continue
            counts = [n // N_DET + (1 if i < n % N_DET else 0) for i in range(N_DET)]
            try:
                lam_e = lambda3_exact(counts, m)
            except ValueError:
                continue
            lam_u = lambda3_uniform(counts, m)
            rows.append((n / nm, lam_u, lam_e))
            print("%5d %9.1f %7d %6.3f %13.4e %13.4e %8.2f"
                  % (n, w_us, nm, n / nm, lam_u, lam_e, lam_e / lam_u))
    r = np.array([(s, e / u) for s, u, e in rows])
    print()
    for lo, hi in ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
                   (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)):
        msk = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if msk.sum():
            print("saturation %.2f–%.2f：%2d 格，含死时间 / 均匀 = 中位 %.2f，全距 %.2f–%.2f"
                  % (lo, hi, msk.sum(), np.median(r[msk, 1]), r[msk, 1].min(), r[msk, 1].max()))
    print()
    print("读法：比值 > 1 = 均匀式**低估**偶然（判据偏向误杀，不安全的方向）。")
    print("      既然闭式可算，判据里直接用含死时间的 λ₃，这一项就不必再留安全系数。")


def per_candidate(index_csv):
    rows = list(csv.DictReader(open(index_csv)))
    base = os.path.dirname(index_csv)
    print("%-22s %3s %9s %6s %12s %12s %7s %s" %
          ("start", "n", "W µs", "sat", "λ₃ 均匀", "λ₃ 含死时间", "比值", "闪电"))
    out = []
    for meta in rows:
        p = os.path.join(base, meta["csv"])
        if not os.path.exists(p):
            continue
        ev = [r for r in csv.DictReader(open(p)) if r["in_best_bin"] == "1"]
        counts = [0] * N_DET
        for r in ev:
            counts[int(r["det"])] += 1
        n = sum(counts)
        if n != int(meta["count"]):
            print("对账不过 %s" % meta["start"][:23])
            continue
        w = float(meta["bin_size_best_us"])
        m = int(round(w / Q_US))
        lam_u = lambda3_uniform(counts, m)
        try:
            lam_e = lambda3_exact(counts, m)
        except ValueError:
            lam_e = float("nan")
        sat = n / n_max(w)
        out.append((lam_u, lam_e))
        print("%-22s %3d %9.2f %6.3f %12.4e %12.4e %7.2f %s"
              % (meta["start"][:19], n, w, sat, lam_u, lam_e, lam_e / lam_u,
                 "★" if meta["lightning"] == "1" else ""))
    a = np.array(out)
    print()
    print("38 个合计：均匀 Σλ₃ = %.4f，含死时间 Σλ₃ = %.4f，比 %.2f"
          % (a[:, 0].sum(), a[:, 1].sum(), a[:, 1].sum() / a[:, 0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="burst_events/index.csv；给了就逐候选重算")
    args = ap.parse_args()
    if args.index:
        per_candidate(args.index)
    else:
        grid()


if __name__ == "__main__":
    main()
