"""f₃ 判据三步法的第 (b) 步：逐候选算三重同戳的偶然期望。

f₃ = 落在任意 ≥3 重同戳簇里的计数占比。这个量要立得住，条件不是"f₃ 大"，而是
**实测 / 偶然有数量级分离**。本脚本把"偶然"定量成两个零模型，逐候选给出期望值。

零模型（都只假设"窗内的 n 个事例彼此无关"，不假设本底率）：

  朴素口径 A —— n 个事例在窗内 m = T/q 个时戳格上独立均匀，格与格之间无约束。
    这是 OPEN-QUESTIONS 第 14 条里那张表用的式子。它允许同一个探头两次落进同一格，
    而 GRID-03B 实测同一探头从不出同戳（1.11e8 个探头内相邻对里严格为 0），所以
    A 高估了偶然。

  逐探头口径 B —— 探头 d 在窗内的 n_d 个事例落在 m 格上、同探头不重格（死时间
    20 tick 使同探头最近的两个事例也差 20 格），四路彼此独立。三重同戳因此必须
    来自 ≥3 个不同探头。

两个口径的期望都用期望的线性性**精确**算，不靠近似：
  单格占据概率 a_d = n_d / m（不放回均匀抽样下逐格边缘分布严格如此）；
  E[落在 ≥3 重簇里的事例数] = m · Σ_{|S|≥3} |S| · Π_{d∈S} a_d · Π_{d∉S} (1 − a_d)。
  λ₃ = m · P(某格重数 ≥ 3) = 窗内偶然三重簇个数的期望。
朴素口径同式，只是把"四路"换成"n 个可区分事例落 m 格"的二项分布。

MC 只用来验证解析式（含死时间最小间隔 20 tick 的影响），不用来出主结果——
λ₃ 在 1e-5 量级，抽样估不动。

用法：
    python3 f3_chance.py <burst_events 目录> [-o 输出.csv] [--mc-check]

burst_events 目录来自 `diag/grid03b/grid03b_burst_export.py`，逐候选一份 CSV
（列 t_us_rel_best_start, det, pi, e_kev, in_best_bin）加一份 index.csv。
"""

import argparse
import csv
import math
import os
from itertools import combinations

import numpy as np

# 四星统一的时戳量化步，2^-22 s
Q_US = (2.0**-22) * 1e6
# GRID-03B 的探头死时间，20 tick（实测 1.11e8 个相邻对，半高恒落在 20–21 tick）
DEADTIME_TICKS = 20
N_DET = 4
TRIPLE = 3


def runs_ge(times, k):
    """落在重数 ≥ k 的同戳簇里的事例数。times 已排序。"""
    n = len(times)
    if n == 0:
        return 0
    total = run = 1
    hit = 0
    for i in range(1, n):
        if times[i] == times[i - 1]:
            run += 1
        else:
            if run >= k:
                hit += run
            run = 1
        total += 1
    if run >= k:
        hit += run
    return hit


def n_clusters_ge(times, k):
    """重数 ≥ k 的同戳簇的个数。times 已排序。"""
    n = len(times)
    if n == 0:
        return 0
    cnt = run = 1
    out = 0
    for i in range(1, n):
        if times[i] == times[i - 1]:
            run += 1
        else:
            if run >= k:
                out += 1
            run = 1
    if run >= k:
        out += 1
    return out


def naive_expect(n, m):
    """口径 A：n 个可区分事例独立均匀落 m 格。

    返回 (E[f₃], λ₃)，λ₃ = 重数 ≥ 3 的格数期望。
    """
    if m <= 0 or n <= 0:
        return float("nan"), float("nan")
    p = 1.0 / m
    ev = 0.0  # Σ_{k≥3} k·C(n,k)p^k(1-p)^{n-k}
    pk = 0.0  # Σ_{k≥3}   C(n,k)p^k(1-p)^{n-k}
    for k in range(TRIPLE, n + 1):
        term = math.comb(n, k) * p**k * (1.0 - p) ** (n - k)
        ev += k * term
        pk += term
    return m * ev / n, m * pk


def perdet_expect(counts, m):
    """口径 B：探头 d 放 n_d 个事例、同探头不重格，四路独立。

    返回 (E[f₃], λ₃)。用期望的线性性逐格求和，精确。
    """
    n = int(sum(counts))
    if m <= 0 or n <= 0:
        return float("nan"), float("nan")
    a = [min(c / m, 1.0) for c in counts]
    ev = 0.0
    pk = 0.0
    dets = range(len(a))
    for size in range(TRIPLE, len(a) + 1):
        for s in combinations(dets, size):
            pr = 1.0
            for d in dets:
                pr *= a[d] if d in s else (1.0 - a[d])
            ev += size * pr
            pk += pr
    return m * ev / n, m * pk


def _f3_rows(x, n):
    """逐行算落在 ≥3 重同戳簇里的事例数。x 每行是排好序的时戳格号。"""
    idx = np.arange(n)
    starts = np.empty(x.shape, dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = x[:, 1:] != x[:, :-1]
    ends = np.empty(x.shape, dtype=bool)
    ends[:, -1] = True
    ends[:, :-1] = starts[:, 1:]
    start_idx = np.maximum.accumulate(np.where(starts, idx, 0), axis=1)
    end_idx = np.minimum.accumulate(np.where(ends, idx, n - 1)[:, ::-1], axis=1)[:, ::-1]
    runlen = end_idx - start_idx + 1
    return (runlen >= TRIPLE).sum(axis=1)


def perdet_mc(counts, m, trials, rng, min_gap=DEADTIME_TICKS, chunk=250_000):
    """口径 B 的 MC 验证。返回 (E[f₃], P(f₃>0), 有效抽样数)。

    同探头内部用拒绝抽样施加"两两间隔 ≥ min_gap 格"（min_gap = 1 即只要求不重格），
    跨探头不做任何约束——三重同戳只能由 ≥3 个不同探头凑出来。
    """
    m = int(round(m))
    n = int(sum(counts))
    nd_list = [int(c) for c in counts if c > 0]
    tot_hit = 0.0
    tot_nz = 0
    tot_ok = 0
    done = 0
    while done < trials:
        k = min(chunk, trials - done)
        done += k
        cols = []
        ok = np.ones(k, dtype=bool)
        for nd in nd_list:
            pos = np.sort(rng.integers(0, m, size=(k, nd)), axis=1)
            if nd >= 2:
                ok &= (np.diff(pos, axis=1) >= min_gap).all(axis=1)
            cols.append(pos)
        x = np.sort(np.concatenate(cols, axis=1), axis=1)[ok]
        if x.shape[0] == 0:
            continue
        hit = _f3_rows(x, n)
        tot_hit += hit.sum()
        tot_nz += int((hit > 0).sum())
        tot_ok += x.shape[0]
    return tot_hit / tot_ok / n, tot_nz / tot_ok, tot_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("burst_dir")
    ap.add_argument("-o", "--out", default="f3_chance.csv")
    ap.add_argument("--mc-check", action="store_true", help="对 λ₃ 最大的几个候选跑 MC 验证解析式")
    ap.add_argument("--mc-trials", type=int, default=2_000_000)
    args = ap.parse_args()

    index = {r["csv"]: r for r in csv.DictReader(open(os.path.join(args.burst_dir, "index.csv")))}
    rows = []
    for name in sorted(index):
        meta = index[name]
        path = os.path.join(args.burst_dir, name)
        if not os.path.exists(path):
            continue
        ev = [r for r in csv.DictReader(open(path)) if r["in_best_bin"] == "1"]
        # 时戳按 tick 取整后比较：CSV 里是相对最佳格起点的 µs，栅格是 q
        ticks = sorted(int(round(float(r["t_us_rel_best_start"]) / Q_US)) for r in ev)
        n = len(ticks)
        count_json = int(meta["count"])
        if n != count_json:
            print("对账不过：%s 窗内 %d 对 JSON count %d" % (meta["start"][:23], n, count_json))
            continue
        counts = [0] * N_DET
        for r in ev:
            counts[int(r["det"])] += 1
        t_us = float(meta["bin_size_best_us"])
        m = t_us / Q_US
        f3_obs = runs_ge(ticks, TRIPLE) / n
        f2_obs = runs_ge(ticks, 2) / n
        k3_obs = n_clusters_ge(ticks, TRIPLE)
        f3_a, lam_a = naive_expect(n, m)
        f3_b, lam_b = perdet_expect(counts, m)
        rows.append(
            dict(
                start=meta["start"][:23],
                lightning=meta["lightning"],
                n=n,
                T_us="%.2f" % t_us,
                m_ticks="%.1f" % m,
                n_det=",".join(str(c) for c in counts),
                f2_obs="%.4f" % f2_obs,
                f3_obs="%.4f" % f3_obs,
                k3_obs=k3_obs,
                f3_chance_naive="%.3e" % f3_a,
                lambda3_naive="%.3e" % lam_a,
                f3_chance_perdet="%.3e" % f3_b,
                lambda3_perdet="%.3e" % lam_b,
                ratio_naive="%.3e" % (f3_obs / f3_a) if f3_obs > 0 else "",
                ratio_perdet="%.3e" % (f3_obs / f3_b) if f3_obs > 0 else "",
            )
        )

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    arr = lambda k: np.array([float(r[k]) for r in rows])
    f3 = arr("f3_obs")
    lam_a = arr("lambda3_naive")
    lam_b = arr("lambda3_perdet")
    ca = arr("f3_chance_naive")
    cb = arr("f3_chance_perdet")
    nz = f3 > 0
    print("候选 %d 个，f₃ > 0 的 %d 个" % (len(rows), nz.sum()))
    print()
    print("E[f₃] 偶然期望：朴素 中位 %.2e 最大 %.2e；逐探头 中位 %.2e 最大 %.2e"
          % (np.median(ca), ca.max(), np.median(cb), cb.max()))
    print("逐探头 / 朴素 的比值：中位 %.3f" % np.median(cb / ca))
    print()
    print("λ₃（窗内偶然三重簇个数期望）合计：朴素 %.4f，逐探头 %.4f" % (lam_a.sum(), lam_b.sum()))
    print("实测三重簇总个数：%d（分布在 %d 个候选上）"
          % (sum(int(r["k3_obs"]) for r in rows), nz.sum()))
    print("→ 偶然可解释的候选个数 ≈ %.4f（逐探头口径），实测 %d 个" % (lam_b.sum(), nz.sum()))
    print()
    if nz.any():
        print("非零者 实测/偶然：朴素 中位 %.1e 最小 %.1e；逐探头 中位 %.1e 最小 %.1e"
              % (np.median(f3[nz] / ca[nz]), (f3[nz] / ca[nz]).min(),
                 np.median(f3[nz] / cb[nz]), (f3[nz] / cb[nz]).min()))
    print()
    print("7 个闪电认证（f₃ 实测 / 偶然期望 / λ₃ 逐探头）：")
    for r in rows:
        if r["lightning"] == "1":
            print("   %s n=%2d T=%8s µs  f₃ 实测 %s  偶然 %s  λ₃ %s"
                  % (r["start"][:19], r["n"], r["T_us"], r["f3_obs"],
                     r["f3_chance_perdet"], r["lambda3_perdet"]))

    if args.mc_check:
        rng = np.random.default_rng(20260911)
        order = np.argsort(-lam_b)[:3]
        print()
        print("MC 验证（%d 次/候选；解析式不含死时间约束，MC 两种都跑）：" % args.mc_trials)
        for i in order:
            r = rows[i]
            counts = [int(x) for x in r["n_det"].split(",")]
            f3_1, p0_1, ok1 = perdet_mc(counts, float(r["m_ticks"]), args.mc_trials, rng, min_gap=1)
            f3_2, p0_2, ok2 = perdet_mc(
                counts, float(r["m_ticks"]), args.mc_trials, rng, min_gap=DEADTIME_TICKS
            )
            print("   %s  解析 E[f₃] %s | MC 只禁重格 %.3e (n=%d) | MC 加 %d tick 死时间 %.3e (n=%d)"
                  % (r["start"][:19], r["f3_chance_perdet"], f3_1, ok1,
                     DEADTIME_TICKS, f3_2, ok2))
            print("        解析 λ₃ %s | MC P(f₃>0) 只禁重格 %.3e | 加死时间 %.3e"
                  % (r["lambda3_perdet"], p0_1, p0_2))


if __name__ == "__main__":
    main()
