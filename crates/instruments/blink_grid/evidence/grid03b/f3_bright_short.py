"""v12 判据三步验证第二步：往亮而短的一端注入真暴廓形，测误杀率。

**为什么必须实测**：GBM 那边 `f₃ > 0` 在 n = 100 / W = 30 µs 上误杀 58%——亮短端是这
一类同戳判据的死穴。候选池里最亮的 A 角暴只有 21 个计数，用全量候选池的通过率当参照
会让任何同戳判据看起来无害（GRID-02 全量 f₂ ≤ 0.35 只砍 5 个，同一道门在注入的真暴上
否掉 62.6%）。**参照必须是注入的、亮度档匹配的真暴。**

**廓形不能用均匀抽样。** 真暴的光子是成团的，成团更容易挤进同一个时戳；`grid04_inject2.py`
只把这一条从真廓形换成 [0, T] 均匀，同戳门的单独因子就从 0.339 跳到 0.829（2.4 倍）。
本脚本按 **36 个 03B 真暴自己的到达时刻经验分布**做逆 CDF 抽样再拉伸到目标 W，探头从该
暴自己的探头分布抽；n 大于实测值时这是"沿用它自己的廓形形状"，**不是**新造一个廓形，
这一点在解读时要认。

**判据是自归一的，这正是要测的点。** λ₃ 用的是窗内**实测**的逐探头计数，所以暴越亮
λ₃ 越大——不像 `f₃ > 阈值` 那样亮端必挂。误杀率到底怎么随亮度走，只能实测。

流程（逐次试验）：
  1. 从某个真暴的经验到达分布抽 n 个时刻，拉伸到窗长 W；探头按该暴的探头占比抽；
  2. **施加 03B 逐探头死时间 20 tick**（真读出就是这样，挤掉的事例记下来）；
  3. 量化到 2⁻²² s 的时戳格，数 ≥3 重同戳簇个数 k₃、算 f₃；
  4. λ₃ = 含死时间闭式量化项（用存活下来的逐探头计数）+ 安全系数 × 本底项；
  5. v12 否决 ⇔ P(K₃ ≥ k₃ | λ₃) < 1e-3；对照臂同时算旧门 f₃ > 0.5。

用法：
    python3 f3_bright_short.py --npz inject_all.npz [--trials 2000] [--r3 1.4]
"""

import argparse
import math

import numpy as np

from f3_deadtime_exact import DEADTIME_TICKS, N_DET, Q_US, TRIPLE, lambda3_exact, n_max
from f3_truth_scan import THRESHOLD, clusters_ge, pois_ge

# 03B 本底三重簇率（个/s）：f3_background.py 全体候选的 far 段中位
R3_PER_S_DEFAULT = 1.4


def apply_deadtime(ticks, dets):
    """逐探头施加 20 tick 死时间：距该探头上一个存活事例 < 20 格的丢掉。

    返回 (存活 ticks 已排序, 存活逐探头计数, 被挤掉的个数)。
    """
    order = np.argsort(ticks, kind="stable")
    ticks, dets = ticks[order], dets[order]
    last = np.full(N_DET, -10**9, dtype=np.int64)
    keep = np.zeros(ticks.size, dtype=bool)
    for i in range(ticks.size):
        d = dets[i]
        if ticks[i] - last[d] >= DEADTIME_TICKS:
            keep[i] = True
            last[d] = ticks[i]
    kt = ticks[keep]
    counts = np.bincount(dets[keep], minlength=N_DET).tolist()
    return kt, counts, int((~keep).sum())


def draw(rng, shape_rel, det_p, n, w_s):
    """逆 CDF 抽样：按真暴自己的到达时刻经验分布抽 n 个时刻，拉伸到窗长 w_s。"""
    s = np.sort(shape_rel - shape_rel.min())
    span = s[-1]
    u = rng.random(n)
    # 经验分布的逆：在实测到达时刻之间线性插值（保留成团，不铺平）
    pos = np.interp(u, np.linspace(0.0, 1.0, s.size), s)
    if span > 0:
        pos = pos / span * w_s
    else:
        pos = np.zeros(n)
    dets = rng.choice(N_DET, size=n, p=det_p)
    return pos, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--safety", type=float, default=2.0)
    ap.add_argument("--r3", type=float, default=R3_PER_S_DEFAULT, help="本底三重簇率 个/s")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--truth", help="burst_events/index.csv；给了就把第一步的结论接上第二步")
    args = ap.parse_args()

    # 纯数值数组，不需要 allow_pickle（默认 False 即可，避免反序列化任意对象）
    z = np.load(args.npz)
    nsrc = int(z["n_src"])
    shapes = []
    for i in range(nsrc):
        rel = np.asarray(z["src%d_rel" % i], dtype=float)
        det = np.asarray(z["src%d_det" % i], dtype=int)
        p = np.bincount(det, minlength=N_DET).astype(float)
        p = p / p.sum() if p.sum() else np.full(N_DET, 0.25)
        shapes.append((rel, p, rel.max() - rel.min()))
    spans_us = np.array([s[2] for s in shapes]) * 1e6
    print("真暴廓形 %d 个：实测计数 %d–%d，实测窗长 %.1f–%.1f µs（中位 %.1f）"
          % (nsrc, min(len(s[0]) for s in shapes), max(len(s[0]) for s in shapes),
             spans_us.min(), spans_us.max(), np.median(spans_us)))
    print("本底三重簇率 %.2f 个/s，安全系数 %.1f（只加在本底项上），阈值 %.0e"
          % (args.r3, args.safety, THRESHOLD))
    print("逐探头死时间 %d tick = %.4f µs；tick = %.4f µs"
          % (DEADTIME_TICKS, DEADTIME_TICKS * Q_US, Q_US))
    print()

    rng = np.random.default_rng(args.seed)
    cache = {}
    print("%6s %8s %7s %8s %9s %9s %9s %9s %9s" %
          ("n", "W µs", "n_max", "sat", "存活n", "k₃中位", "f₃中位",
           "v12 误杀", "f₃>0.5 误杀"))
    out = []
    for w_us in (8.0, 15.0, 30.0, 60.0, 120.0, 300.0):
        w_s = w_us * 1e-6
        m = int(round(w_us / Q_US))
        lam_b = args.r3 * w_s
        for n in (8, 12, 20, 30, 50, 100):
            veto12 = veto_f3 = 0
            k3s, f3s, ns = [], [], []
            for _ in range(args.trials):
                rel, p, _ = shapes[rng.integers(nsrc)]
                pos, dets = draw(rng, rel, p, n, w_s)
                ticks = np.rint(pos / (Q_US * 1e-6)).astype(np.int64)
                kt, counts, _ = apply_deadtime(ticks, dets)
                ns.append(kt.size)
                if kt.size == 0:
                    continue
                k3, n_in3 = clusters_ge(kt.tolist(), TRIPLE)
                f3 = n_in3 / kt.size
                k3s.append(k3)
                f3s.append(f3)
                if f3 > 0.5:
                    veto_f3 += 1
                # k₃ = 0 时 P = 1，判据恒不触发，λ₃ 不必算（闭式是 O(m·Σn_d)）
                if k3 == 0:
                    continue
                key = (tuple(counts), m)
                lam_q = cache.get(key)
                if lam_q is None:
                    try:
                        lam_q = lambda3_exact(counts, m)
                    except ValueError:
                        lam_q = float("nan")
                    cache[key] = lam_q
                lam = lam_q + args.safety * lam_b
                if math.isfinite(lam) and pois_ge(k3, lam) < THRESHOLD:
                    veto12 += 1
            nm = n_max(w_us)
            row = (n, w_us, nm, n / nm, float(np.mean(ns)),
                   float(np.median(k3s)) if k3s else 0.0,
                   float(np.median(f3s)) if f3s else 0.0,
                   veto12 / args.trials, veto_f3 / args.trials)
            out.append(row)
            print("%6d %8.1f %7d %8.3f %9.1f %9.1f %9.3f %9.3f %9.3f" % row)
    print()

    a = np.array(out)
    print("=== 亮短端（W ≤ 30 µs 且 n ≥ 30）===")
    msk = (a[:, 1] <= 30.0) & (a[:, 0] >= 30)
    if msk.sum():
        print("   v12 误杀率 %.3f–%.3f（中位 %.3f）；f₃ > 0.5 误杀率 %.3f–%.3f（中位 %.3f）"
              % (a[msk, 7].min(), a[msk, 7].max(), np.median(a[msk, 7]),
                 a[msk, 8].min(), a[msk, 8].max(), np.median(a[msk, 8])))
    print("=== 目标人群档（n = 8–21，这是 A 角候选的实测计数范围）===")
    msk2 = (a[:, 0] >= 8) & (a[:, 0] <= 20)
    print("   v12 误杀率 %.3f–%.3f（中位 %.3f）；f₃ > 0.5 误杀率 %.3f–%.3f（中位 %.3f）"
          % (a[msk2, 7].min(), a[msk2, 7].max(), np.median(a[msk2, 7]),
             a[msk2, 8].min(), a[msk2, 8].max(), np.median(a[msk2, 8])))
    print()
    print("读法：v12 的 λ₃ 用窗内实测计数，暴越亮 λ₃ 越大，所以亮端不必然挂；")
    print("      `f₃ > 0.5` 的阈值与亮度无关，亮端必挂。两条的走向要并排看。")

    if args.truth:
        truth_power(args.truth, a, args.trials)


def truth_power(index_csv, grid_rows, trials):
    """把第一步（真值扫阈）和第二步（注入误杀率）接起来：**真值全过说明了多少。**

    在每个闪电证实候选**自己的 (n, W)** 上读注入误杀率 q_i，则"7 个全过"的概率是
    Π(1 − q_i)。若这个概率并不小，那么"7 个全过"就不是判据安全的证据——同样的结果
    在判据有 q_i 那么高的误杀率时本来就常见。**一个验证的价值等于它对目标错误的
    实测灵敏度。**
    """
    import csv as _csv

    ns = np.unique(grid_rows[:, 0])
    ws = np.unique(grid_rows[:, 1])
    tab = np.full((ns.size, ws.size), np.nan)
    for r in grid_rows:
        tab[np.searchsorted(ns, r[0]), np.searchsorted(ws, r[1])] = r[7]

    def rate_at(n, w):
        """在网格上对 (log n, log W) 双线性插值；出界夹到边缘。"""
        i = np.clip(np.interp(math.log(n), np.log(ns), np.arange(ns.size)), 0, ns.size - 1)
        j = np.clip(np.interp(math.log(w), np.log(ws), np.arange(ws.size)), 0, ws.size - 1)
        i0, j0 = int(i), int(j)
        i1, j1 = min(i0 + 1, ns.size - 1), min(j0 + 1, ws.size - 1)
        fi, fj = i - i0, j - j0
        return ((1 - fi) * (1 - fj) * tab[i0, j0] + fi * (1 - fj) * tab[i1, j0]
                + (1 - fi) * fj * tab[i0, j1] + fi * fj * tab[i1, j1])

    rows = [r for r in _csv.DictReader(open(index_csv)) if r["lightning"] == "1"]
    print()
    print("=== 把第一步和第二步接起来：真值全过说明了多少 ===")
    print("%-22s %4s %9s %12s" % ("闪电证实", "n", "W µs", "它自己的误杀率"))
    keep_p = 1.0
    for r in rows:
        n = int(r["count"])
        w = float(r["bin_size_best_us"])
        q = float(rate_at(n, w))
        keep_p *= 1.0 - q
        print("%-22s %4d %9.2f %12.3f" % (r["start"][:19], n, w, q))
    exp_kill = sum(rate_at(int(r["count"]), float(r["bin_size_best_us"])) for r in rows)
    print()
    print("按各自的 (n, W)，这 %d 个里期望被误杀 **%.2f 个**，实测被误杀 0 个。"
          % (len(rows), exp_kill))
    print("「%d 个全过」出现的概率 = Π(1−qᵢ) = **%.3f**。" % (len(rows), keep_p))
    if keep_p > 0.05:
        print("⇒ 这个概率不小，**「真值全过」因此不能当作判据安全的证据**：即便判据有上面")
        print("  那么高的误杀率，%d 个全过也是常见结果。第一步的灵敏度不够，结论要看第二步。"
              % len(rows))
    else:
        print("⇒ 这个概率很小，「真值全过」确实是判据安全的证据。")
    print("（注入 %d 次/格，误杀率的统计误差约 %.3f）" % (trials, 0.5 / math.sqrt(trials)))


if __name__ == "__main__":
    main()
