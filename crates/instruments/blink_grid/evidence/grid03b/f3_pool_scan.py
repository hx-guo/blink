"""全池 p 阈扫描：v10 的 2632 个 GRID-03B 候选上，三套判据各否决多少、误杀期望多少。

输入是 `f3_pool.py` 的产物（逐候选 n / T / f₂ / f₃ / k₃ / λ₃ / Poisson p）。

三套判据并排：
    v10 旧门   f₂ > 0.35（最长同戳串占比）——全量实测砍掉 7797 个原始候选，
               但那 7797 在 v10 就被丢了、不在这张表里；这里算的是**幸存下来的
               2632 个**上旧门还会砍多少（应当为 0，是自洽性检查）。
    v11 现行   f₃ > 0.5
    乙案       P(K₃ ≥ k₃ | λ₃) < p，λ₃ 逐候选按窗内逐探头计数算

**误杀期望的算法**：一个候选在"窗内没有真粒子"的零假设下被否决的概率，就是它
出现足够多偶然三重簇的概率，即 P(K₃ ≥ k_min | λ₃)，其中 k_min 是让 p 越过阈的
最小簇数。求和就是"这道门一共误杀几个候选"，可以直接写进论文当代价。

**代价要按两个池分开报，混起来会得出相反的结论。** 全池 2632 个里绝大多数 fa 远大于
1e-5，砍掉它们**一个都不进目录、代价为零**；真正的代价只发生在"本来会进目录"的候选上。
所以主口径是**显著池**（fa ≤ 1e-5），全池那个数只当上界参考。

**λ₃ 必须含本底粒子项。** 量化偶然（n 个不相干事例撞进同一格）在全池只贡献 Σλ₃ = 0.054，
而**真本底里的穿星粒子**落进窗的期望是 1.82（本底三重簇率实测中位 1.06 个/s × 窗长），
大 34 倍。只算量化项会把 λ₃ 低估一个多数量级、判据偏向误杀。真实现里这一项应当由候选
自己的本底窗现算（搜索已经有 ±0.5 s 的本底窗），不是写死的常数。

用法：
    python3 f3_pool_scan.py f3_pool.csv [--sig-fa 1e-5] [--lightning <index.csv>]
"""

import argparse
import csv
import math

import numpy as np

P_THRESHOLDS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5)
SAFETY = (1.0, 2.0, 3.0)


def pois_ge(k, lam):
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    s = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, 1.0 - s)


def false_reject_prob(lam, p_thresh, sf):
    """零假设（窗内无真粒子）下这个候选被乙案否决的概率。

    找到最小的 k 使 P(K₃ ≥ k | sf·λ₃) < p_thresh，那么误杀概率就是 P(K₃ ≥ k | λ₃)：
    判据用的是放大过的 λ₃，而真实的偶然分布是没放大的那个。
    """
    for k in range(1, 40):
        if pois_ge(k, sf * lam) < p_thresh:
            return pois_ge(k, lam)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_csv")
    ap.add_argument("--sig-fa", type=float, default=1e-5)
    ap.add_argument("--bkg-rate3", type=float, default=0.0,
                    help="本底三重簇率（个/s）。`f3_background.py` 在 38 个候选过境上实测中位 "
                         "1.06、四分位 0.96–1.22。给了就把 λ₃ᵇᵏᵍ = 率 × T 加进 λ₃——真实现里"
                         "这一项应当由候选自己的本底窗现算，不是常数。")
    ap.add_argument("--lightning", help="burst_events/index.csv，取 7 个闪电认证的时刻")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.pool_csv)))
    lit = set()
    if args.lightning:
        for r in csv.DictReader(open(args.lightning)):
            if r["lightning"] == "1":
                lit.add(r["start"][:23])

    n = np.array([int(r["n"]) for r in rows])
    fa = np.array([float(r["fa"]) for r in rows])
    f2 = np.array([float(r["f2"]) for r in rows])
    f3 = np.array([float(r["f3"]) for r in rows])
    k3 = np.array([int(r["k3"]) for r in rows])
    lam_q = np.array([float(r["lambda3"]) for r in rows])
    T_s = np.array([float(r["T_us"]) for r in rows]) * 1e-6
    lam_b = args.bkg_rate3 * T_s
    lam = lam_q + lam_b
    is_lit = np.array([r["start"][:23] in lit for r in rows])
    sig = fa <= args.sig_fa

    print("池 %d 个候选（v10 全部 GRID-03B），其中显著（fa ≤ %.0e）%d 个，闪电认证 %d 个"
          % (len(rows), args.sig_fa, sig.sum(), is_lit.sum()))
    print("计数 n：中位 %d，四分位 %d–%d，最大 %d" % (np.median(n), *np.percentile(n, [25, 75]).astype(int), n.max()))
    print("λ₃：中位 %.2e，四分位 %.2e–%.2e，最大 %.2e，> 1e-3 的 %d 个（%.2f%%）"
          % (np.median(lam), *np.percentile(lam, [25, 75]), lam.max(),
             (lam > 1e-3).sum(), 100 * (lam > 1e-3).mean()))
    print("k₃ ≥ 1 的 %d 个（%.2f%%），k₃ ≥ 2 的 %d 个" % ((k3 >= 1).sum(), 100 * (k3 >= 1).mean(), (k3 >= 2).sum()))
    print("全池 Σλ₃ = %.4f（量化 %.4f + 本底粒子 %.4f，本底率取 %.2f 个/s）"
          % (lam.sum(), lam_q.sum(), lam_b.sum(), args.bkg_rate3))
    print("→ 零假设下全池应有 %.2f 个候选偶然含三重簇，实测 %d 个，%.0f 倍"
          % (lam.sum(), (k3 >= 1).sum(), (k3 >= 1).sum() / max(lam.sum(), 1e-12)))
    print()
    print("== 对照：两个固定阈 ==")
    for name, mask in (("v10 旧门 f₂ > 0.35", f2 > 0.35), ("v11 现行 f₃ > 0.5", f3 > 0.5),
                       ("甲案 f₃ > 0", f3 > 0)):
        fp_sig = lam[sig].sum() if name == "甲案 f₃ > 0" else float("nan")
        fp_all = lam.sum() if name == "甲案 f₃ > 0" else float("nan")
        print("  %-20s 砍 %4d 个（显著的 %2d 个，闪电认证 %d 个），"
              "误杀期望 显著池 %.4f / 全池 %.4f"
              % (name, mask.sum(), (mask & sig).sum(), (mask & is_lit).sum(), fp_sig, fp_all))
    print("  （v10 旧门在幸存候选上砍 0 个是自洽性检查：这 2632 个本来就是过了那道门的。）")
    print()
    print("== 乙案：p 阈扫描 ==")
    print("  %-8s %-6s %6s %6s %6s %10s %10s %8s" %
          ("p 阈", "λ₃×", "砍", "显著", "闪电", "误杀(显著池)", "误杀(全池)", "k₃=1"))
    for sf in SAFETY:
        for th in P_THRESHOLDS:
            rej = np.array([pois_ge(k3[i], sf * lam[i]) < th for i in range(len(rows))])
            fpa = np.array([false_reject_prob(lam[i], th, sf) for i in range(len(rows))])
            print("  %-8.0e %-6g %6d %6d %6d %10.4f %10.4f %8d"
                  % (th, sf, rej.sum(), (rej & sig).sum(), (rej & is_lit).sum(),
                     fpa[sig].sum(), fpa.sum(), (rej & (k3 == 1)).sum()))
    print()
    print("== 被乙案（p < 1e-3，λ₃×1）否决的候选里，显著的那些 ==")
    rej = np.array([pois_ge(k3[i], lam[i]) < 1e-3 for i in range(len(rows))])
    show = np.flatnonzero(rej & sig)
    print("  %-25s %3s %9s %6s %3s %10s %10s %6s" %
          ("start", "n", "T_us", "f₃", "k₃", "λ₃", "p", "fa"))
    for i in show:
        print("  %-25s %3d %9s %6s %3d %10s %10.2e %6.0e"
              % (rows[i]["start"][:19], n[i], rows[i]["T_us"], rows[i]["f3"], k3[i],
                 rows[i]["lambda3"], pois_ge(k3[i], lam[i]), fa[i]))


if __name__ == "__main__":
    main()
