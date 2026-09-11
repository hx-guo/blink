"""v12 判据 `P(K₃ ≥ k₃ | λ₃) < 1e-3` 的三步验证第一步：拿真值扫阈。

**杀掉一个闪电证实的 TGF，这条判据就废。** 所以不能只报"7 个全过"，要**逐候选给出
它自己的余量**：它的 P 值离阈值差多少倍，以及"实测 ÷ 它自己的偶然期望" k₃ / λ₃。

λ₃ 两项（与 v12 定的形式一致，本脚本不改形式只验证）：
    λ₃ = λ₃ᵍ（含死时间约束的闭式量化项，`f3_deadtime_exact.lambda3_exact`）
       + SAFETY × λ₃ᵇ（本底三重簇率 × 窗长，`f3_background.py` 的 far/loc 两版）
安全系数 2 只加在本底那一项上——量化项是精确可算的，不留系数；本底那一项用的是
整段过境的平均率，窗内非均匀才是它的误差来源。

**这条判据有一处结构性的危险，扫阈之前先说清楚**：k₃ 是小整数，λ₃ 是 1e-5–5e-3。
k₃ = 0 时 P = 1，永不否决；k₃ = 1 时 P ≈ λ₃，于是**凡是 λ₃ < 1e-3 的候选，只要出现
一个三重同戳簇就被否**。窗越长、越稀疏，λ₃ 越小，这道门反而越紧——方向与直觉相反。
所以真值里只要有一个窗长几百微秒的暴带一个三重簇，判据当场就废。扫阈要先回答这个。

用法：
    python3 f3_truth_scan.py --index <burst_events>/index.csv [--background f3bkg_all.csv]
                             [--safety 2.0] [-o out.csv]
"""

import argparse
import csv
import math
import os

import numpy as np

from f3_deadtime_exact import N_DET, Q_US, TRIPLE, lambda3_exact, lambda3_uniform, n_max

THRESHOLD = 1e-3


def clusters_ge(ticks, k):
    """(重数 ≥ k 的簇个数, 落在这些簇里的事例数)。ticks 已排序。"""
    if not ticks:
        return 0, 0
    n_cl = n_ev = 0
    run = 1
    for i in range(1, len(ticks)):
        if ticks[i] == ticks[i - 1]:
            run += 1
        else:
            if run >= k:
                n_cl += 1
                n_ev += run
            run = 1
    if run >= k:
        n_cl += 1
        n_ev += run
    return n_cl, n_ev


def pois_ge(k, lam):
    """P(X ≥ k)，X ~ Poisson(lam)。k ≤ 0 时为 1。"""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    s = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, min(1.0, 1.0 - s))


def load(index_csv, bkg_csv, safety):
    base = os.path.dirname(index_csv)
    bkg = {}
    if bkg_csv:
        for r in csv.DictReader(open(bkg_csv)):
            bkg[r["start"][:23]] = r
    rows = []
    for meta in csv.DictReader(open(index_csv)):
        path = os.path.join(base, meta["csv"])
        if not os.path.exists(path):
            continue
        ev = [r for r in csv.DictReader(open(path)) if r["in_best_bin"] == "1"]
        counts = [0] * N_DET
        ticks = []
        for r in ev:
            counts[int(r["det"])] += 1
            ticks.append(int(round(float(r["t_us_rel_best_start"]) / Q_US)))
        ticks.sort()
        n = sum(counts)
        # 先用搜索报的 count 对账，对不上不算数
        if n != int(meta["count"]):
            print("对账不过 %s：窗内 %d 对 count %d" % (meta["start"][:23], n, meta["count"]))
            continue
        w_us = float(meta["bin_size_best_us"])
        m = int(round(w_us / Q_US))
        k3, n_in3 = clusters_ge(ticks, TRIPLE)
        k2, _ = clusters_ge(ticks, 2)
        b = bkg.get(meta["start"][:23])
        lam_b_far = float(b["far_lambda3_in_T"]) if b else 0.0
        lam_b_loc = float(b["loc_lambda3_in_T"]) if b else 0.0
        lam_q = lambda3_exact(counts, m)
        lam_b = max(lam_b_far, lam_b_loc)
        rows.append(dict(
            start=meta["start"][:23],
            lightning=meta["lightning"] == "1",
            n=n, w_us=w_us, m=m, counts=counts,
            sat=n / n_max(w_us),
            k2=k2, k3=k3, f3=n_in3 / n,
            lam_q=lam_q,
            lam_q_uniform=lambda3_uniform(counts, m),
            lam_b_far=lam_b_far, lam_b_loc=lam_b_loc,
            lam=lam_q + safety * lam_b,
        ))
    for r in rows:
        r["p"] = pois_ge(r["k3"], r["lam"])
        r["ratio"] = (r["k3"] / r["lam"]) if r["lam"] > 0 else float("inf")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--background", help="f3_background.py 的合并输出，给了才有本底项")
    ap.add_argument("--safety", type=float, default=2.0, help="只加在本底项上")
    ap.add_argument("-o", "--out", default="f3_truth_scan.csv")
    args = ap.parse_args()

    rows = load(args.index, args.background, args.safety)
    lit = [r for r in rows if r["lightning"]]
    rest = [r for r in rows if not r["lightning"]]
    print("候选 %d 个（闪电证实 %d 个），安全系数 %.1f 只加在本底项上"
          % (len(rows), len(lit), args.safety))
    if not args.background:
        print("！没给 --background，λ₃ 只有量化项，判据被人为收紧，结论偏悲观")
    print()

    print("=== 逐候选：实测 k₃ ÷ 它自己的偶然期望 λ₃ ===")
    print("%-20s %2s %8s %5s %3s %3s %10s %10s %10s %9s %9s %s" %
          ("start", "n", "W µs", "sat", "k₂", "k₃", "λ₃ᵍ量化", "λ₃ᵇ本底", "λ₃ 合计",
           "k₃/λ₃", "P", "判"))
    for r in sorted(rows, key=lambda r: r["p"]):
        kill = r["p"] < THRESHOLD
        print("%-20s %2d %8.2f %5.3f %3d %3d %10.3e %10.3e %10.3e %9.2e %9.2e %s%s" %
              (r["start"][:19], r["n"], r["w_us"], r["sat"], r["k2"], r["k3"],
               r["lam_q"], args.safety * max(r["lam_b_far"], r["lam_b_loc"]), r["lam"],
               r["ratio"] if r["k3"] else 0.0, r["p"],
               "否决" if kill else "放行", " ★闪电" if r["lightning"] else ""))
    print()

    k_lit = [r for r in lit if r["p"] < THRESHOLD]
    print("=== 第一步结论：真值上杀了几个 ===")
    print("闪电证实 %d 个，在阈 %.0e 上被否决 %d 个" % (len(lit), THRESHOLD, len(k_lit)))
    for r in k_lit:
        print("   杀掉 %s  k₃ = %d, λ₃ = %.3e, P = %.3e（差阈值 %.1f 倍）"
              % (r["start"][:19], r["k3"], r["lam"], r["p"], THRESHOLD / r["p"]))
    surv = [r for r in lit if r["p"] >= THRESHOLD]
    if surv:
        worst = min(surv, key=lambda r: r["p"] / THRESHOLD)
        print("幸存者里余量最小的：%s  P = %.3e，是阈值的 %.2f 倍（k₃ = %d）"
              % (worst["start"][:19], worst["p"], worst["p"] / THRESHOLD, worst["k3"]))
    print()

    print("=== 扫阈：阈值放到哪里才不杀真值 ===")
    print("%10s %10s %10s %s" % ("阈值", "杀真值", "否决其余", "备注"))
    for thr in (1e-1, 1e-2, 5e-3, 2e-3, 1e-3, 1e-4, 1e-5, 1e-6, 0.0):
        a = sum(1 for r in lit if r["p"] < thr)
        b = sum(1 for r in rest if r["p"] < thr)
        note = "判据关掉" if thr == 0.0 else ("v12" if thr == THRESHOLD else "")
        print("%10.0e %10d %10d %s" % (thr, a, b, note))
    print()
    ps_lit = sorted(r["p"] for r in lit)
    print("真值 P 的最小值 %.3e ⇒ **阈值必须 ≤ %.3e 才不杀真值**（v12 取 %.0e）"
          % (ps_lit[0], ps_lit[0], THRESHOLD))
    print("真值 P 分布：%s" % ", ".join("%.2e" % p for p in ps_lit))
    print()

    print("=== k₃ = 1 的候选：这条判据的真实形状 ===")
    one = [r for r in rows if r["k3"] == 1]
    print("k₃ = 1 的有 %d 个；k₃ = 1 时 P ≈ λ₃，所以 λ₃ < %.0e 的候选一个簇就被否。"
          % (len(one), THRESHOLD))
    lam_lt = [r for r in one if r["lam"] < THRESHOLD]
    print("其中 λ₃ < 阈值的 %d 个（会被否），含闪电证实 %d 个。"
          % (len(lam_lt), sum(1 for r in lam_lt if r["lightning"])))
    print("k₃ = 0 的 %d 个：P = 1，这道门对它们恒不触发。"
          % sum(1 for r in rows if r["k3"] == 0))
    print()

    print("=== 第三步：λ₃ 的模型比较（同一批候选、同一批 k₃，无抽样噪声）===")
    models = [
        ("只有量化项（均匀）", lambda r: r["lam_q_uniform"]),
        ("只有量化项（含死时间闭式）", lambda r: r["lam_q"]),
        ("只有本底项（far，无安全系数）", lambda r: r["lam_b_far"]),
        ("只有本底项（loc，无安全系数）", lambda r: r["lam_b_loc"]),
        ("量化 + 本底（安全系数 1）", lambda r: r["lam_q"] + max(r["lam_b_far"], r["lam_b_loc"])),
        ("v12：量化 + %.0f×本底" % args.safety, lambda r: r["lam"]),
    ]
    print("%-30s %11s %9s %9s %9s" % ("λ₃ 模型", "Σλ₃", "杀真值", "否决其余", "中位 λ₃"))
    for name, fn in models:
        lams = [max(fn(r), 0.0) for r in rows]
        ps = [pois_ge(r["k3"], l) for r, l in zip(rows, lams)]
        a = sum(1 for r, p in zip(rows, ps) if r["lightning"] and p < THRESHOLD)
        b = sum(1 for r, p in zip(rows, ps) if not r["lightning"] and p < THRESHOLD)
        print("%-30s %11.5f %9d %9d %9.3e" % (name, sum(lams), a, b, np.median(lams)))
    print()
    print("本底项 far vs loc（同一候选两种本底段）：")
    bf = np.array([r["lam_b_far"] for r in rows])
    bl = np.array([r["lam_b_loc"] for r in rows])
    ok = (bf > 0) & (bl > 0)
    print("   loc / far：中位 %.3f，四分位 %.3f–%.3f，全距 %.3f–%.3f"
          % (np.median(bl[ok] / bf[ok]), np.percentile(bl[ok] / bf[ok], 25),
             np.percentile(bl[ok] / bf[ok], 75), (bl[ok] / bf[ok]).min(), (bl[ok] / bf[ok]).max()))
    print("   两者对判决的影响：取 far %d 个被否，取 loc %d 个被否，取大者 %d 个"
          % (sum(1 for r in rows if pois_ge(r["k3"], r["lam_q"] + args.safety * r["lam_b_far"]) < THRESHOLD),
             sum(1 for r in rows if pois_ge(r["k3"], r["lam_q"] + args.safety * r["lam_b_loc"]) < THRESHOLD),
             sum(1 for r in rows if r["p"] < THRESHOLD)))
    print()
    print("本底项 / 量化项（哪一项在做决定）：")
    q_ = np.array([r["lam_q"] for r in rows])
    b_ = np.array([max(r["lam_b_far"], r["lam_b_loc"]) for r in rows])
    rr = b_ / q_
    print("   中位 %.1f 倍；λ₃ 里本底占比中位 %.3f"
          % (np.median(rr), np.median(b_ / (q_ + b_))))
    dec = [r for r in rows if r["k3"] >= 1]
    if dec:
        qd = np.array([r["lam_q"] for r in dec])
        bd = np.array([max(r["lam_b_far"], r["lam_b_loc"]) for r in dec])
        print("   只看会触发判据的 %d 个（k₃ ≥ 1）：本底 / 量化 中位 %.1f 倍，本底占比中位 %.3f"
              % (len(dec), np.median(bd / qd), np.median(bd / (qd + bd))))
    print()

    print("=== 死时间闭式 vs 均匀式（同一批候选，无抽样噪声）===")
    ratio = np.array([r["lam_q"] / r["lam_q_uniform"] for r in rows if r["lam_q_uniform"] > 0])
    sat = np.array([r["sat"] for r in rows])
    print("λ₃ᵍ 含死时间 / 均匀：中位 %.4f，全距 %.4f–%.4f" % (np.median(ratio), ratio.min(), ratio.max()))
    for lo, hi in ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4)):
        msk = (sat >= lo) & (sat < hi)
        if msk.sum():
            print("   saturation %.2f–%.2f：%2d 个，比值中位 %.4f，最大 %.4f"
                  % (lo, hi, msk.sum(), np.median(ratio[msk]), ratio[msk].max()))
    print("   比值恒 ≥ 1：死时间把同探头的事例互相推开，逐格占据概率 P_d(t) 在窗中间抬高、")
    print("   两端压低，而 Σ_t P_d(t) = n_d 不变；λ₃ 是 ≥3 个 P 的乘积之和，按凸性只能升。")
    print("   **符号不会翻**，稀疏窗上只是趋近 1，不是掉到 1 以下。")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "lightning", "n", "w_us", "saturation", "k2", "k3", "f3",
                    "lambda3_quant_exact", "lambda3_quant_uniform",
                    "lambda3_bkg_far", "lambda3_bkg_loc", "safety", "lambda3_total",
                    "k3_over_lambda3", "p_value", "verdict_v12"])
        for r in sorted(rows, key=lambda r: r["start"]):
            w.writerow([r["start"], int(r["lightning"]), r["n"], "%.2f" % r["w_us"],
                        "%.4f" % r["sat"], r["k2"], r["k3"], "%.4f" % r["f3"],
                        "%.6e" % r["lam_q"], "%.6e" % r["lam_q_uniform"],
                        "%.6e" % r["lam_b_far"], "%.6e" % r["lam_b_loc"],
                        "%.1f" % args.safety, "%.6e" % r["lam"],
                        "%.6e" % r["ratio"] if r["k3"] else "",
                        "%.6e" % r["p"], "veto" if r["p"] < THRESHOLD else "keep"])
    print()
    print("写出 %s" % args.out)


if __name__ == "__main__":
    main()
