"""把 f₃ 的偶然期望（`f3_chance.py`）与本底对照（`f3_background.py`）合成一张表。

这是 f₃ 判据三步法第 (b) 步的收口：**判据成立的条件是"实测 / 偶然"有数量级分离，
不是"f₃ 大"。** 三个"偶然"要分清，它们量的不是同一件事：

  偶然 1（量化偶然）—— 窗内 n 个互不相干的事例撞进同一个 2⁻²² s 格。`f3_chance.py`。
  偶然 2（本底粒子）—— 真本底里的穿星粒子恰好落进这个窗。`f3_background.py` 的
                      λ₃ᵇᵏᵍ = 本底三重簇率 × 窗长。
  偶然 3（旧口径）—— 随机取 n 个**连续**本底事例算 f₃ > 0 的占比。它匹配计数不匹配
                      窗长，跨度是候选窗的几十到几百倍，**不能直接和候选的 f₃ 比**。

另外报一条结构上限：GRID-03B 四路探头、同一探头从不同戳，所以同戳簇最多 4 重；
`min_number = 8` ⇒ **只含一个簇的候选 f₃ ≤ 4/8 = 0.5**。阈值 `f₃ > 0.5` 因此对
"一次穿越"的粒子恒不触发，只对"重复穿越"触发。

用法：
    python3 f3_summary.py --chance f3_chance.csv --background f3_background.csv
"""

import argparse
import csv
import math

import numpy as np


def _f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def pois_ge(k, lam):
    """P(X ≥ k)，X ~ Poisson(lam)。

    **直接累尾，不用 `1 − 前 k 项`。** 后者在尾部小的时候是灾难性相消：λ ~ 1e-3 时
    k = 3 已有 2.6e-8 的相对误差，**k ≥ 5 直接返回 0**（真值 8.3e-18）；λ = 1.3e-5、
    k = 3 时错 21%。本判据要在 1e-3 附近比大小，而全池里 λ₃ 能小到 1e-6 量级，
    正落在会出错的那一档。直接累尾在同样参数下相对误差 ≤ 5e-15。
    """
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    total = 0.0
    for i in range(k, k + 1000):
        term = math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1))
        total += term
        if term < 1e-18 * max(total, 1e-300):
            break
    return min(1.0, total)


def q(a, name, unit=""):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "%s: 无" % name
    return "%s: 中位 %.3g，四分位 %.3g–%.3g，最大 %.3g%s" % (
        name, np.median(a), np.percentile(a, 25), np.percentile(a, 75), a.max(), unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chance", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("-o", "--out", default="f3_step_b.csv")
    args = ap.parse_args()

    ch = {r["start"]: r for r in csv.DictReader(open(args.chance))}
    bg = {r["start"]: r for r in csv.DictReader(open(args.background))}
    keys = sorted(set(ch) & set(bg))
    if len(keys) != len(ch):
        print("警告：两表只对上 %d / %d 个候选" % (len(keys), len(ch)))

    rows = []
    for k in keys:
        c, b = ch[k], bg[k]
        n = int(c["n"])
        f3 = _f(c, "f3_obs")
        k3 = int(c["k3_obs"])
        lam_q = _f(c, "lambda3_perdet")
        lam_b = _f(b, "far_lambda3_in_T")
        lam = lam_q + lam_b
        rows.append(dict(
            start=k, lightning=c["lightning"], n=n, T_us=c["T_us"], n_det=c["n_det"],
            f2_obs=c["f2_obs"], f3_obs="%.4f" % f3, k3_obs=k3,
            f3_chance_naive=c["f3_chance_naive"], f3_chance_perdet=c["f3_chance_perdet"],
            lambda3_quant=c["lambda3_perdet"],
            bkg_rate_cps=b["far_rate_cps"], bkg_r3_per_s=b["far_r3_per_s"],
            lambda3_bkg=b["far_lambda3_in_T"],
            lambda3_total="%.3e" % lam,
            p_poisson="%.3e" % pois_ge(k3, lam) if k3 else "1",
            ctrlA_frac=b.get("ctrlA_frac_f3_pos", ""),
            ctrlA_frac_n_ge8=b.get("ctrlA_frac_n_ge8", ""),
            ctrlB_frac=b.get("ctrlB_frac_f3_pos", ""),
            ctrlB_span_over_T=b.get("ctrlB_span_over_T", ""),
            ratio_perdet=c["ratio_perdet"],
        ))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    A = lambda key: np.array([_f(r, key) for r in rows])
    f3 = A("f3_obs")
    k3 = np.array([r["k3_obs"] for r in rows])
    lam_q = A("lambda3_quant")
    lam_b = A("lambda3_bkg")
    lam = A("lambda3_total")
    lit = np.array([r["lightning"] == "1" for r in rows])
    nz = f3 > 0

    print("候选 %d 个（其中 7 个有闪电认证），f₃ > 0 的 %d 个，三重簇总数 %d"
          % (len(rows), nz.sum(), k3.sum()))
    print()
    print("== 偶然 1：量化偶然（窗内 n 个互不相干事例撞进同一格）==")
    print("  " + q(A("f3_chance_perdet"), "E[f₃] 逐探头口径"))
    print("  " + q(A("f3_chance_naive"), "E[f₃] 朴素口径"))
    print("  λ₃ 合计 %.4f（逐探头）→ 38 个候选里因量化偶然出现 ≥1 个三重簇的期望个数"
          % lam_q.sum())
    print()
    print("== 偶然 2：本底粒子落进窗 ==")
    print("  " + q(A("bkg_rate_cps"), "本底率", " c/s"))
    print("  " + q(A("bkg_r3_per_s"), "本底三重簇率", " 个/s"))
    print("  " + q(lam_b, "λ₃ᵇᵏᵍ = 率 × 窗长"))
    print("  λ₃ᵇᵏᵍ 合计 %.4f → 38 个真暴的窗里偶然套进本底粒子三重的期望个数" % lam_b.sum())
    print("  随机同长窗实测 f₃>0 占比：" + q(A("ctrlA_frac"), "ctrlA"))
    print("  同一批随机窗里计数 ≥ 8 的占比：" + q(A("ctrlA_frac_n_ge8"), "ctrlA_n≥8"))
    print()
    print("== 偶然 3：旧口径（随机取 n 个连续本底事例）==")
    print("  " + q(A("ctrlB_frac"), "f₃ > 0 的占比"))
    print("  " + q(A("ctrlB_span_over_T"), "这 n 个事例的跨度 ÷ 候选窗长", " 倍"))
    print("  → 旧口径量的是几十毫秒里有没有粒子，不是候选窗里有没有；两者差这么多倍，")
    print("    所以 OPEN-QUESTIONS 里\"本底对照 7% 非零\"不能与候选的 12/38 = 32% 并排比。")
    print()
    print("== 合起来 ==")
    tot = lam.sum()
    print("  两种偶然相加 λ₃ 合计 %.4f，实测带三重簇的候选 %d 个" % (tot, nz.sum()))
    print("  实测 / 偶然（个数口径）= %.0f 倍" % (nz.sum() / tot))
    if nz.any():
        r = f3[nz] / A("f3_chance_perdet")[nz]
        print("  非零者逐候选 实测 f₃ / 量化偶然 E[f₃]：中位 %.1e，最小 %.1e" % (np.median(r), r.min()))
    print()
    print("== 7 个闪电认证 ==")
    print("  f₃ 实测全为 0；它们各自的 λ₃ 合计 %.4f（量化 %.4f + 本底 %.4f）"
          % (lam[lit].sum(), lam_q[lit].sum(), lam_b[lit].sum()))
    print("  → 这 7 个即便全是普通光子暴，也本来就不该出现三重簇，f₃ = 0 与偶然相容。")
    print()
    print("== 逐候选 Poisson 检验（k₃ 实测对 λ₃ 期望）==")
    print("  %-25s %3s %8s %6s %4s %10s %10s" % ("start", "n", "T_us", "f₃", "k₃", "λ₃", "P(≥k₃)"))
    for r in rows:
        if r["k3_obs"]:
            print("  %-25s %3s %8s %6s %4d %10s %10s"
                  % (r["start"][:19], r["n"], r["T_us"], r["f3_obs"], r["k3_obs"],
                     r["lambda3_total"], r["p_poisson"]))
    print()
    print("== 阈值的结构上限 ==")
    print("  同戳簇最多 4 重（四路探头、同探头从不同戳），min_number = 8")
    print("  ⇒ 只含 1 个簇的候选 f₃ ≤ 4/8 = 0.500，`f₃ > 0.5` 对它恒不触发")
    for nn in (8, 9, 10, 12, 16):
        print("     n = %2d：单个 4 重簇 f₃ = %.3f，单个 3 重簇 f₃ = %.3f"
              % (nn, 4 / nn, 3 / nn))
    one = np.array([r["k3_obs"] == 1 for r in rows])
    two = np.array([r["k3_obs"] >= 2 for r in rows])
    print("  实测印证：k₃ = 1 的 %d 个候选 f₃ 最大 %.3f；k₃ ≥ 2 的 %d 个最小 %.3f"
          % (one.sum(), f3[one].max() if one.any() else float("nan"),
             two.sum(), f3[two].min() if two.any() else float("nan")))


if __name__ == "__main__":
    main()
