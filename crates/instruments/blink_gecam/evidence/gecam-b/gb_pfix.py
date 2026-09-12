"""GECAM-B：泊松 p 值从 `P(X > count)` 改成 `P(X ≥ count)` 之后，目录动了多少。

`blink_algorithms` 现版的 `Poisson::sf(count)` 给的是 `P(X > count)`，而观测到
`count` 个的 p 值应当是 `P(X ≥ count)`——**极端尾巴上"恰好这么多"那一项就是
尾巴的主体**，漏掉它等于系统性低估 `fa`。低估因子在尾部精确趋于 `(count+1)/λ`。

**不改代码，离线重算。** 三个数：
1. `(count+1)/λ` 的中位与跨度（跨度有多大就决定排序被搅动多少——SVOM 那个
   "改正因子中位只有 6.54、形状不动"**不能搬到 GECAM**）；
2. 改对后跨判选阈的进/出各多少；
3. **跨阈那些的外部真值怎么说**——这是三项里唯一能判好坏的。前两项只说"变了多少"。

召回口径用二维扫描定下的 **[−1000, +3000] µs**（141/147，偶然期望 4.45）。

用法: python3 gb_pfix.py
"""

import csv
import datetime as dt
import json
import os

import numpy as np
from scipy import stats as sps

EPOCH = (2019, 1, 1)
ROOT = os.environ.get("GB_SIGNALS", "/scratchfs2/gecam/guohx/gecambrun/b2/data/GECAM-B")
CATALOG = os.environ.get("GB_CATALOG", "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")
MATCH_LO, MATCH_HI = -1000e-6, 3000e-6      # 二维扫描定下的非对称匹配窗（秒）
YEAR = 365.25 * 86400.0


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def main():
    catalog = list(csv.DictReader(open(CATALOG)))
    days = sorted({c["UT"][:10] for c in catalog})
    hours = {c["UT"][:13] for c in catalog}

    t0s, cnt, lam, fa_old, keys = [], [], [], [], []
    for d in days:
        p = f"{ROOT}/{d[:4]}/{d[5:7]}/{d.replace('-', '')}_signals.json"
        if not os.path.exists(p):
            continue
        for s in json.load(open(p)):
            k = s["start"][:13]
            if k not in hours:
                continue
            t0s.append(met(s["start"]) + s["delay"])
            cnt.append(s["count"])
            lam.append(s["mean"])
            fa_old.append(s["false_positive_per_year"])
            keys.append(k)
    t0s = np.array(t0s)
    cnt = np.array(cnt, float)
    lam = np.array(lam, float)
    fa_old = np.array(fa_old, float)
    print(f"候选 {t0s.size}（138 个 TGF 小时）")

    # 1. 改正因子
    fac = (cnt + 1) / lam
    print()
    print("【1】改正因子 (count+1)/λ")
    q = np.percentile(fac, [10, 25, 50, 75, 90])
    print(f"  p10 = {q[0]:.4g}   p25 = {q[1]:.4g}   p50 = {q[2]:.4g}   "
          f"p75 = {q[3]:.4g}   p90 = {q[4]:.4g}")
    print(f"  min = {fac.min():.4g}   max = {fac.max():.4g}   "
          f"p90/p10 = {q[4] / q[0]:.4g} 倍（跨度 = 排序被搅动的程度）")

    # 精确比值 ratio = P(X≥c)/P(X>c)。**必须在对数空间算**：`P(X>c)` 在最显著那批
    # 候选上会小到 1e−300 以下，直接相除得 0/0 = nan，nan 一路传进分位数，看着像
    # "算不出来"，其实是把最显著那批整段丢了——而那批恰恰是最要紧的。
    log_ratio = sps.poisson.logsf(cnt - 1, lam) - sps.poisson.logsf(cnt, lam)
    finite = np.isfinite(log_ratio)
    print(f"  对数空间可算的：{finite.sum()}/{finite.size}"
          f"（直接相除会下溢的有 {int((sps.poisson.sf(cnt, lam) <= 0).sum())} 个）")
    rel = np.abs(np.exp(log_ratio[finite]) - fac[finite]) / fac[finite]
    print(f"  与精确式 P(X≥c)/P(X>c) 的相对差：中位 {np.median(rel):.3g}，"
          f"p99 {np.percentile(rel, 99):.3g}（渐近式够用）")

    # fa ∝ sf，所以新旧之比就是 sf 之比，**不需要反解每年试验数**
    #（那个数逐候选随窗宽走、不是常数，反解出来的全局值没有意义）
    ratio = np.where(finite, np.exp(np.minimum(log_ratio, 700.0)), fac)
    fa_new = fa_old * ratio
    assert np.all(fa_new >= fa_old - 1e-300), "fa 逐点只增：P(X≥c) ≥ P(X>c) 恒成立"

    # 2/3. 跨阈的进出 + 外部真值
    print()
    print("【2】改对后跨判选阈的进/出")
    ut = np.array([met(c["UT"]) for c in catalog])
    order = np.argsort(t0s)
    ts, idx = t0s[order], order

    def is_tgf(i):
        d = t0s[i] - ut
        return bool(np.any((d >= MATCH_LO) & (d <= MATCH_HI)))

    for thr in (1e-5, 1e-3, 1.0, 20.0):
        a = fa_old <= thr
        b = fa_new <= thr
        out = a & ~b
        into = b & ~a
        print(f"  阈 fa ≤ {thr:<6g}：旧 {a.sum():>7d} → 新 {b.sum():>7d} "
              f"（掉出 {out.sum()}，新进 {into.sum()}）")
        if out.sum():
            n_tgf_out = sum(1 for i in np.flatnonzero(out) if is_tgf(i))
            print(f"      掉出去的 {out.sum()} 个里，是已发表 TGF 的：{n_tgf_out}")
        if into.sum():
            n_tgf_in = sum(1 for i in np.flatnonzero(into) if is_tgf(i))
            print(f"      新进来的 {into.sum()} 个里，是已发表 TGF 的：{n_tgf_in}")

    print()
    print("【3】147 个已发表 TGF 在新旧口径下的 fa（匹配窗 [−1000, +3000] µs）")
    rows = []
    for c in catalog:
        u = met(c["UT"])
        d = t0s - u
        m = (d >= MATCH_LO) & (d <= MATCH_HI)
        if not m.any():
            continue
        j = np.flatnonzero(m)[np.argmin(fa_old[m])]
        rows.append((fa_old[j], fa_new[j], fac[j], cnt[j], lam[j]))
    rows = np.array(rows)
    print(f"  匹配上 {rows.shape[0]}/147")
    print(f"  改正因子在这 {rows.shape[0]} 个上：中位 {np.median(rows[:, 2]):.4g}，"
          f"5–95% {np.percentile(rows[:, 2], 5):.4g} .. {np.percentile(rows[:, 2], 95):.4g}")
    for thr in (1e-5, 1e-3, 1.0):
        a = int((rows[:, 0] <= thr).sum())
        b = int((rows[:, 1] <= thr).sum())
        print(f"  阈 fa ≤ {thr:<6g}：旧 {a}/{rows.shape[0]} → 新 {b}/{rows.shape[0]}"
              f"   掉出 {int(((rows[:, 0] <= thr) & (rows[:, 1] > thr)).sum())}")

    print()
    print("【形状】改正是否改变幂律形状（对 fa 排序的影响）")
    for thr in (1e-5, 1.0):
        a = fa_old <= thr
        b = fa_new <= thr
        both = a & b
        print(f"  阈 {thr:g}：旧集合 {a.sum()}、新集合 {b.sum()}、交集 {both.sum()}"
              f"（Jaccard {both.sum() / max((a | b).sum(), 1):.4f}）")
    r = sps.spearmanr(fa_old, fa_new).statistic
    print(f"  fa_old 与 fa_new 的秩相关 = {r:.6f}（= 1 则排序完全不动）")
    # 第 8b 条：要比的不是改正因子自己跨多少，是它的散布 ÷ 被改量自身的散布
    lf = np.log10(np.maximum(fa_old, 1e-300))
    lr = np.log10(np.maximum(ratio, 1e-300))
    span_r = float(np.percentile(lr, 99) - np.percentile(lr, 1))
    span_f = float(np.percentile(lf, 99) - np.percentile(lf, 1))
    print(f"  log10(改正因子) 跨度 p1–p99 = {span_r:.3f} 个数量级")
    print(f"  log10(fa) 自身跨度 p1–p99 = {span_f:.3f} 个数量级"
          f"（p1/p50/p99 = {np.percentile(lf, 1):.1f}/{np.percentile(lf, 50):.1f}"
          f"/{np.percentile(lf, 99):.1f}）")
    print(f"  比值 = {span_r / span_f:.4f} ← 这个数决定排序被搅动多少，不是改正因子自己的跨度")
    print()
    print("  固定名额选法（按 fa 排序取前 N）——固定阈是严格套嵌只出不进，固定名额才会换人：")
    o_old = np.argsort(fa_old)
    o_new = np.argsort(fa_new)
    for N in (10, 100, 1000, 10000, 100000):
        if N > fa_old.size:
            break
        a, b = set(o_old[:N].tolist()), set(o_new[:N].tolist())
        print(f"    前 {N:>6d} 名：换掉 {N - len(a & b):>5d} 个 = {(N - len(a & b)) / N * 100:.2f}%")


if __name__ == "__main__":
    main()
