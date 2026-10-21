"""SVOM 三路 GRD 的「单探头占比」还剩多少判别力。

HXMT 那边发现：按**物理分组**聚合的占比（单箱）判别力远高于单路读出（AUC 0.93
对 0.58）。SVOM/GRM 只有 3 个 GRD，读出粒度与分组粒度本来就是一回事，所以这条
经验在 SVOM 上只有一个层级可算。这里量它在**已经过了 0.8 否决**的显著候选池里
还剩多少判别力，以及池子里有没有残留的「某一路独占」的假信号人群。

判别量与选样标准在构造上无关：占比来自事例的探头号，闪电关联来自 WWLLN，
两者互不知情。零假设不是「1/3」——三路等概率时，n 个计数里最大一路的占比
期望本来就明显高于 1/3（n=20 时 0.44），所以用**逐候选按自己的 n 抽多项分布**
当基准，而不是拿别的候选当基准（那是循环的）。

输入：`svom_features.py` 的特征表与 `blink wwlln` 导出的关联表。
用法: python3 diag_svom_detector_share.py <features.csv> <assoc.csv>
"""
import csv
import sys

import numpy as np


def auc(pos, neg, key, sign=1):
    """Mann-Whitney AUC 与 Hanley-McNeil 标准误。"""
    x = np.array([sign * r[key] for r in pos], float)
    y = np.array([sign * r[key] for r in neg], float)
    n = sum((y < v).sum() + 0.5 * (y == v).sum() for v in x)
    a = n / (len(x) * len(y))
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (len(x) - 1) * (q1 - a * a)
                  + (len(y) - 1) * (q2 - a * a)) / (len(x) * len(y)))
    return a, se


def main(fcsv, acsv):
    feat = {r["start"]: r for r in csv.DictReader(open(fcsv))}
    assoc = {r["start"]: r for r in csv.DictReader(open(acsv))}
    rows = []
    for s, r in feat.items():
        a = assoc.get(s)
        rows.append(dict(start=s, fa=float(r["fa"]), frac=float(r["det_frac_max"]),
                         nhit=int(r["n_det_hit"]), n=int(r["n_core"]),
                         pir=float(r["pi_med_ratio"]),
                         cov=int(a["in_coverage"]) if a else -1,
                         ass=int(a["associated"]) if a else -1))
    sig = [r for r in rows if r["fa"] <= 1e-5]
    conf = [r for r in sig if r["cov"] == 1 and r["ass"] == 1]
    unc = [r for r in sig if r["cov"] == 1 and r["ass"] == 0]
    out = [r for r in sig if r["cov"] == 0]
    print("显著候选 %d：闪电证实 %d，覆盖内未证实 %d，覆盖外 %d"
          % (len(sig), len(conf), len(unc), len(out)))

    print("\n== 判别力（证实 vs 覆盖内未证实）==")
    for key, sign, name in (("frac", -1, "单 GRD 占比（越小越像真）"),
                            ("pir", +1, "能道中位数比"),
                            ("n", +1, "窗内计数"),
                            ("nhit", +1, "命中探头数")):
        a, se = auc(conf, unc, key, sign)
        print("  AUC %-24s %.3f ± %.3f" % (name, a, se))
    print("  （HXMT 的对照：单箱 0.93、单路 0.58）")

    print("\n== 占比分布 vs 三路等概率的多项分布基准（逐候选按自己的 n 抽）==")
    rng = np.random.default_rng(7)
    for nm, grp in (("闪电证实", conf), ("覆盖内未证实", unc), ("覆盖外显著", out)):
        v = np.array([r["frac"] for r in grp])
        nn = np.array([max(r["n"], 1) for r in grp])
        sim = np.array([rng.multinomial(k, (1 / 3, 1 / 3, 1 / 3), size=400).max(1) / k
                        for k in nn])
        print("  %-12s N=%3d  实测中位 %.3f（四分位 %.3f–%.3f，最大 %.3f）；"
              "等概率基准中位 %.3f，实测超出基准的占 %.0f%%"
              % (nm, len(v), np.median(v), np.percentile(v, 25), np.percentile(v, 75),
                 v.max(), np.median(sim), 100 * np.mean(v > np.median(sim, 1))))

    print("\n== 有没有「某一路独占」的残留人群：按占比分档看关联率 ==")
    cov = [r for r in sig if r["cov"] == 1]
    for lo, hi in ((0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.80)):
        m = [r for r in cov if lo <= r["frac"] < hi]
        if not m:
            continue
        k = sum(r["ass"] == 1 for r in m)
        print("  占比 %.2f–%.2f  N=%3d  关联 %3d（%.1f%%）" % (lo, hi, len(m), k, 100 * k / len(m)))
    hi = [r for r in sig if r["frac"] > 0.6]
    print("  占比 > 0.6 的显著候选只有 %d 个（覆盖内 %d，其中关联 %d），"
          "0.8 的否决线对现在的池子已经不起作用"
          % (len(hi), sum(r["cov"] == 1 for r in hi),
             sum(r["cov"] == 1 and r["ass"] == 1 for r in hi)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
