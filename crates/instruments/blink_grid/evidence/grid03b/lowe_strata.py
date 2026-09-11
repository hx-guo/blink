"""GRID-03B 低能端（E_MIN < 30 keV）占比：按纪元 × 速率分层，给全任务代表值。

背景：四个成员先前给出的 0.07% / 3.09% / 23.6% / 3.24% / 56–76% 全都是单过境或
少数过境的抽样，彼此对不上。分歧的根子有两条，**必须同时分层**：

  纪元 —— 03B 在 2023-03 有一个台阶，之前 30 keV 以下收得多、之后几乎不收；
  速率 —— 同一纪元内，辐射带过境（高速率）的低能占比比低纬安静过境高一到两个数量级。

本脚本吃 `grid03b_lowe.py` 的全量产物（788 天、1529 次过境，逐过境一行），
按这两个轴分层，并给两个口径的全任务值：
  逐过境中位（每次过境一票，代表"一次典型过境长什么样"）；
  计数加权（Σ低能事例 / Σ事例，代表"整批数据里低能占多少"，被高速率过境主导）。

用法：
    python3 lowe_strata.py lowe/all.csv [--split 2023-03-15]
"""

import argparse
import csv

import numpy as np

RATE_EDGES = [0, 600, 3000, 20000, 10**9]
RATE_NAMES = ["< 600 c/s（低纬安静）", "600–3k", "3k–20k", "> 20k c/s（辐射带）"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--split", default="2023-03-15", help="纪元台阶的分界日")
    ap.add_argument("--rate-axis", choices=["kept", "total"], default="kept",
                    help="分层用的速率：kept = 30 keV 以上（默认，去掉自循环）；total = 全部 EVT_TYPE==1")
    args = ap.parse_args()

    day, rate, f, peak, n = [], [], [], [], []
    for r in csv.DictReader(open(args.csv)):
        if not r.get("rate4") or not r.get("f_lt_ch8"):
            continue
        day.append(r["day"])
        rate.append(float(r["rate4"]))
        f.append(float(r["f_lt_ch8"]))
        peak.append(int(r["peak_ch"]))
        n.append(int(r["n_type1"]))
    day = np.array(day)
    rate, f, peak, n = map(np.array, (rate, f, peak, n))
    pre = day < args.split
    # 分层用 30 keV 以上的速率，不用总速率：总速率本身被低能成分抬高，拿它分层
    # 等于把因变量塞进自变量——台阶前的过境会被系统性地推进高速率档。
    rate_kept = rate * (1.0 - f)
    if args.rate_axis == "kept":
        rate = rate_kept

    print("过境 %d 次（%s .. %s），事例合计 %.3g 个（EVT_TYPE==1，未加 30 keV 阈）"
          % (len(day), min(day), max(day), n.sum()))
    print("纪元分界 %s：之前 %d 次过境，之后 %d 次" % (args.split, pre.sum(), (~pre).sum()))
    print()
    print("%-22s %6s %8s %8s %8s %8s %6s" % ("分层", "过境", "f 中位", "f 四分位下", "四分位上", "计数加权", "峰道"))
    for tag, m0 in (("台阶前", pre), ("台阶后", ~pre)):
        for i in range(len(RATE_EDGES) - 1):
            m = m0 & (rate >= RATE_EDGES[i]) & (rate < RATE_EDGES[i + 1])
            if m.sum() == 0:
                print("%-22s %6d" % (tag + " " + RATE_NAMES[i], 0))
                continue
            wq = (f[m] * n[m]).sum() / n[m].sum()
            print("%-22s %6d %8.4f %8.4f %8.4f %8.4f %6d"
                  % (tag + " " + RATE_NAMES[i], m.sum(), np.median(f[m]),
                     np.percentile(f[m], 25), np.percentile(f[m], 75), wq,
                     int(np.median(peak[m]))))
    print()
    for tag, m in (("台阶前", pre), ("台阶后", ~pre), ("全任务", np.ones(len(f), bool))):
        wq = (f[m] * n[m]).sum() / n[m].sum()
        print("%-8s 逐过境中位 %.4f（四分位 %.4f–%.4f，全距 %.4f–%.4f）；计数加权 %.4f；峰道中位 %d"
              % (tag, np.median(f[m]), np.percentile(f[m], 25), np.percentile(f[m], 75),
                 f[m].min(), f[m].max(), wq, int(np.median(peak[m]))))
    print()
    print("台阶两侧的速率分布（说明速率分层不是自变量混进来的）：")
    for tag, m in (("台阶前", pre), ("台阶后", ~pre)):
        print("   %s 速率中位 %.0f c/s，四分位 %.0f–%.0f；> 3 kc/s 的过境占 %.1f%%"
              % (tag, np.median(rate[m]), np.percentile(rate[m], 25),
                 np.percentile(rate[m], 75), 100 * (rate[m] > 3000).mean()))
    print()
    print("旧数字各自落在哪一档（拿 f 最接近的分层反查）：")
    for name, val in (("grid03b 旧报 安静过境", 0.0007), ("grid03b 旧报 含辐射带", 0.0903),
                      ("grid02 报 03B 低端", 0.236), ("grid02 报 03B 高端", 0.441),
                      ("grid07 撤回前 低端", 0.56)):
        hit = []
        for tag, m0 in (("台阶前", pre), ("台阶后", ~pre)):
            for i in range(len(RATE_EDGES) - 1):
                m = m0 & (rate >= RATE_EDGES[i]) & (rate < RATE_EDGES[i + 1])
                if m.sum() >= 5 and np.percentile(f[m], 10) <= val <= np.percentile(f[m], 90):
                    hit.append("%s %s" % (tag, RATE_NAMES[i].split("（")[0]))
        print("   %-22s f = %.4f → 落在 %s" % (name, val, "、".join(hit) if hit else "任何分层的 10–90% 区间之外"))


if __name__ == "__main__":
    main()
