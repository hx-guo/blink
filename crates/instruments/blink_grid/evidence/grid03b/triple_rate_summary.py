"""汇总 `triple_rate_scan.py` 的分片产物，回答三个问题。

  1. 全任务有多少次过境 `frac_ev_in3` 高，占比多少、落在哪些日期；
  2. 它们是成群（同一天、同一轨道段）还是散的；
  3. **`frac_ev_in3` 与 |偶极磁纬| 的关系。** 若它就是辐射带的影子，这道门与磁纬强相关，
     那必须说清它挡的是**环境**而不是**位置**——否则等于偷偷加了一道磁纬门。

**入口先硬断言「产出天数 == 应有天数」，不等非零退出。** 队列空 ≠ 跑完：残缺输入上跑完
整条汇总链会产物齐全、格式正确、数字看着合理且不报错，栽过一次（5.7% 的数据上出了结论）。

用法：
    python3 triple_rate_summary.py <tripscan 目录> [--nchunk 4] [--sat GRID-03B]
                                   [--thr 0.05] [--archive <fits7 根>]
"""

import argparse
import collections
import csv
import glob
import os

import numpy as np


def dt_coverage(rows, root):
    """复现 `grid03b_dt_series.py` 的"每天按文件大小取中间两个"，量它排除掉了哪一档速率。

    未决项 6 的"788 天没变过"就建立在这个子集上，而该节自己写着 22–32 tick 的偏离是
    **高计数率下估计量的伪影**。若这个选法排除的恰是高速率过境，那条结论的适用范围就
    被悄悄收窄了——**选样准则与估计量的已知失效模式相关**。这里把它变成一个数。
    """
    rate = {r["pass_file"]: _f(r["rate_cps"]) for r in rows if r["rate_cps"]}
    picked = set()
    for daydir in sorted(glob.glob(root + "/*/*/*")):
        vers = sorted(glob.glob(daydir + "/evt_v*"))
        if not vers:
            continue
        files = sorted(glob.glob(vers[-1] + "/*.fits"), key=os.path.getsize)
        if not files:
            continue
        mid = len(files) // 2
        pick = files[max(mid - 1, 0):mid + 1] if len(files) > 1 else files
        picked.update(os.path.basename(f) for f in pick)
    a_in = np.array([v for k, v in rate.items() if k in picked])
    a_out = np.array([v for k, v in rate.items() if k not in picked])
    if a_in.size == 0 or a_out.size == 0:
        print("   无法复现选样（文件名对不上？）")
        return
    print("   进样本 %d 次，未进 %d 次（覆盖 %.1f%%）"
          % (a_in.size, a_out.size, 100.0 * a_in.size / (a_in.size + a_out.size)))
    print("   %-10s %9s %9s %9s %9s %9s" % ("", "p5", "中位", "p95", "p99", "最大"))
    for name, a in (("进样本", a_in), ("未进样本", a_out)):
        print("   %-10s %9.0f %9.0f %9.0f %9.0f %9.0f"
              % (name, np.percentile(a, 5), np.median(a), np.percentile(a, 95),
                 np.percentile(a, 99), a.max()))
    print("   中位之比（未进 / 进）= %.2f 倍，最大之比 = %.2f 倍"
          % (np.median(a_out) / max(np.median(a_in), 1e-9), a_out.max() / max(a_in.max(), 1e-9)))
    for t in (3000, 5000, 8000, 15000):
        n_in = int((a_in > t).sum())
        n_out = int((a_out > t).sum())
        print("   速率 > %5d c/s：进样本 %d 次，未进样本 %d 次" % (t, n_in, n_out))
    hi_in = (a_in > 5000).sum()
    print()
    if hi_in == 0:
        print("   ⇒ **高速率档（> 5000 c/s）在死时间样本里一次都没有**：未决项 6 的"
              "「788 天没变过」")
        print("     完全没覆盖这一档。现行结论若有任何一条落在这个速率范围，就要额度重算。")
    else:
        print("   ⇒ 高速率档在样本里有 %d 次，覆盖不是零，但仍需按比例限定措辞。" % hi_in)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def assert_complete(d, nchunk, expected_days):
    """硬断言：4 份 done 文件齐、Σdays_done == 应有天数。不齐就退出，不往下算。"""
    dones = sorted(glob.glob(os.path.join(d, "done_*.txt")))
    if len(dones) != nchunk:
        raise SystemExit("done 文件 %d 份，应有 %d 份 —— worker 没跑完，不汇总"
                         % (len(dones), nchunk))
    tot_done = tot_exp = tot_pass = 0
    nopos = 0.0
    for p in dones:
        kv = dict(line.strip().split(",", 1) for line in open(p) if "," in line)
        tot_done += int(kv["days_done"])
        tot_exp += int(kv["days_expected"])
        tot_pass += int(kv["passes"])
        nopos += float(kv["no_position_seconds"])
    print("完成检查：%d 份 done，Σdays_done = %d，Σdays_expected = %d，过境 %d 次"
          % (len(dones), tot_done, tot_exp, tot_pass))
    if expected_days is not None and tot_exp != expected_days:
        raise SystemExit("分片合计应有 %d 天，归档里是 %d 天 —— 分片没盖全，不汇总"
                         % (tot_exp, expected_days))
    if tot_done > tot_exp:
        raise SystemExit("days_done 大于 days_expected，分片有重叠")
    if tot_done < tot_exp:
        print("！%d 天没有 evt_v* 目录（脚本里 continue 掉的），占 %.1f%%"
              % (tot_exp - tot_done, 100.0 * (tot_exp - tot_done) / tot_exp))
    print("没有位置的秒数合计 %.0f s" % nopos)
    return tot_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--nchunk", type=int, default=4)
    ap.add_argument("--sat", default="GRID-03B")
    ap.add_argument("--thr", type=float, default=0.05, help="拟议的 frac_ev_in3 上限")
    ap.add_argument("--archive", default="/gecamfs/Exchange/GSDC/missions/GRID/{sat}/fits7")
    args = ap.parse_args()

    expected = None
    root = args.archive.format(sat=args.sat)
    if os.path.isdir(root):
        expected = len(glob.glob(root + "/*/*/*"))
    assert_complete(args.dir, args.nchunk, expected)
    print()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "trip_*.csv"))):
        rows.extend(csv.DictReader(open(p)))
    ok = [r for r in rows if r["note"] in ("", None) and r["frac_ev_in3"]]
    err = [r for r in rows if (r["note"] or "").startswith("ERR")]
    few = [r for r in rows if (r["note"] or "") == "too few events"]
    print("过境行 %d：可用 %d，事例太少 %d，读不出 %d" % (len(rows), len(ok), len(few), len(err)))
    if err:
        for r in err[:5]:
            print("   读不出：%s %s %s" % (r["day"], r["pass_file"], r["note"]))

    frac = np.array([_f(r["frac_ev_in3"]) for r in ok])
    rate = np.array([_f(r["rate_cps"]) for r in ok])
    print()
    print("=== 1. frac_ev_in3 的全任务分布 ===")
    qs = [0, 50, 90, 99, 99.9, 100]
    print("   分位 " + "，".join("p%g %.3e" % (q, np.percentile(frac, q)) for q in qs))
    for t in (0.02, 0.05, 0.1, 0.2, 0.5):
        n = int((frac > t).sum())
        print("   > %.2f：%d 次过境（%.4f%%）" % (t, n, 100.0 * n / frac.size))

    hi = [r for r in ok if _f(r["frac_ev_in3"]) > args.thr]
    print()
    print("=== 2. 超过拟议阈 %.2f 的过境：成群还是散的 ===" % args.thr)
    if not hi:
        print("   **一次都没有。** 那么这道门在 03B 全任务上只会命中已知的那一次异常过境，")
        print("   n = 1 这个样本量本身就是结论：这一类是孤例，不是一群。")
    else:
        byday = collections.Counter(r["day"] for r in hi)
        print("   %d 次过境，落在 %d 天上" % (len(hi), len(byday)))
        for day, n in sorted(byday.items()):
            sub = [r for r in hi if r["day"] == day]
            print("   %s  %d 次  frac %s  速率 %s c/s  |磁纬| %s"
                  % (day, n,
                     "/".join("%.3f" % _f(r["frac_ev_in3"]) for r in sub[:4]),
                     "/".join("%.0f" % _f(r["rate_cps"]) for r in sub[:4]),
                     "/".join((r["mlat_at_maxfrac"] or "?") for r in sub[:4])))
        print("   成群判据：%d 天 / %d 次 = 每天 %.2f 次 ⇒ %s"
              % (len(byday), len(hi), len(hi) / len(byday),
                 "成群（同一天多次）" if len(hi) / len(byday) > 1.5 else "散的（各天独立）"))

    print()
    print("=== 3. 不是速率效应吗：frac_ev_in3 对速率 ===")
    edges = [0, 500, 1000, 2000, 3000, 5000, 8000, 1e9]
    for a, b in zip(edges, edges[1:]):
        m = (rate >= a) & (rate < b)
        if m.sum():
            print("   速率 %5.0f–%-8.0f c/s：%5d 次，frac 中位 %.3e，最大 %.3e"
                  % (a, min(b, 99999), m.sum(), np.median(frac[m]), frac[m].max()))

    print()
    print("=== 4. **frac_ev_in3 对 |偶极磁纬|：这道门挡的是环境还是位置** ===")
    acc = collections.defaultdict(lambda: np.zeros(4))
    for p in sorted(glob.glob(os.path.join(args.dir, "mlat_*.csv"))):
        for line in open(p):
            if line.startswith("#") or line.startswith("mlat_lo"):
                continue
            c = line.strip().split(",")
            if len(c) < 6:
                continue
            acc[(float(c[0]), float(c[1]))] += np.array(
                [float(c[2]), float(c[3]), float(c[4]), float(c[5])])
    if not acc:
        print("   没有 mlat 分箱账（全任务都没定上位？）")
        return
    print("   %9s %10s %12s %12s %12s" % ("|磁纬| °", "曝光 s", "速率 c/s", "frac_ev_in3", "r₃ 个/s"))
    ks = sorted(acc)
    fr, ml, wt = [], [], []
    for k in ks:
        sec, nev, nin3, ncl = acc[k]
        if sec <= 0 or nev <= 0:
            continue
        f = nin3 / nev
        fr.append(f)
        ml.append(0.5 * (k[0] + k[1]))
        wt.append(nev)
        print("   %4.0f–%-4.0f %10.0f %12.0f %12.3e %12.3f"
              % (k[0], k[1], sec, nev / sec, f, ncl / sec))
    fr, ml, wt = np.array(fr), np.array(ml), np.array(wt)
    lo = ml < 33.0
    if lo.any() and (~lo).any():
        f_lo = np.average(fr[lo], weights=wt[lo])
        f_hi = np.average(fr[~lo], weights=wt[~lo])
        print()
        print("   |磁纬| < 33°（A 角所在）：frac_ev_in3 = %.3e" % f_lo)
        print("   |磁纬| ≥ 33°（B 角所在）：frac_ev_in3 = %.3e" % f_hi)
        print("   高/低 = **%.2f 倍**" % (f_hi / f_lo if f_lo > 0 else float("inf")))
        r = np.corrcoef(ml, fr)[0, 1]
        print("   分箱曲线与 |磁纬| 的相关系数 r = %.3f" % r)
        print()
        # 判据必须看**符号**，不能只看 |r|。与辐射带同向（高磁纬更高）才是"辐射带的影子"；
        # 反向（低磁纬更高）是另一回事，而且对 TGF 更危险——A 角 TGF 全在低磁纬。
        ratio = f_hi / f_lo if f_lo > 0 else float("inf")
        if abs(r) < 0.5 and 0.5 < ratio < 2.0:
            print("   ⇒ **与磁纬不强相关**：这道门挡的是环境（那一次过境的读出/粒子状态），")
            print("     不是位置。可以当判据用，且不等于偷偷加了一道磁纬门。")
        elif ratio > 1.0:
            print("   ⇒ **与磁纬同向强相关（高磁纬更高）**：这道门在很大程度上是辐射带的影子。")
            print("     必须说清它挡的是环境不是位置，否则等于偷偷加了一道磁纬门。")
        else:
            print("   ⇒ **与磁纬反向强相关（低磁纬更高，r < 0）**：**这不是辐射带的影子**")
            print("     ——辐射带在高磁纬，而这个量在那里反而最低。但它在低磁纬最高，")
            print("     **而 A 角 TGF 恰恰全在 |磁纬| < 33°**，所以这道门是往信号区收的，")
            print("     比「辐射带门」更需要说清楚。要判它能不能用，看两件事：")
            print("     (1) 分布在阈值附近是不是有真正的谷（双峰），不是把尾巴切一刀；")
            print("     (2) 真值候选离阈值的余量，以及「真值全过」的偶然概率 (1-q)^n。")
    print()
    print("   注：分箱账只含定上位的秒数；位置来源逐过境记在 trip_*.csv 的 pos_src 列。")
    src = collections.Counter(r["pos_src"] for r in ok)
    print("   位置来源：" + "，".join("%s %d 次" % kv for kv in sorted(src.items())))

    print()
    print("=== 5. 顺带：未决项 6 的死时间样本排除了哪一档速率 ===")
    if os.path.isdir(root):
        dt_coverage(ok, root)
    else:
        print("   拿不到归档，跳过")


if __name__ == "__main__":
    main()
