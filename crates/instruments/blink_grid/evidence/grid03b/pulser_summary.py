"""汇总 `pulser_scan.py`：这批过境里多少是脉冲、**不是脉冲的那些长什么样**。

回答四件事：

  1. **梳齿占比在"超阈"与"速率配平对照"上分不分得开**——没有对照就分不清"判据有分辨力"
     和"判据到处都报正"；
  2. **超阈过境里多少是脉冲**，阈值落在谷里还是尾上；
  3. **不是脉冲的那些长什么样**——若有一批不是脉冲，**那才是真正需要判据的那一类**；
  4. **episode 的时长与位置**。时长要先分清口径：`on_start_s`/`on_stop_s` 是梳齿上**首尾**
     簇的跨度，**不是连续段时长**。用占空比 `实际梳齿簇数 ÷ (跨度/周期)` 判：≈1 是一段连续，
     ≪1 说明中间有停顿、跨度里混着间隙。**不先分清这一条，就会拿首尾跨度去比别人的单段时长。**

位置用来检验"脉冲由地面站触发"：脉冲锁在固定 UTC 小时 ⇒ 落在地面站可见弧段内 ⇒ 星下点
应聚成团。**UTC 小时比经纬度更直接**——地面站固定在地球上，小时锁死就说明是地面触发。

用法：
    python3 pulser_summary.py <pulserscan 目录> [--nchunk 4] [--trip <tripscan 目录>]
                              [--comb 0.5] [--sat GRID-03B]
"""

import argparse
import collections
import csv
import datetime as dt
import glob
import os

import numpy as np

TICK = 2.0**22
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)


def _f(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def assert_complete(d, nchunk):
    dones = sorted(glob.glob(os.path.join(d, "done_*.txt")))
    if len(dones) != nchunk:
        raise SystemExit("done 文件 %d 份，应有 %d 份 —— 没跑完，不汇总" % (len(dones), nchunk))
    exp = got = 0
    for p in dones:
        kv = dict(line.strip().split(",", 1) for line in open(p) if "," in line)
        exp += int(kv["expected"])
        got += int(kv["done"])
    print("完成检查：%d 份 done，Σexpected = %d，Σdone = %d" % (len(dones), exp, got))
    if got < exp:
        print("！%d 次过境没出结果（读不出），占 %.1f%%" % (exp - got, 100.0 * (exp - got) / exp))
    return exp


def q(a, name):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if a.size == 0:
        return "%s：无" % name
    return ("%s：n=%d 中位 %.4g，四分位 %.4g–%.4g，全距 %.4g–%.4g"
            % (name, a.size, np.median(a), np.percentile(a, 25), np.percentile(a, 75),
               a.min(), a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--nchunk", type=int, default=4)
    ap.add_argument("--trip", help="tripscan 目录，用来取 gti_start 换算 UTC 与位置")
    ap.add_argument("--comb", type=float, default=0.5, help="判脉冲的梳齿占比阈")
    ap.add_argument("--sat", default="GRID-03B")
    args = ap.parse_args()

    assert_complete(args.dir, args.nchunk)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "pulser_*.csv"))):
        rows += list(csv.DictReader(open(f)))
    ok = [r for r in rows if r["comb_frac"]]
    err = [r for r in rows if (r["note"] or "").startswith("ERR")]
    print("行 %d：可用 %d，读不出 %d" % (len(rows), len(ok), len(err)))
    for r in err[:5]:
        print("   读不出 %s %s" % (r["pass_file"], r["note"]))

    over = [r for r in ok if r["tag"] == "over"]
    ctrl = [r for r in ok if r["tag"] == "control"]
    cf_o = np.array([_f(r["comb_frac"]) for r in over])
    cf_c = np.array([_f(r["comb_frac"]) for r in ctrl])

    print()
    print("=== 1. 梳齿占比：超阈 vs 速率配平的对照 ===")
    print("   " + q(cf_o, "超阈过境"))
    print("   " + q(cf_c, "对照过境"))
    print()
    print("   %10s %10s %10s" % ("梳齿占比", "超阈", "对照"))
    for a, b in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        print("   %4.2f–%-5.2f %10d %10d"
              % (a, b, int(((cf_o >= a) & (cf_o < b)).sum()), int(((cf_c >= a) & (cf_c < b)).sum())))
    if cf_c.size:
        print("   对照里梳齿占比 > %.2f 的：%d / %d = %.2f%%（这是这道判据的假阳率）"
              % (args.comb, int((cf_c > args.comb).sum()), cf_c.size,
                 100.0 * (cf_c > args.comb).mean()))

    puls = [r for r in over if _f(r["comb_frac"]) > args.comb]
    rest = [r for r in over if _f(r["comb_frac"]) <= args.comb]
    print()
    print("=== 2. 超阈过境里多少是脉冲（梳齿占比 > %.2f）===" % args.comb)
    print("   脉冲 %d / %d = **%.1f%%**，不是脉冲 %d"
          % (len(puls), len(over), 100.0 * len(puls) / max(len(over), 1), len(rest)))
    if puls:
        print("   " + q([_f(r["period_hz"]) for r in puls], "脉冲频率 Hz"))
        print("   " + q([_f(r["resid_ns"]) for r in puls], "回归残差 ns"))

    print()
    print("=== 3. **不是脉冲的那些长什么样**（这才是真正需要判据的一类）===")
    if not rest:
        print("   **一个都没有** —— 超阈过境全是脉冲过境。")
        print("   ⇒ 这条线可以按「把脉冲过境当非观测时段剔除」收尾，**不需要新判据**。")
    else:
        print("   %d 次。逐次：" % len(rest))
        print("   %-12s %-34s %9s %9s %9s %8s"
              % ("day", "pass_file", "frac_in3", "comb_frac", "rate", "n_quads"))
        for r in sorted(rest, key=lambda r: -_f(r["frac_ev_in3"]))[:25]:
            print("   %-12s %-34s %9.4f %9.4f %9.0f %8s"
                  % (r["day"], r["pass_file"][:34], _f(r["frac_ev_in3"]),
                     _f(r["comb_frac"]), _f(r["rate_cps"]), r["n_quads"]))
        print("   " + q([_f(r["frac_ev_in3"]) for r in rest], "它们的 frac_ev_in3"))
        print("   " + q([_f(r["rate_cps"]) for r in rest], "它们的速率 c/s"))

    print()
    print("=== 4. episode 的时长：先分清口径 ===")
    span = np.array([_f(r["on_stop_s"]) - _f(r["on_start_s"]) for r in puls])
    per = np.array([_f(r["period_tick"]) for r in puls])
    nq = np.array([_f(r["n_quads"]) * _f(r["comb_frac"]) for r in puls])
    duty = nq / np.maximum(span * TICK / np.maximum(per, 1e-9), 1e-9)
    print("   " + q(span, "首尾跨度 s"))
    print("   " + q(duty, "占空比 = 实际梳齿簇数 ÷ (跨度/周期)"))
    print()
    print("   读法：占空比 ≈ 1 ⇒ 跨度就是一段连续的 episode，可以直接跟别人的单段时长比；")
    print("         占空比 ≪ 1 ⇒ 跨度里混着停顿，**首尾跨度不能当单段时长**。")
    if duty.size:
        med = np.median(duty[np.isfinite(duty)])
        if med > 0.8:
            print("   ⇒ 实测中位 %.2f，**是一段连续的**，跨度可直接比。" % med)
        else:
            print("   ⇒ 实测中位 %.2f，**不是一段连续的**，比之前必须先拆段。" % med)

    if args.trip:
        gti = {}
        for f in sorted(glob.glob(os.path.join(args.trip, "trip_*.csv"))):
            for r in csv.DictReader(open(f)):
                if r.get("gti_start"):
                    gti[r["pass_file"]] = _f(r["gti_start"])
        hrs = []
        for r in puls:
            g = gti.get(r["pass_file"])
            if g is None or not np.isfinite(_f(r["on_start_s"])):
                continue
            t = REF + dt.timedelta(seconds=g + _f(r["on_start_s"]))
            hrs.append(t.hour + t.minute / 60.0)
        if hrs:
            print()
            print("=== 5. 脉冲的 UTC 小时（地面站假说：锁死 ⇒ 地面触发）===")
            h = np.array(hrs)
            cnt = np.bincount(h.astype(int), minlength=24)
            for i in range(24):
                if cnt[i]:
                    print("   %02dh : %4d %s" % (i, cnt[i], "#" * min(int(cnt[i] / max(cnt.max(), 1) * 50), 50)))
            R = abs(np.mean(np.exp(2j * np.pi * h / 24.0)))
            print("   圆集中度 R = %.4f（n = %d）；R → 1 表示锁在一个小时上" % (R, h.size))
            print("   瑞利 p ≈ exp(-n R²) = %.3e" % np.exp(-h.size * R * R))
            if R > 0.5:
                print("   ⇒ **锁死在固定 UTC 小时 ⇒ 地面指令触发**，磁纬只是轨道几何的副产品。")
            else:
                print("   ⇒ 没有锁在固定小时，地面站假说在 03B 上不成立，要另找解释。")


if __name__ == "__main__":
    main()
