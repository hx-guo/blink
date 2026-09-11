"""GBM 全量跑的体检：列表天数 / 真正出了曝光的天数 / 被排除的小时数。

用法：python3 gbm_health.py <days.txt[,days2.txt]> <产出根目录> [年份 ...]

**每一次全量跑完都先跑这个，三个数对不上就停下来查，别直接做统计。**
这条规矩是两次踩坑换来的，两次都是同一类缺陷：镜像的目录布局有五种，
`day_dirs()` 只试了其中三种，落在另两种里的天光有 TTE、找不到 poshist，
每小时都被 `missing_data` 排除。搜索照跑不误、退出码为零、日志里没有一行
错误——**只有"列表里有这一天"和"这一天真的出了搜索秒数"这两个数对不上,
才看得见它**。

  2014-12 那次：账面 352 天，真正有曝光的 331 天，速率算高 6%。
  2015-05..07 那次：92 天，抓到时还没跑。

除了三个主数，还打印几个逐小时诊断量的分布（`n_detectors` 说明这一段是
2 路 BGO 还是 14 路全编制——完备性口径直接依赖它；`time_reversals` 是
GRM 上出过的整段重写；`saa_seconds` 与 GTI 缺口互校）。
"""

import collections
import glob
import json
import os
import sys


def main(days_file, root, years):
    listed = []
    for path in days_file.split(","):
        listed += [line.strip() for line in open(path) if line.strip()]
    listed = sorted(set(listed))
    if years:
        listed = [d for d in listed if d[:4] in years]
    listed_set = set(listed)

    produced = {}
    pattern = os.path.join(root, "Fermi_GBM", "*", "*", "*_hours.json")
    for path in sorted(glob.glob(pattern)):
        day = json.load(open(path))
        if years and day["date"][:4] not in years:
            continue
        produced[day["date"]] = day

    exposed, zero_exposure, no_output = [], [], []
    total_seconds = 0.0
    excluded = collections.Counter()
    detectors = collections.Counter()
    reversals = 0
    signals = 0
    searched_hours = 0
    for date in listed:
        day = produced.get(date)
        if day is None:
            no_output.append(date)
            continue
        seconds = day.get("searched_seconds", 0.0)
        total_seconds += seconds
        signals += day.get("n_signals", 0)
        searched_hours += day.get("searched_hours", 0)
        for reason, n in day.get("excluded_by_reason", {}).items():
            excluded[reason] += n
        for hour in day.get("hours", []):
            metrics = hour.get("metrics", {})
            if hour.get("status") == "searched":
                detectors[int(metrics.get("n_detectors", 0))] += 1
                reversals += int(metrics.get("time_reversals", 0))
        (exposed if seconds > 0 else zero_exposure).append(date)

    extra = sorted(set(produced) - listed_set)

    print(f"days.txt: {days_file}   产出: {root}   年份: {', '.join(years) or '全部'}")
    print()
    print(f"  列表天数            {len(listed)}")
    print(f"  有产出的天数        {len(listed) - len(no_output)}")
    print(f"  真正有曝光的天数    {len(exposed)}")
    print(f"  活时间              {total_seconds:.4e} s = {total_seconds / 86400:.1f} 天")
    print(f"  搜索小时            {searched_hours}")
    print(f"  候选                {signals}")
    print()
    print(f"  被排除的小时        {sum(excluded.values())}")
    for reason, n in excluded.most_common():
        print(f"      {reason:20s} {n}")
    print()
    print("  逐小时 n_detectors 分布（完备性口径直接靠它：2 = 只有 BGO，14 = 全编制）")
    for n, hours in sorted(detectors.items()):
        print(f"      {n:2d} 路   {hours} 小时")
    print(f"  time_reversals 合计 {reversals}")

    bad = False
    if no_output:
        bad = True
        print(f"\n  ⚠ 列表里有、却没有产出的 {len(no_output)} 天：{no_output[:10]}")
    if zero_exposure:
        bad = True
        print(f"\n  ⚠ 有产出、曝光却是 0 的 {len(zero_exposure)} 天：{zero_exposure[:10]}")
        print("    这正是 2014-12 与 2015-05..07 那一类：多半是 poshist 找不到。")
        print("    查法：gbm_layout_scan.sh 看镜像布局，对齐 io/file.rs 的 day_dirs()。")
    if extra:
        bad = True
        print(f"\n  ⚠ 有产出、却不在列表里的 {len(extra)} 天：{extra[:10]}")
    if not bad:
        print("\n  三个数一致，可以往下做统计。")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
