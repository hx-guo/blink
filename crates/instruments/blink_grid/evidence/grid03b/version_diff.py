"""两个版本的搜索产物逐候选对账：谁回流了、谁被砍了、真值动没动。

判据一改就要回答三件事，而且要**逐候选**回答，不能只比总数：
  1. 7 个有闪电认证的 TGF 全部存活，且 `count` / `fa` **逐位不变**——少一个立刻停；
  2. 新旧两版各自独有的候选是什么形态（四路同戳？谱硬？低磁纬？）；
  3. 跳过这道门的卫星（共帧的 02/04/07）候选数**只能按旧版被这道门砍掉的数目增加**，
     多一个都是有别的东西混进来了。

对键用 `(instrument, start[:23])`：两版的 `start` 都写全精度，截到毫秒两边都能对上，
而毫秒内两个候选同时出现的情形在实测里没有（真出现会打印警告）。

用法：
    python3 version_diff.py --old <data_v10_dir> --new <data_dir> [--sat GRID-03B]
        [--lightning <burst_events>/index.csv]
"""

import argparse
import collections
import csv
import glob
import json
import os


def load(root, sat):
    out = {}
    dup = 0
    for p in sorted(glob.glob(os.path.join(root, sat, "*", "*", "*_signals.json"))):
        for c in json.load(open(p)):
            k = c["start"][:23]
            if k in out:
                dup += 1
            out[k] = c
    return out, dup


def metrics(root, sat):
    tot = collections.Counter()
    days = 0
    for p in sorted(glob.glob(os.path.join(root, sat, "*", "*", "*_hours.json"))):
        days += 1
        for h in json.load(open(p))["hours"]:
            for k, v in (h.get("metrics") or {}).items():
                try:
                    tot[k] += int(v)
                except (TypeError, ValueError):
                    pass
    return tot, days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sat", nargs="*", default=["GRID-02", "GRID-03B", "GRID-04", "GRID-07"])
    ap.add_argument("--sig-fa", type=float, default=1e-5)
    ap.add_argument("--lightning", help="burst_events/index.csv，取 7 个闪电认证的时刻")
    args = ap.parse_args()

    lit = set()
    if args.lightning:
        for r in csv.DictReader(open(args.lightning)):
            if r["lightning"] == "1":
                lit.add(r["start"][:23])

    for sat in args.sat:
        o, odup = load(args.old, sat)
        n, ndup = load(args.new, sat)
        om, od = metrics(args.old, sat)
        nm, nd = metrics(args.new, sat)
        if od != nd:
            print("!! %s 天数不同：旧 %d 新 %d —— 新版没跑完就不要对账" % (sat, od, nd))
        if odup or ndup:
            print("!! %s 毫秒键碰撞：旧 %d 新 %d" % (sat, odup, ndup))
        osig = {k for k, c in o.items() if c["false_positive_per_year"] <= args.sig_fa}
        nsig = {k for k, c in n.items() if c["false_positive_per_year"] <= args.sig_fa}
        print("== %s  天 %d/%d  候选 %d → %d（%+d）  显著 %d → %d（%+d）"
              % (sat, od, nd, len(o), len(n), len(n) - len(o),
                 len(osig), len(nsig), len(nsig) - len(osig)))
        print("   dropped_simultaneous %d → %d；dropped_dead_gap %d → %d；"
              "dropped_high_rate %d → %d；dropped_single_detector %d → %d"
              % (om["dropped_simultaneous"], nm["dropped_simultaneous"],
                 om["dropped_dead_gap"], nm["dropped_dead_gap"],
                 om["dropped_high_rate"], nm["dropped_high_rate"],
                 om["dropped_single_detector"], nm["dropped_single_detector"]))
        gone, came = set(o) - set(n), set(n) - set(o)
        print("   旧有新无 %d 个（其中显著 %d）；新有旧无 %d 个（其中显著 %d）"
              % (len(gone), len(gone & osig), len(came), len(came & nsig)))
        exp = om["dropped_simultaneous"] - nm["dropped_simultaneous"]
        print("   对账：这道门少砍了 %d 个 ⇒ 候选数应当 %+d，实测 %+d %s"
              % (exp, exp, len(n) - len(o), "✓" if exp == len(n) - len(o) else "✗ 不等，查"))

        same = set(o) & set(n)
        moved = [k for k in same
                 if o[k]["count"] != n[k]["count"]
                 or o[k]["false_positive_per_year"] != n[k]["false_positive_per_year"]]
        print("   两版都在的 %d 个里，count 或 fa 变了的 %d 个 %s"
              % (len(same), len(moved), "✓ 逐位不变" if not moved else "✗"))
        for k in moved[:5]:
            print("      %s count %s→%s fa %.3e→%.3e"
                  % (k, o[k]["count"], n[k]["count"],
                     o[k]["false_positive_per_year"], n[k]["false_positive_per_year"]))

        if lit and sat == "GRID-03B":
            miss = [k for k in lit if k not in n]
            print("   **7 个闪电认证：新版在 %d / %d**%s"
                  % (len(lit) - len(miss), len(lit),
                     "" if not miss else "  ✗✗ 少了：" + ", ".join(miss)))
            for k in sorted(lit):
                if k in o and k in n:
                    ok = (o[k]["count"] == n[k]["count"]
                          and o[k]["false_positive_per_year"] == n[k]["false_positive_per_year"])
                    print("      %s count %s fa %.3e %s"
                          % (k, n[k]["count"], n[k]["false_positive_per_year"],
                             "逐位不变" if ok else "✗ 变了"))

        if came & nsig:
            print("   新进来的显著候选：")
            for k in sorted(came & nsig):
                c = n[k]
                pos = c.get("position") or {}
                print("      %s count %-3d fa %.2e 窗 %.1f µs lat %s"
                      % (k, c["count"], c["false_positive_per_year"],
                         c["bin_size_best"] * 1e6,
                         "%.1f" % pos["latitude"] if pos else "-"))
        if gone & osig:
            print("   被砍掉的显著候选：")
            for k in sorted(gone & osig):
                c = o[k]
                print("      %s count %-3d fa %.2e 窗 %.1f µs"
                      % (k, c["count"], c["false_positive_per_year"], c["bin_size_best"] * 1e6))
        print()


if __name__ == "__main__":
    main()
