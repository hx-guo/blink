"""GECAM-C 30 天重搜的**应有账**：事前把每一天归档里到底有几个可搜小时数死，
事后拿它硬断言。

起因是一个不报错的失效模式：链条轮询到队列空就往下走，于是在残缺输入上跑完了汇总、
特征、关联——产物齐全、格式正确、数字看着合理。**"队列空"不是"跑完"。**

可搜的硬条件与 `from_epoch` 一致：GRD + CPD + POSATT 三件齐全（C 星 CPD 缺失非致命，
但仍记下来）。另外这一轮实测到 **2022-08-03 .. 2022-10-15 的 1,121 小时用的是 896 道
能量梯**，现有二进制整小时拒收，所以那些小时不计进"应有"——它们是**已知的口径缺口**，
不是失败。

用法:
  gc_manifest.py build <tasks.txt> <manifest.json>     生成应有账
  gc_manifest.py check <manifest.json> <data 根目录>   对账，不齐就非零退出
"""

import glob
import json
import os
import sys

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
# 896 道能量梯的纪元（实测 gc_ladder_2022.csv）：现有二进制整小时拒收
LEGACY_LADDER_FIRST = "2022-08-03"
LEGACY_LADDER_LAST = "2022-10-15"


def hours_of(day, kind):
    """该天该产品里逐小时去重后的小时号集合。版本号单独解析，不靠字典序。"""
    directory = f"{ROOT}/{day.replace('-', '/')}/{kind}"
    hours = set()
    for path in glob.glob(f"{directory}/*_v*.fits"):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) < 5 or not parts[-1].startswith("v"):
            continue
        hh = parts[3][:2]
        if hh.isdigit():
            hours.add(int(hh))
    return hours


def build(tasks_path, out_path):
    days = [d for d in open(tasks_path).read().split() if d]
    manifest = {"days": {}, "n_days_listed": len(days)}
    total_hours = 0
    n_with_data = 0
    for day in days:
        grd = hours_of(day, "GRD_EVT")
        cpd = hours_of(day, "CPD_EVT")
        pos = hours_of(day, "POSATT")
        searchable = sorted(grd & pos)
        legacy = LEGACY_LADDER_FIRST <= day <= LEGACY_LADDER_LAST
        if legacy:
            searchable = []
        manifest["days"][day] = {
            "n_grd": len(grd),
            "n_cpd": len(cpd),
            "n_posatt": len(pos),
            "expected_hours": len(searchable),
            "hours": searchable,
            "legacy_ladder": legacy,
        }
        total_hours += len(searchable)
        n_with_data += 1 if searchable else 0
        print(f"{day}  GRD {len(grd):2d}  CPD {len(cpd):2d}  POSATT {len(pos):2d}  "
              f"→ 应有 {len(searchable):2d} 小时" + ("  [896 道梯纪元]" if legacy else ""))
    manifest["expected_days_with_data"] = n_with_data
    manifest["expected_hours_total"] = total_hours
    json.dump(manifest, open(out_path, "w"), indent=1)
    print(f"\n应有：有数据的天 {n_with_data}/{len(days)}，可搜小时合计 {total_hours}")
    print(f"写出 {out_path}")


def check(manifest_path, data_root):
    manifest = json.load(open(manifest_path))
    bad = []
    got_days = got_hours = 0
    for day, want in sorted(manifest["days"].items()):
        y, m, d = day.split("-")
        hours_json = f"{data_root}/{y}/{m}/{y}{m}{d}_hours.json"
        signals_json = f"{data_root}/{y}/{m}/{y}{m}{d}_signals.json"
        if not os.path.exists(hours_json) or not os.path.exists(signals_json):
            bad.append(f"{day}: 产物缺失")
            continue
        got = json.load(open(hours_json))
        searched = got["searched_hours"]
        got_hours += searched
        if searched:
            got_days += 1
        if searched != want["expected_hours"]:
            detail = got.get("excluded_by_reason", {})
            bad.append(
                f"{day}: searched {searched} ≠ 应有 {want['expected_hours']}  排除原因 {detail}"
            )
    print(f"产出：有候选的天 {got_days}/{manifest['expected_days_with_data']}，"
          f"搜到的小时 {got_hours}/{manifest['expected_hours_total']}")
    if bad:
        print("\n对不上的天：")
        for line in bad:
            print("  " + line)
        sys.exit(1)
    print("对账通过：逐天小时数与应有账完全相等。")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
    else:
        sys.exit("用法见文件头")
