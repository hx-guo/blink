"""896 道纪元生产跑的收尾断言：**产出小时数 == 应有小时数，不等就非零退出**。

**"队列空 ≠ 跑完"。** 轮询到队列空就往下走，会在残缺输入上跑完整条汇总链——
产物齐全、格式正确、数字看着合理、不报错。所以断言看的是**产出**不是队列，
而且不等非零退出，**不等搜索的返回码**。

农场现在有抢占，被抢的那一天会整天重来（搜索是按天写产物的），所以这个脚本
还要把**"应有而未出"的天单独列出来**，让"反复提交直到齐"有个准确的重提清单
——否则会把"被抢反复重来"误当成"在推进"。

用法: gc_fine_manifest.py <finerun 目录> [tasks_small.txt tasks_big.txt ...]
"""

import datetime as dt
import glob
import json
import os
import sys

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"


def expected_hours(day):
    """归档里这一天有几个小时有 GRD 文件（逐小时取最大版本，与搜索看到的一致）。"""
    hours = set()
    for path in glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/gcg_evt_*.fits"):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) >= 5:
            hours.add(parts[3][:2])
    return hours


def main():
    run = sys.argv[1]
    lists = sys.argv[2:] or [f"{run}/tasks_small.txt", f"{run}/tasks_big.txt"]
    days = []
    for path in lists:
        days += [line.strip() for line in open(path) if line.strip()]
    days = sorted(set(days))

    want = 0
    got = 0
    missing_days = []
    short_days = []
    live = 0.0
    signals = 0
    for day in days:
        want_hours = expected_hours(day)
        want += len(want_hours)
        stem = f"{run}/data/GECAM-C/{day[:4]}/{day[5:7]}/{day.replace('-', '')}"
        try:
            rows = json.load(open(f"{stem}_hours.json"))
            rows = rows if isinstance(rows, list) else rows["hours"]
            sig = json.load(open(f"{stem}_signals.json"))
        except Exception:
            missing_days.append(day)
            continue
        # 归档里有文件的那些小时，产物里必须有对应的一行（`excluded` 也算有）
        have = {f"{r['hour']:02d}" for r in rows}
        covered = want_hours & have
        got += len(covered)
        if len(covered) != len(want_hours):
            short_days.append((day, sorted(want_hours - have)))
        live += sum(r["searched_seconds"] for r in rows)
        signals += len(sig)

    print(f"天 {len(days)}：应有小时 {want}，产出小时 {got}")
    print(f"活时间 {live:.1f} s = {live / 3600:.2f} 小时；候选 {signals}")
    if missing_days:
        print(f"完全没有产物的天 {len(missing_days)}：{' '.join(missing_days)}")
    for day, hours in short_days:
        print(f"缺小时的天 {day}：{' '.join(hours)}")
    if got != want:
        print(f"\n断言失败：产出 {got} != 应有 {want}，**不要往下跑汇总**")
        raise SystemExit(1)
    print("\n断言通过：逐天逐小时齐全")


if __name__ == "__main__":
    main()
