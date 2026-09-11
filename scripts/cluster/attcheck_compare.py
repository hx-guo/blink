"""针对性重跑 vs 权威目录：逐天按起始时间（±1 ms）配对，找只在新结果里出现且没有 attitude 字段的候选。"""
import json, glob, os, sys
from datetime import datetime, timezone
# 小数秒不能交给 datetime：它只到微秒，把 9 位纳秒串截到 6 位会把时刻前移
# 最多 1 µs。HXMT 的候选窗端点就是事例时刻本身（实测端点只有 0.96% 恰好落在
# 整微秒上，平均前移 775 ns），截断会直接丢掉边界那个事例。整数秒交给
# strptime，小数部分单独按 float 加回来。
def t(iso):
    b = iso.rstrip("Z"); h, _, f = b.partition(".")
    stamp = datetime.strptime(h, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return stamp.timestamp() + (float("0." + f) if f else 0.0)
for tag, new_root, old_root, sub in (("HXMT", "/scratchfs2/gecam/guohx/v6run/attcheck/data", "/scratchfs2/gecam/guohx/v6run/data", "Insight-HXMT_HE"), ("SVOM", "/scratchfs2/gecam/guohx/svomrun5/attcheck/data", "/scratchfs2/gecam/guohx/svomrun5/data", "SVOM_GRM")):
    days = [l.strip() for l in open(os.path.dirname(new_root) + "/days.txt") if l.strip()]
    done = 0; n_new = n_old = 0; only_new = []; only_old = []; no_att = 0; without = 0; noeph_new = 0; single = 0
    for d in days:
        y, m, dd = d.split("-"); rel = f"{sub}/{y}/{m}/{y}{m}{dd}"
        fn, fo = f"{new_root}/{rel}_signals.json", f"{old_root}/{rel}_signals.json"
        if not os.path.exists(fn): continue
        done += 1
        new = json.load(open(fn)); old = json.load(open(fo)) if os.path.exists(fo) else []
        for x in json.load(open(f"{new_root}/{rel}_hours.json"))["hours"]:
            mt = x.get("metrics") or {}; without += int(mt.get("without_attitude", 0)); noeph_new += int(mt.get("dropped_no_ephemeris", 0)); single += int(mt.get("dropped_single_detector", 0))
        n_new += len(new); n_old += len(old)
        to = [t(c["start"]) for c in old]; tn = [t(c["start"]) for c in new]
        for c, tc in zip(new, tn):
            if not any(abs(tc - x) < 1e-3 for x in to):
                only_new.append((d, c["start"][:23], c["false_positive_per_year"], "attitude" in c))
                if "attitude" not in c: no_att += 1
        for c, tc in zip(old, to):
            if not any(abs(tc - x) < 1e-3 for x in tn): only_old.append((d, c["start"][:23], c["false_positive_per_year"]))
    print(f"=== {tag}: 重跑 {done}/{len(days)} 天；旧 {n_old} 个候选，新 {n_new} 个；新结果 dropped_no_ephemeris {noeph_new}、without_attitude {without}、dropped_single_detector {single}")
    print(f"  只在新结果里: {len(only_new)}（其中无 attitude 字段 {no_att}）；只在旧结果里: {len(only_old)}")
    for r in only_new[:12]: print("   新:", r)
    for r in only_old[:6]: print("   旧:", r)
