"""细梯 7 天：新旧二进制逐小时比 `hours.json` 的状态与活时间。

`old` 是当前 main 的二进制（只认 470 道那把梯子），`new` 是两把梯子那版。
两边跑的是同一批天、同一份数据，**逐小时的状态差别只能归到这一处改动**。

**`new` 那一半不另跑，直接用 71 天生产跑的产物**——同一个二进制
（`blink_ladder`）、同一批天，另起一路只是重复烧机时。生产跑的天单覆盖
2022-08-03 .. 10-15，7 天里的 6 天在内；2022-10-16（全 470 道的对照日）
单独提一路补上。

用法: cmp_fine.py [old 目录] [new 目录]
"""

import collections
import json
import pathlib
import sys

OLD = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "/scratchfs2/gecam/guohx/gecamrun_fine/old/data")
NEW = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   "/scratchfs2/gecam/guohx/finerun/data")
TASKS = pathlib.Path("/scratchfs2/gecam/guohx/gecamrun_fine/tasks.txt")


def hours(variant, day):
    root = OLD if variant == "old" else NEW
    path = root / "GECAM-C" / day[:4] / day[5:7] / f"{day.replace('-', '')}_hours.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    rows = raw if isinstance(raw, list) else raw["hours"]
    return {int(r["hour"]): r for r in rows}


def signals(variant, day):
    root = OLD if variant == "old" else NEW
    path = root / "GECAM-C" / day[:4] / day[5:7] / f"{day.replace('-', '')}_signals.json"
    return json.load(open(path)) if path.exists() else []


days = [line.strip() for line in open(TASKS) if line.strip()]
total = collections.Counter()
print("日期         旧 searched/活时间(s)      新 searched/活时间(s)     多搜的小时")
for day in days:
    a, b = hours("old", day), hours("new", day)
    if a is None or b is None:
        print(f"{day}  产物未齐（old={a is not None} new={b is not None}）")
        continue
    gained = sorted(h for h in b if b[h]["status"] == "searched" and a.get(h, {}).get("status") != "searched")
    lost = sorted(h for h in a if a[h]["status"] == "searched" and b.get(h, {}).get("status") != "searched")
    sa = sum(1 for r in a.values() if r["status"] == "searched")
    sb = sum(1 for r in b.values() if r["status"] == "searched")
    ta = sum(r["searched_seconds"] for r in a.values())
    tb = sum(r["searched_seconds"] for r in b.values())
    total["old_hours"] += sa
    total["new_hours"] += sb
    total["old_seconds"] += ta
    total["new_seconds"] += tb
    total["gained"] += len(gained)
    total["lost"] += len(lost)
    note = f"+{gained}" if gained else ""
    if lost:
        note += f"  ⚠ 少搜 {lost}"
    print(f"{day}   {sa:2d}/24  {ta:9.1f}        {sb:2d}/24  {tb:9.1f}     {note}")
    # 旧版也搜过的小时，候选数必须逐小时相同
    sig_a = collections.Counter(s["start"][11:13] for s in signals("old", day))
    sig_b = collections.Counter(s["start"][11:13] for s in signals("new", day))
    shared = [h for h in a if a[h]["status"] == "searched" and b.get(h, {}).get("status") == "searched"]
    bad = [h for h in shared if sig_a[f"{h:02d}"] != sig_b[f"{h:02d}"]]
    if bad:
        print(f"    ⚠ 两边都搜过但候选数不同的小时：{sorted(bad)}")
        for h in sorted(bad):
            print(f"        {h:02d}h  旧 {sig_a[f'{h:02d}']}  新 {sig_b[f'{h:02d}']}")
    else:
        print(f"    两边都搜过的 {len(shared)} 个小时候选数逐小时相同")

print()
print(f"合计：旧 searched {total['old_hours']} 小时 / {total['old_seconds']:.1f} s")
print(f"      新 searched {total['new_hours']} 小时 / {total['new_seconds']:.1f} s")
print(f"      多搜 {total['gained']} 小时，少搜 {total['lost']} 小时"
      f"（少搜必须是 0，不是 0 就是修坏了）")
