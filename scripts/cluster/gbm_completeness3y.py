"""三年（2014 + 2015 + 2016H1）对已发表目录的完备性，**分亮度档**。

为什么要分档：**完备性高只证明亮端不漏**。已发表目录本身是触发式/亮样本偏置的，
给一个总数会把"我们在暗端漏多少"这个唯一要紧的问题盖掉。样本大到能分档，就必须分。

**口径（沿用第 11 条，不要用别的）**：分母**不是**"全年目录条目"，
而是"**落在我们真正出了曝光的天上的**条目"——落在没曝光的天上的 TGF 不该算漏。
"有曝光"按 `*_hours.json` 的 `searched_seconds > 0` 判，不是"天表里列了这一天"。

用法: python3 gbm_completeness3y.py <run_data_dir> <catalog_dir> [out.csv]
"""
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# 2001 年之后的闰秒（UTC 日期）；MET = (UTC − 2001-01-01) + 已过闰秒数
LEAPS = ("2006-01-01", "2009-01-01", "2012-07-01", "2015-07-01", "2017-01-01")
REF = datetime(2001, 1, 1, tzinfo=timezone.utc)
TOL_S = 0.01           # 目录给时刻、我们给候选窗起点，10 ms 足够宽
SIG = 1e-5
YEARS = ("2014", "2015", "2016")


def met_of(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    t = datetime.strptime(h + "." + (f + "000000")[:6],
                          "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    n = sum(1 for l in LEAPS if t >= datetime.fromisoformat(l).replace(tzinfo=timezone.utc))
    return (t - REF).total_seconds() + n


def table(rows, found, fa, key, edges, label, fmt="%4.0f"):
    print(f"  {label}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        if not m.sum():
            continue
        hi_s = "inf" if hi > 1e8 else fmt % hi
        print(f"    {fmt % lo:>7s}–{hi_s:<7s} n={m.sum():5d}   找到 {100 * found[m].mean():5.1f}%"
              f"   达显著 {100 * (fa[m] <= SIG).mean():5.1f}%")


def main(run_dir, cat_dir, out=None):
    # 1) 真正有曝光的天
    exposed, seconds, hours = set(), 0.0, 0
    for path in sorted(glob.glob(os.path.join(run_dir, "Fermi_GBM", "*", "*", "*_hours.json"))):
        day = json.load(open(path))
        if day["date"][:4] not in YEARS:
            continue
        if day.get("searched_seconds", 0.0) > 0:
            exposed.add(day["date"])
            seconds += day["searched_seconds"]
            hours += day.get("searched_hours", 0)
    print(f"有曝光的天 {len(exposed)}，活时间 {seconds:.4e} s = {seconds / 86400:.1f} 天，"
          f"搜索小时 {hours}")

    # 2) 目录条目，按日期落在有曝光的天上筛（口径 B/C）
    cat, dropped = [], 0
    for r in csv.reader(open(os.path.join(cat_dir, "gbm_tgf_catalog_offline.csv"))):
        if not r or r[0].startswith("#"):
            continue
        date = r[6].strip()
        if date[:4] not in YEARS:
            continue
        if date not in exposed:
            dropped += 1
            continue
        cat.append({"id": r[0].strip(), "met": float(r[1]), "date": date,
                    "counts": float(r[3]) + float(r[4]) + float(r[5]),
                    "width_ms": float(r[8]), "trig": r[14].strip()})
    wwlln = {r[0].strip() for r in
             csv.reader(open(os.path.join(cat_dir, "gbm_tgf_catalog_wwlln.csv")))
             if r and not r[0].startswith("#")}
    print(f"目录三年条目：落在有曝光的天上 {len(cat)}（分母），"
          f"落在无曝光的天上 {dropped}（不计入，不算漏）")

    # 3) 我们的候选
    ours = []
    for f in sorted(glob.glob(os.path.join(run_dir, "Fermi_GBM", "*", "*", "*_signals.json"))):
        if os.path.basename(f)[:4] not in YEARS:
            continue
        for s in json.load(open(f)):
            ours.append((met_of(s["start"]), s["false_positive_per_year"], s["count"]))
    ours.sort()
    om = np.array([o[0] for o in ours])
    print(f"我们的候选 {len(ours)} 个，显著（fa ≤ {SIG:g}）"
          f"{sum(1 for o in ours if o[1] <= SIG)} 个\n")

    # 4) 逐条匹配
    rows = []
    for c in cat:
        i = np.searchsorted(om, c["met"])
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(om) and abs(om[j] - c["met"]) <= TOL_S:
                if best is None or ours[j][1] < ours[best][1]:
                    best = j
        rows.append({"id": c["id"], "date": c["date"], "met": c["met"],
                     "counts": c["counts"], "width_ms": c["width_ms"],
                     "wwlln": int(c["id"] in wwlln),
                     "triggered": int(c["trig"] not in ("", "NULL")),
                     "found": int(best is not None),
                     "our_fa": ours[best][1] if best is not None else "",
                     "our_count": ours[best][2] if best is not None else ""})
    found = np.array([r["found"] for r in rows], bool)
    fa = np.array([r["our_fa"] if r["our_fa"] != "" else np.inf for r in rows], float)
    counts = np.array([r["counts"] for r in rows])
    width = np.array([r["width_ms"] for r in rows])

    print(f"=== 三年合计：目录 {len(rows)} 个里找到 {found.sum()}（{100 * found.mean():.1f}%），"
          f"达显著 {int((fa <= SIG).sum())}（{100 * (fa <= SIG).mean():.1f}%）===\n")

    print("=== 分亮度档（目录计数 = BGO_0 + BGO_1 + NAI）===")
    print("  **这才是要害**：完备性高只证明亮端不漏。")
    table(rows, found, fa, counts, [0, 30, 50, 75, 100, 150, 200, 1e9], "按目录计数：")
    print()
    table(rows, found, fa, width, [0, 0.1, 0.2, 0.5, 1.0, 1e9],
          "按目录 discovery bin 宽度（ms，注意它不是测出来的时长）：", fmt="%.3g")
    print()

    print("  按年（同一口径）：")
    yr = np.array([r["date"][:4] for r in rows])
    for y in YEARS:
        m = yr == y
        if m.sum():
            print(f"    {y}  n={m.sum():5d}   找到 {100 * found[m].mean():5.1f}%"
                  f"   达显著 {100 * (fa[m] <= SIG).mean():5.1f}%")
    print()
    ww = np.array([r["wwlln"] for r in rows], bool)
    tg = np.array([r["triggered"] for r in rows], bool)
    print(f"  带 WWLLN 关联 n={ww.sum():5d}   找到 {100 * found[ww].mean():5.1f}%"
          f"   达显著 {100 * (fa[ww] <= SIG).mean():5.1f}%")
    print(f"  星上触发     n={tg.sum():5d}   找到 {100 * found[tg].mean():5.1f}%"
          f"   达显著 {100 * (fa[tg] <= SIG).mean():5.1f}%")
    print()

    # 5) 反向：我们有而目录没有
    cm = np.array(sorted(c["met"] for c in cat))
    extra = extra_sig = 0
    for met, f_, _c in ours:
        i = np.searchsorted(cm, met)
        if not any(0 <= j < len(cm) and abs(cm[j] - met) <= TOL_S for j in (i - 1, i, i + 1)):
            extra += 1
            if f_ <= SIG:
                extra_sig += 1
    print(f"=== 我们有而目录没有：{extra} 个候选，其中显著 {extra_sig} 个 ===")
    print("  这些要另判（第 11 条那套：闪电关联 + 时间平移对照 + 亚阈负对照）。")

    if out:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
