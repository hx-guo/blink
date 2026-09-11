"""GECAM-C：闪电关联结果的第一遍看。

`blink wwlln` 是富集不是筛选，所以问法是"关联上的候选在各判据量上怎么分布"，
不是"切完还剩几个关联"。本底对照用 coincidence_probability（偶然关联概率）。
"""

import collections
import json
import sys

import numpy as np

rows = json.load(open(sys.argv[1]))
print("候选 %d" % len(rows))
print("字段:", sorted(rows[0].keys()))
sig = rows[0].get("signal") or {}
if sig:
    print("signal 字段:", sorted(sig.keys()))

assoc = np.array([bool(r.get("associated")) for r in rows])
prob = np.array([r.get("coincidence_probability") if r.get("coincidence_probability") is not None else np.nan
                 for r in rows], float)
inside = np.array([r.get("inside_coverage", True) is not False for r in rows])
train = np.array([bool((r.get("train") or {}).get("is_train")) for r in rows])
nb = np.array([(r.get("train") or {}).get("neighbors_10min", np.nan) for r in rows], float)


def get(r, *path, default=np.nan):
    cur = r
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


start = [get(r, "signal", "start", default=get(r, "start", default="")) for r in rows]
fa = np.array([get(r, "signal", "false_positive_per_year",
                   default=get(r, "false_positive_per_year")) for r in rows], float)

print("\n覆盖内 %d（%.1f%%）；关联上的 %d（%.2f%%，覆盖内 %.2f%%）"
      % (inside.sum(), inside.mean() * 100, assoc.sum(), assoc.mean() * 100,
         assoc[inside].mean() * 100 if inside.any() else np.nan))
print("被标成列车成员的 %d（%.1f%%）；邻居数中位 %.0f" % (train.sum(), train.mean() * 100, np.nanmedian(nb)))
finite = np.isfinite(prob)
print("偶然关联概率 coincidence_probability：中位 %.4f，均值 %.4f（n=%d）"
      % (np.nanmedian(prob[finite]), np.nanmean(prob[finite]), finite.sum()))
print("→ 期望的偶然关联数 %.1f，实测关联数 %d，比 %.2f"
      % (np.nansum(prob[finite]), assoc.sum(), assoc.sum() / max(np.nansum(prob[finite]), 1e-9)))

print("\n按显著性分层（只看覆盖内）")
for thr in (np.inf, 1, 0.1, 0.01, 1e-3, 1e-5, 7e-7):
    m = inside & (fa <= thr)
    if m.sum() < 1:
        continue
    exp = np.nansum(prob[m & finite])
    print("  fa<=%-8g n=%6d 关联 %4d (%.2f%%)  偶然期望 %6.1f  比 %5.2f"
          % (thr, m.sum(), assoc[m].sum(), assoc[m].mean() * 100, exp,
             assoc[m].sum() / max(exp, 1e-9)))

print("\n关联上的候选：是不是列车成员")
if assoc.any():
    print("  关联样本里 is_train 占 %.1f%%（全样本 %.1f%%）"
          % (train[assoc].mean() * 100, train.mean() * 100))
    print("  关联样本的邻居数中位 %.0f（全样本 %.0f）" % (np.nanmedian(nb[assoc]), np.nanmedian(nb)))

print("\n最显著的关联候选（fa 最小的 25 个）")
idx = np.flatnonzero(assoc & inside)
idx = idx[np.argsort(fa[idx])][:25]
for i in idx:
    print("  %-26s fa=%.3e  邻居 %4.0f  is_train=%s  P偶然=%.4f"
          % (start[i], fa[i], nb[i], bool(train[i]), prob[i]))

TARGETS = [t for t in sys.argv[2:]]
if TARGETS:
    print("\n=== 指定候选 ===")
    for t in TARGETS:
        hits = [i for i, s in enumerate(start) if s.startswith(t)]
        if not hits:
            print("  %s 没找到" % t)
            continue
        for i in hits:
            print("  %-26s fa=%.3e  associated=%s  P偶然=%s  邻居 %.0f  is_train=%s  覆盖内=%s"
                  % (start[i], fa[i], bool(assoc[i]), prob[i], nb[i], bool(train[i]), bool(inside[i])))
            lt = rows[i].get("lightnings") or rows[i].get("strokes")
            if lt:
                print("     闪电:", json.dumps(lt, ensure_ascii=False)[:600])

print("\n逐日关联数")
by_day = collections.Counter()
day_tot = collections.Counter()
for s, a in zip(start, assoc):
    day_tot[s[:10]] += 1
    if a:
        by_day[s[:10]] += 1
for d in sorted(day_tot):
    print("  %s  候选 %5d  关联 %3d (%.2f%%)" % (d, day_tot[d], by_day[d], by_day[d] / day_tot[d] * 100))
