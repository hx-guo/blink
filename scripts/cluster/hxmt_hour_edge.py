#!/usr/bin/env python3
"""小时文件边界上的候选堆积：搜索按整小时分块，块首/块尾是不是在造候选。"""
import os, csv, collections
import numpy as np

D = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/'
cat = list(csv.DictReader(open(D + 'catalog_v6.csv')))


def off(s):
    """距最近整点的秒数（0..1800）。"""
    x = int(s[14:16]) * 60 + float(s[17:26])
    return min(x, 3600.0 - x)


o = np.array([off(c['start']) for c in cat])
n = len(cat)
print("目录 %d 行，距整点的秒数分布（均匀期望 = 每 1 s 宽的窗 %.2f 行）" % (n, 2 * n / 3600))
print("  %-14s %8s %10s %8s" % ("窗", "实测", "均匀期望", "倍数"))
for lo, hi in ((0, 0.1), (0, 1), (1, 10), (10, 60), (60, 300), (300, 900), (900, 1800)):
    k = ((o >= lo) & (o < hi)).sum()
    e = 2 * n * (hi - lo) / 3600
    print("  %-14s %8d %10.1f %8.1f" % ("%g–%g s" % (lo, hi), k, e, k / e if e else float('nan')))

edge = [c for c in cat if off(c['start']) < 1.0]
print("\n整点 ±1 s 内的 %d 行，按天:" % len(edge))
for d, k in collections.Counter(c['date'] for c in edge).most_common():
    print("   %s  %d" % (d, k))
print("  其中闪电关联 %d 个，tier: %s"
      % (sum(c['associated'] == '1' for c in edge),
         dict(collections.Counter(c['tier'] for c in edge))))

# 同一批候选换个参照：距最近整分的秒数（整分不是分块边界，应当是均匀的）
def offmin(s):
    x = float(s[17:26])
    return min(x, 60.0 - x)


om = np.array([offmin(c['start']) for c in cat])
k = (om < 1.0).sum()
print("\n对照：距整分 ±1 s 内 %d 行（均匀期望 %.1f，倍数 %.2f）—— 整分不是分块边界"
      % (k, 2 * n / 60, k / (2 * n / 60)))
