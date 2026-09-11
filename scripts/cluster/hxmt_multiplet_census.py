#!/usr/bin/env python3
"""多脉冲重复计数对整本目录的影响（不只是那 3467 个未关联的）。

这关系到论文里的行数口径：目录报的是 5421 个"TGF"，如果其中一部分是
同一个物理事件的多个脉冲，那"TGF 个数"和"搜索窗个数"就不是一回事。
"""
import csv, sys, collections
import numpy as np


def daynum(d):
    from datetime import date
    return date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal()


def secs(iso):
    return int(iso[11:13]) * 3600 + int(iso[14:16]) * 60 + float(iso[17:26])


cat = list(csv.DictReader(open(sys.argv[1])))
t = np.array([daynum(c['date']) * 86400.0 + secs(c['start']) for c in cat])
lat = np.array([float(c['latitude']) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])
tier = np.array([c['tier'] for c in cat])
year = np.array([c['date'][:4] for c in cat])
o = np.argsort(t)

print("目录 %d 行，%d 关联 / %d 未关联；direct %d / lightning %d"
      % (len(cat), int(assoc.sum()), int((~assoc).sum()),
         int((tier == 'direct').sum()), int((tier == 'lightning').sum())))

print("")
print("按合并尺度扫一遍（同一天、间隔 <= T、纬度差 <= 0.5 度 算同一物理事件）：")
print("  %-8s %-10s %-10s %-12s %s" % ("T", "物理事件数", "多出的行", "多出比例", "最大组"))
for T in (0.01, 0.05, 0.2, 1.0, 5.0, 60.0, 600.0):
    gid = np.empty(len(cat), int); g = -1
    for j, idx in enumerate(o):
        if j == 0 or t[idx] - t[o[j - 1]] > T or abs(lat[idx] - lat[o[j - 1]]) > 0.5:
            g += 1
        gid[idx] = g
    size = np.bincount(gid)
    print("  %-8s %-10d %-10d %-12s %d"
          % ("%g s" % T, size.size, len(cat) - size.size,
             "%.2f%%" % (100 * (len(cat) - size.size) / len(cat)), size.max()))

T = 1.0
gid = np.empty(len(cat), int); g = -1
for j, idx in enumerate(o):
    if j == 0 or t[idx] - t[o[j - 1]] > T or abs(lat[idx] - lat[o[j - 1]]) > 0.5:
        g += 1
    gid[idx] = g
size = np.bincount(gid)
mult = size[gid]
print("")
print("取 T = 1 s：%d 行 -> %d 个物理事件（少 %.2f%%）" % (len(cat), size.size, 100 * (len(cat) - size.size) / len(cat)))
print("  组大小分布 %s" % dict(sorted(collections.Counter(size).items())))
for nm, m in (("关联", assoc), ("未关联", ~assoc), ("direct 档", tier == 'direct'), ("lightning 档", tier == 'lightning')):
    print("  %-12s N=%-5d 在多脉冲组里 %-4d（%.1f%%）" % (nm, int(m.sum()), int((mult[m] > 1).sum()), 100 * (mult[m] > 1).mean()))
print("  逐年多脉冲行占比：")
for y in sorted(set(year)):
    m = year == y
    print("    %s  N=%-5d %.1f%%" % (y, int(m.sum()), 100 * (mult[m] > 1).mean()))
big = [i for i in range(size.size) if size[i] >= 4]
print("  >= 4 个脉冲的组 %d 个：" % len(big))
for gg in big:
    k = gid == gg
    tt = np.sort(t[k])
    print("    %s  %d 个，跨度 %.3f s，lat %.1f lon %.1f，关联 %d，tier %s"
          % (cat[int(np.where(k)[0][0])]['date'], int(k.sum()), tt[-1] - tt[0],
             lat[k][0], float(cat[int(np.where(k)[0][0])]['longitude']),
             int(assoc[k].sum()), cat[int(np.where(k)[0][0])]['tier']))
