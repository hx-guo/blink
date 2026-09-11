#!/usr/bin/env python3
"""逐路高压 + 机箱状态字：直接看"某个机箱是不是关着的"。

HE-HV/HE_HV_PHODet 给 18 路相里探测器的高压（1 Hz），
HE-InsStat/HE_Status 给 PDAU A/B/C 主备份的状态字（1 Hz）。
这两个量跟事例流完全独立，是机箱开关机的独立证据，顺带独立验证机箱划分。

用法: hv_probe.py <YYYY-MM-DDTHH> [中心 UTC 秒串]
"""
import os, sys
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.io import fits

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
K1 = "/hxmt/work/HXMT-DATA/1K"
LAUNCH = datetime(2017, 6, 15, tzinfo=timezone.utc)


def pick(folder_path, prefix):
    if not os.path.isdir(folder_path):
        return None
    best, bestv = None, -1
    for name in os.listdir(folder_path):
        if name.startswith(prefix):
            try:
                v = int(name[len(prefix):len(prefix) + 1])
            except ValueError:
                v = 0
            if v > bestv:
                best, bestv = name, v
    return os.path.join(folder_path, best) if best else None


hour = datetime.strptime(sys.argv[1], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
num = (hour.replace(hour=0) - LAUNCH).days + 1
f = "%s/Y%04d%02d/%04d%02d%02d-%04d" % (K1, hour.year, hour.month,
                                        hour.year, hour.month, hour.day, num)
stem = "HXMT_%04d%02d%02dT%02d_" % (hour.year, hour.month, hour.day, hour.hour)
hv = pick(f, stem + "HE-HV_FFFFFF_V")
st = pick(f, stem + "HE-InsStat_FFFFFF_V")
print(hv)
print(st)

with fits.open(hv) as h:
    d = h["HE_HV_PHODet"].data
    t = np.array(d["Time"], float)
    v = np.stack([np.array(d["HV_PHODet_%d" % i], float) for i in range(18)], 1)

print("\n=== 18 路高压统计（这一小时 %d 个采样）===" % len(t))
print("  路  中位   最小   最大   低于中位一半的秒数")
for i in range(18):
    med = np.median(v[:, i])
    low = (v[:, i] < 0.5 * max(med, 1e-9)).sum()
    print("  %2d %7.1f %6.1f %7.1f   %5d   %s" % (i, med, v[:, i].min(), v[:, i].max(), low, "ABC"[i // 6]))

# 按机箱看有没有整箱一起掉
box = np.stack([v[:, 0:6].mean(1), v[:, 6:12].mean(1), v[:, 12:18].mean(1)], 1)
ref = np.median(box, 0)
off = box < 0.5 * np.maximum(ref, 1e-9)
print("\n  整箱高压掉到中位一半以下的秒数: A %d, B %d, C %d （共 %d 秒）"
      % (off[:, 0].sum(), off[:, 1].sum(), off[:, 2].sum(), len(t)))

with fits.open(st) as h:
    s = h["HE_Status"].data
    ts = np.array(s["Time"], float)
    cols = [c.name for c in h["HE_Status"].columns if c.name != "Time"]
    vals = {c: np.array(s[c]) for c in cols}
print("\n=== PDAU 状态字取值分布 ===")
for c in cols:
    u, n = np.unique(vals[c], return_counts=True)
    print("  %-14s %s" % (c, ", ".join("%s:%d" % (a, b) for a, b in zip(u, n))))

if len(sys.argv) > 2:
    ctr = datetime.strptime(sys.argv[2][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    m = (ctr - MET_EPOCH).total_seconds()
    k = np.argmin(np.abs(t - m))
    print("\n=== %s 前后 ±8 s 逐秒 18 路高压 ===" % sys.argv[2])
    for j in range(max(0, k - 8), min(len(t), k + 9)):
        tag = " <<<" if j == k else ""
        print("  %+3d s  A %s | B %s | C %s%s"
              % (t[j] - m,
                 " ".join("%5.0f" % x for x in v[j, 0:6]),
                 " ".join("%5.0f" % x for x in v[j, 6:12]),
                 " ".join("%5.0f" % x for x in v[j, 12:18]), tag))
    ks = np.argmin(np.abs(ts - m))
    print("  状态字附近:")
    for j in range(max(0, ks - 3), min(len(ts), ks + 4)):
        print("    %+3d s  %s" % (ts[j] - m, "  ".join("%s=%s" % (c, vals[c][j]) for c in cols)))
