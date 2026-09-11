#!/usr/bin/env python3
"""陆/海拆分的对照：本底在两类里也必须是平的。

陆海分类是按星下点做的，而星下点又决定经度、经度又进 LST 的定义。
若曝光在陆上和海上的 LST 覆盖本来就不同，拆出来的形状差就可能是曝光造的。
用留出的 fa > 1 本底在同一分类下跑一遍，平不平一看便知。
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5
R = 6371.0


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def classify(lat, lon, ring=600.0):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    d = np.degrees(ring / R)
    la = [lat]; lo = [lon180]
    for az in range(0, 360, 45):
        a = np.radians(az)
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)) + 180) % 360) - 180)
    f = globe.is_land(np.array(la), np.array(lo))
    return 'land' if f.all() else ('ocean' if not f.any() else 'coast')


rows = []
with open(sys.argv[1]) as f:
    r = csv.reader(f); next(r)
    for i, (start, fpy, lon, lat, assoc, nb, train) in enumerate(r):
        if float(fpy) > 1 and train == '0' and i % 12 == 0:      # 抽 1/12，够统计又不慢
            rows.append((lst_hours(start, float(lon)), float(lat), float(lon)))
print("本底抽样 %d 个" % len(rows))
lst = np.array([x[0] for x in rows])
cls = np.array([classify(x[1], x[2]) for x in rows])
for k in ('land', 'ocean', 'coast'):
    x = lst[cls == k]
    if len(x) < 100:
        continue
    h = np.histogram(x, bins=NBIN, range=(0, 24))[0].astype(float)
    p = 100 * h / h.sum()
    A = (h.max() - h.min()) / 2 / h.mean()
    chi2 = ((h - h.mean()) ** 2 / h.mean()).sum()
    print("  本底 %-6s N=%6d  %s" % (k, len(x), " ".join("%5.1f" % v for v in p)))
    print("  %-11s         幅度 A = %.3f，对均匀的 chi2 = %.1f (dof=7)" % ("", A, chi2))
print("\n信号侧拆开的幅度是 陆 1.05 / 海 0.22；本底侧若都远小于这个，拆分就不是曝光造的。")
