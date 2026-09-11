#!/usr/bin/env python3
"""把供给全队的模板按"一个物理事件一行"重导一次。

第 22 条量到 3.9% 的行属于 100 ms 内的多脉冲组，同一个物理事件最多被数 7 次。
供出去的模板是**计数直方**，重复计数会按组大小给某些 LST 格加权，所以要去重。
去重判据：同一天、时间间隔 <= 1 s、纬度差 <= 0.5 度；每组留最显著的一行。
"""
import csv, sys, collections
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def daynum(d):
    from datetime import date
    return date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal()


def secs(iso):
    return int(iso[11:13]) * 3600 + int(iso[14:16]) * 60 + float(iso[17:26])


def classify(lat, lon):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    if bool(globe.is_land(lat, lon180)):
        return 'land'
    d = np.degrees(300.0 / R)
    la, lo = [], []
    for k in range(16):
        a = 2 * np.pi * k / 16
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)) + 180) % 360) - 180)
    return 'coast' if globe.is_land(np.array(la), np.array(lo)).any() else 'ocean'


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


cat = list(csv.DictReader(open(sys.argv[1])))
lat = np.array([float(c['latitude']) for c in cat]); lon = np.array([float(c['longitude']) for c in cat])
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])
fa = np.array([float(c['false_positive_per_year']) for c in cat])
t = np.array([daynum(c['date']) * 86400.0 + secs(c['start']) for c in cat])

o = np.argsort(t)
gid = np.empty(len(cat), int); g = -1
for j, idx in enumerate(o):
    if j == 0 or t[idx] - t[o[j - 1]] > 1.0 or abs(lat[idx] - lat[o[j - 1]]) > 0.5:
        g += 1
    gid[idx] = g
keep = np.zeros(len(cat), bool)
for gg, members in collections.defaultdict(list, {}).items():
    pass
byg = collections.defaultdict(list)
for i, gg in enumerate(gid):
    byg[gg].append(i)
for gg, mem in byg.items():
    keep[min(mem, key=lambda i: fa[i])] = True   # 每组留最显著的一行

cls = np.array([classify(a, b) for a, b in zip(lat, lon)])
sig = ~assoc
print("目录 %d 行 -> %d 个物理事件（去重丢掉 %d 行，%.2f%%）"
      % (len(cat), int(keep.sum()), len(cat) - int(keep.sum()), 100 * (1 - keep.mean())))
print("未关联：%d 行 -> %d 个物理事件" % (int(sig.sum()), int((sig & keep).sum())))
print("")
print("=== 供给全队的模板：去重前 vs 去重后（GBM 单点口径）===")
for k in ('land', 'coast', 'ocean'):
    for lab, m in (("去重前", sig & (cls == k)), ("去重后", sig & keep & (cls == k))):
        h = hist(lst[m])
        print("  %-6s %-7s N=%-5d A=%.3f  计数 %s"
              % (k, lab, int(h.sum()), amp(h), " ".join("%4d" % v for v in h.astype(int))))
        if lab == "去重后":
            print("  %-14s            归一 %s" % ("", " ".join("%.4f" % v for v in h / h.sum())))
print("")
h0 = hist(lst[sig]); h1 = hist(lst[sig & keep])
print("  全体 去重前 N=%d A=%.3f 归一 %s" % (h0.sum(), amp(h0), " ".join("%.4f" % v for v in h0 / h0.sum())))
print("  全体 去重后 N=%d A=%.3f 归一 %s" % (h1.sum(), amp(h1), " ".join("%.4f" % v for v in h1 / h1.sum())))
print("  归一值最大改动 %.4f" % np.abs(h0 / h0.sum() - h1 / h1.sum()).max())
