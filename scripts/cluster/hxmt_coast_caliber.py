#!/usr/bin/env python3
"""我的"近岸"判法可能漏数：只在**半径正好 300 km 的圆周上**取 16 个点，
而"离陆 < 300 km"要的是**整个圆盘**里有没有陆。150 km 处的小岛会被整条漏掉。

对照 gbm 的本底构成（|lat| <= 26，亚阈 600 个）：陆 21.8 / 近岸 24.8 / 远洋 53.3，
我的同段本底是 22.5 / 14.3 / 63.2 —— **陆地几乎相同、近岸差了 10 个百分点**，
正是"圆周 vs 圆盘"会造成的方向。这里把圆盘版做出来比一比。
"""
import csv, sys
import numpy as np
from global_land_mask import globe

R = 6371.0
NBIN = 8


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def disk(lat, lon, km, radii, n_az):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    la, lo = [], []
    cs = max(np.cos(np.radians(lat)), 0.2)
    for rr in radii:
        d = np.degrees(rr * km / R)
        for k in range(n_az):
            a = 2 * np.pi * k / n_az
            la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
            lo.append((((lon180 + d * np.sin(a) / cs) + 180) % 360) - 180)
    return np.array(la), np.array(lo)


def make(radii, n_az):
    def f(lat, lon):
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        if bool(globe.is_land(lat, lon180)):
            return 'land'
        la, lo = disk(lat, lon, 300.0, radii, n_az)
        return 'coast' if globe.is_land(la, lo).any() else 'ocean'
    return f


RING = make([1.0], 16)                      # 原来的：只有圆周
DISK = make([0.25, 0.5, 0.75, 1.0], 24)     # 圆盘：4 圈 x 24 方位 = 96 点


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


cat = list(csv.DictReader(open(sys.argv[1])))
lat = np.array([float(c['latitude']) for c in cat]); lon = np.array([float(c['longitude']) for c in cat])
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])

bl_lst, bl_lat, bl_lon = [], [], []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, ass, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 24 == 0:
            bl_lst.append(lst_hours(start, float(blon_))); bl_lat.append(float(blat_)); bl_lon.append(float(blon_))
bl_lst = np.array(bl_lst); bl_lat = np.array(bl_lat); bl_lon = np.array(bl_lon)

for nm, fn in (("圆周 16 点（原来）", RING), ("圆盘 96 点（修正）", DISK)):
    c_sig = np.array([fn(a, b) for a, b in zip(lat, lon)])
    c_bg = np.array([fn(a, b) for a, b in zip(bl_lat, bl_lon)])
    print("=== %s ===" % nm)
    for tag, m, cc, ll in (("显著未关联", ~assoc, c_sig, lst), ("闪电关联", assoc, c_sig, lst),
                           ("fa>1 本底", np.ones(len(bl_lst), bool), c_bg, bl_lst)):
        sub = cc[m] if m.dtype == bool and len(m) == len(cc) else cc
        p = [np.mean(sub == k) for k in ('land', 'coast', 'ocean')]
        A = [amp(hist(ll[m][sub == k])) for k in ('land', 'coast', 'ocean')]
        print("  %-12s n=%-6d 陆 %.1f%% 近岸 %.1f%% 远洋 %.1f%%   A 陆 %.3f 近岸 %.3f 远洋 %.3f"
              % (tag, int(m.sum()), 100 * p[0], 100 * p[1], 100 * p[2], *A))
    lo26 = np.abs(bl_lat) < 26
    p = [np.mean(c_bg[lo26] == k) for k in ('land', 'coast', 'ocean')]
    print("  本底 |lat|<26  陆 %.1f%% 近岸 %.1f%% 远洋 %.1f%%   （gbm 亚阈 600：21.8 / 24.8 / 53.3）"
          % (100 * p[0], 100 * p[1], 100 * p[2]))
    print("")
