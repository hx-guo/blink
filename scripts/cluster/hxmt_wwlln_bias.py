#!/usr/bin/env python3
"""把 WWLLN 的两种偏置拆开：6 h 相位差里有多少其实是"关联样本偏海上"造成的？

远洋 TGF 本来就在后半夜峰（GBM 远洋模板峰 3.20 LT），陆地 TGF 在傍晚峰（18.88 LT）。
关联样本偏海上 ⇒ 它的整体 LST 峰自然往夜里跑，**不需要任何"夜间探测效率"**。
所以"关联 1.5 h vs 未关联 16.5 h"这 6 小时必须**逐类**再看一遍：
  类内差没了 ⇒ 6 h 是构成效应，夜间偏置这条要改写；
  类内差还在 ⇒ 两种偏置都真，可以各自定量。
"""
import csv, sys
import numpy as np
from global_land_mask import globe

R = 6371.0
NBIN = 8


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


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


def circ(lst):
    th = 2 * np.pi * np.asarray(lst) / 24.0
    n = len(th)
    c, s = np.cos(th).mean(), np.sin(th).mean()
    r = np.hypot(c, s)
    ph = (np.arctan2(s, c) % (2 * np.pi)) * 24 / (2 * np.pi)
    # 相位的标准误（von Mises 近似）
    se = (1.0 / (r * np.sqrt(n))) * 24 / (2 * np.pi) if r > 0 else float('nan')
    return n, 2 * r, ph, se, np.exp(-n * r * r)


cat = list(csv.DictReader(open(sys.argv[1])))
lat = np.array([float(c['latitude']) for c in cat]); lon = np.array([float(c['longitude']) for c in cat])
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])
cls = np.array([classify(a, b) for a, b in zip(lat, lon)])

print("=== 整体（复现原来那条 6 h）===")
for tag, m in (("闪电关联", assoc), ("显著未关联", ~assoc)):
    n, a, ph, se, p = circ(lst[m])
    h = np.histogram(lst[m], bins=NBIN, range=(0, 24))[0]
    print("  %-10s N=%-5d 一次谐波幅 %.3f  峰相 %5.2f ± %.2f h  峰格 %4.1f h"
          % (tag, n, a, ph, se, np.argmax(h) * 3 + 1.5))

print("")
print("=== 逐类再看一遍 ===")
for k in ('land', 'coast', 'ocean'):
    row = []
    for tag, m in (("关联", assoc & (cls == k)), ("未关联", ~assoc & (cls == k))):
        n, a, ph, se, p = circ(lst[m])
        row.append((tag, n, a, ph, se))
    d = (row[0][3] - row[1][3] + 12) % 24 - 12
    sd = np.hypot(row[0][4], row[1][4])
    print("  %-6s 关联 N=%-5d A=%.3f 峰 %5.2f ± %.2f h | 未关联 N=%-5d A=%.3f 峰 %5.2f ± %.2f h"
          " | 类内差 %+5.2f ± %.2f h（%.1f sigma）"
          % (k, row[0][1], row[0][2], row[0][3], row[0][4],
             row[1][1], row[1][2], row[1][3], row[1][4], d, sd, abs(d) / sd))

print("")
print("=== 构成效应能解释多少：把关联样本按未关联的类构成重加权 ===")
wc = {}
for k in ('land', 'coast', 'ocean'):
    na = (assoc & (cls == k)).sum(); nu = ((~assoc) & (cls == k)).sum()
    wc[k] = (nu / (~assoc).sum()) / (na / assoc.sum())
w = np.array([wc[c] for c in cls])
m = assoc
th = 2 * np.pi * lst[m] / 24.0
ww = w[m]
c_, s_ = (ww * np.cos(th)).sum() / ww.sum(), (ww * np.sin(th)).sum() / ww.sum()
ph = (np.arctan2(s_, c_) % (2 * np.pi)) * 24 / (2 * np.pi)
print("  关联样本原始峰相 %.2f h -> 按未关联类构成重加权后 %.2f h" % (circ(lst[assoc])[2], ph))
print("  未关联样本峰相 %.2f h" % circ(lst[~assoc])[2])
print("  权重 %s" % {k: round(v, 3) for k, v in wc.items()})

print("")
print("=== WWLLN 相对关联效率（海/陆），用纯度做单向界 ===")
na_l = int((assoc & (cls == 'land')).sum()); na_o = int((assoc & (cls == 'ocean')).sum())
nu_l = int(((~assoc) & (cls == 'land')).sum()); nu_o = int(((~assoc) & (cls == 'ocean')).sum())
f_land = 0.941
print("  关联 陆 %d / 远洋 %d，比 %.3f" % (na_l, na_o, na_o / na_l))
print("  未关联 陆 %d（纯度 %.3f -> TGF %.0f）/ 远洋 %d（纯度 <= 1 -> TGF <= %d）"
      % (nu_l, f_land, nu_l * f_land, nu_o, nu_o))
print("  => 远洋/陆地 的真 TGF 比 <= %.3f，相对关联效率 >= %.2f"
      % (nu_o / (nu_l * f_land), (na_o / na_l) / (nu_o / (nu_l * f_land))))
mis = 0.081
pl = (na_l / (na_l + na_o + (assoc & (cls == 'coast')).sum()))
print("  再扣关联样本自身 8.1%% 的误关联（按本底构成 24.6%% 陆 / 63.0%% 远洋）后，下界降到 %.2f"
      % (((0.370 - mis * 0.630) / (0.300 - mis * 0.246)) / (nu_o / (nu_l * f_land))))
