#!/usr/bin/env python3
"""三条统筹点名要的：
  A 正对照：拿真值已知 f=1 的人群去拟，看模板能不能把 1 拟回来。
    HXMT 手上唯一的"认证 TGF"是闪电关联样本——而它正是我自己量出来带 6 h
    VLF 夜间相位偏置的那批。先把这件事量成数，说明这个正对照在 LST 这条轴上用不了。
  B 缓冲口径：gbm 的 300 km 是**经纬度平面等角** buffer（2.703 度），
    我的是**大圆** 300 km（除以 cos lat）。在 |lat| 26-43 两者差得最多，重分一次看近岸 A 还等不等。
  C 预先指定的 |lat| 26 分界（由 Fermi 倾角 25.6 定，不是调出来的）的似然比。
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0
T_FULL = np.array([201, 188, 75, 61, 197, 517, 375, 232], float)
T_NT = np.array([172, 156, 68, 57, 164, 382, 288, 195], float)
C_GBM = np.array([0.1516, 0.1638, 0.1125, 0.0648, 0.0990, 0.1510, 0.1375, 0.1198])


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def daynum(d):
    from datetime import date
    return date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal()


def secs(iso):
    return int(iso[11:13]) * 3600 + int(iso[14:16]) * 60 + float(iso[17:26])


def ring(lat, lon, km, n_az, equiangular):
    """equiangular=False 是大圆 km（经度差除 cos lat）；True 是经纬度平面等角。"""
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    d = np.degrees(km / R)
    la, lo = [], []
    for k in range(n_az):
        a = 2 * np.pi * k / n_az
        dlon = d * np.sin(a) if equiangular else d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + dlon) + 180) % 360) - 180)
    return np.array(la), np.array(lo)


def classify(lat, lon, equiangular):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    if bool(globe.is_land(lat, lon180)):
        return 'land'
    la, lo = ring(lat, lon, 300.0, 16, equiangular)
    return 'coast' if globe.is_land(la, lo).any() else 'ocean'


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


FS = np.linspace(-1.0, 2.0, 6001)


def nllc(obs, T, B):
    N = obs.sum(); Tn = T / T.sum(); Bn = B / B.sum()
    M = N * (np.outer(FS, Tn) + np.outer(1 - FS, Bn))
    return -(obs * np.log(np.maximum(M, 1e-12)) - M).sum(axis=1)


def fitf(obs, T, B):
    if obs.sum() < 10:
        return (float('nan'),) * 3
    n = nllc(obs, T, B); i = n.argmin(); ok = FS[n <= n.min() + 1.92]
    return FS[i], ok.min(), ok.max()


cat = list(csv.DictReader(open(sys.argv[1])))
lat_a = np.array([float(c['latitude']) for c in cat]); lon_a = np.array([float(c['longitude']) for c in cat])
lst_a = np.array([lst_hours(c['start'], float(c['longitude'])) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])
t_a = np.array([daynum(c['date']) * 86400.0 + secs(c['start']) for c in cat])
o = np.argsort(t_a); ts = t_a[o]
nbr = np.zeros(len(cat), int)
nbr[o] = (np.searchsorted(ts, ts + 600.0, 'right') - np.searchsorted(ts, ts - 600.0, 'left')) - 1

cls_g = np.array([classify(a, b, False) for a, b in zip(lat_a, lon_a)])   # 大圆
cls_e = np.array([classify(a, b, True) for a, b in zip(lat_a, lon_a)])    # 等角

bg = []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, ass, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 6 == 0:
            bg.append((lst_hours(start, float(blon_)), float(blat_), float(blon_)))
blst = np.array([x[0] for x in bg]); blat = np.array([x[1] for x in bg]); blon = np.array([x[2] for x in bg])
bcls_g = np.array([classify(x[1], x[2], False) for x in bg])
bcls_e = np.array([classify(x[1], x[2], True) for x in bg])
B_land = hist(blst[bcls_g == 'land'])

print("=== A 正对照：唯一的认证 TGF 人群（闪电关联）在 LST 轴上能不能当真值 1 ===")
for tag, m in (("关联 陆地 |lat|<26", assoc & (cls_g == 'land') & (np.abs(lat_a) < 26)),
               ("未关联 陆地 |lat|<26", (~assoc) & (cls_g == 'land') & (np.abs(lat_a) < 26))):
    h = hist(lst_a[m])
    v, lo, hi = fitf(h, T_FULL, B_land)
    pk = (np.argmax(h) * 3 + 1.5)
    print("  %-22s N=%-5d 自身 A=%.3f 峰格 %4.1f h  拟 GBM 陆地模板 f = %.3f [%.3f, %.3f]"
          % (tag, int(m.sum()), amp(h), pk, v, lo, hi))
print("  => 关联样本的峰在凌晨、未关联的峰在傍晚，两者差约 6 h（第 15 条）。")
print("     所以它拟出来的低 f 量的是 WWLLN 夜间探测效率，不是纯度。")
print("     **HXMT 在 LST 这条轴上没有可用的 f=1 正对照。**")

print("")
print("=== B 300 km 缓冲：大圆 vs 等角，重分后近岸 A 还等不等 ===")
sig = (~assoc)
for nm, cc, bb in (("大圆 300 km", cls_g, bcls_g), ("等角 2.703 度", cls_e, bcls_e)):
    row = []
    for k in ('land', 'coast', 'ocean'):
        h = hist(lst_a[sig & (cc == k)])
        row.append("%s N=%d A=%.3f" % (k, int(h.sum()), amp(h)))
    print("  %-16s %s" % (nm, " | ".join(row)))
chg = int((cls_g[sig] != cls_e[sig]).sum())
print("  两种判法给出不同类的候选 %d/%d = %.1f%%" % (chg, int(sig.sum()), 100 * chg / sig.sum()))
for a, b in ((0, 26), (26, 44)):
    mm = sig & (np.abs(lat_a) >= a) & (np.abs(lat_a) < b)
    print("    |lat| %2d-%2d: 改判 %d/%d = %.1f%%；近岸 A 大圆 %.3f 等角 %.3f"
          % (a, b, int((cls_g[mm] != cls_e[mm]).sum()), int(mm.sum()),
             100 * (cls_g[mm] != cls_e[mm]).mean(),
             amp(hist(lst_a[mm & (cls_g == 'coast')])), amp(hist(lst_a[mm & (cls_e == 'coast')]))))

print("")
print("=== C 预先指定的 |lat| 26 分界（Fermi 倾角 25.6 定的，不是调出来的）===")
keep = sig & (cls_g == 'land') & (nbr == 0)
bk = bcls_g == 'land'
mlo = keep & (np.abs(lat_a) < 26); mhi = keep & (np.abs(lat_a) >= 26)
Blo = hist(blst[bk & (np.abs(blat) < 26)]); Bhi = hist(blst[bk & (np.abs(blat) >= 26)])
n1 = nllc(hist(lst_a[mlo]), T_NT, Blo); n2 = nllc(hist(lst_a[mhi]), T_NT, Bhi)
d = 2 * ((n1 + n2).min() - n1.min() - n2.min())
v1 = fitf(hist(lst_a[mlo]), T_NT, Blo); v2 = fitf(hist(lst_a[mhi]), T_NT, Bhi)
print("  口径对齐（两边都摘目录内 ±10 min 有邻居）+ gbm 摘串模板：")
print("    |lat| < 26  N=%-5d f = %.3f [%.3f, %.3f]" % (int(mlo.sum()), *v1))
print("    |lat| >= 26 N=%-5d f = %.3f [%.3f, %.3f]" % (int(mhi.sum()), *v2))
print("    两者相等的似然比 Delta = %.2f，1 自由度 => p = %.2e" % (d, np.exp(-d / 2)))
