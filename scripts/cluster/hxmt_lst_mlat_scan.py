#!/usr/bin/env python3
"""口径对齐版的似然比检验 + |mlat| 阈值扫描的余量。"""
import csv, sys, collections
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0
POLE_LAT, POLE_LON = 80.65, -72.68
T = np.array([172, 156, 68, 57, 164, 382, 288, 195], float)   # gbm 摘串模板 1482


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def daynum(d):
    from datetime import date
    return date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal()


def secs(iso):
    return int(iso[11:13]) * 3600 + int(iso[14:16]) * 60 + float(iso[17:26])


def mlat(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    pa, po = np.radians(POLE_LAT), np.radians(POLE_LON)
    return np.degrees(np.arcsin(np.clip(np.sin(la) * np.sin(pa) + np.cos(la) * np.cos(pa) * np.cos(lo - po), -1, 1)))


def classify_gbm(lat, lon):
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


FS = np.linspace(-1.0, 2.0, 6001)


def nllc(obs, Tt, B):
    N = obs.sum(); Tn = Tt / Tt.sum(); Bn = B / B.sum()
    M = N * (np.outer(FS, Tn) + np.outer(1 - FS, Bn))
    return -(obs * np.log(np.maximum(M, 1e-12)) - M).sum(axis=1)


cat = list(csv.DictReader(open(sys.argv[1])))
sig = [c for c in cat if c['associated'] != '1']
lat = np.array([float(c['latitude']) for c in sig]); lon = np.array([float(c['longitude']) for c in sig])
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
ml = np.abs(mlat(lat, lon)); cls = np.array([classify_gbm(a, b) for a, b in zip(lat, lon)])
t = np.array([daynum(c['date']) * 86400.0 + secs(c['start']) for c in sig])
o = np.argsort(t); ts = t[o]
nbr = np.zeros(len(sig), int)
nbr[o] = (np.searchsorted(ts, ts + 600.0, 'right') - np.searchsorted(ts, ts - 600.0, 'left')) - 1

bg = []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, assoc, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 6 == 0:
            bg.append((lst_hours(start, float(blon_)), float(blat_), float(blon_)))
blst = np.array([x[0] for x in bg]); blat = np.array([x[1] for x in bg]); blon = np.array([x[2] for x in bg])
bml = np.abs(mlat(blat, blon)); bcls = np.array([classify_gbm(x[1], x[2]) for x in bg])

keep = (cls == 'land') & (nbr == 0)
bkeep = bcls == 'land'
print("口径对齐（两边都摘目录内 ±10 min 有邻居的）后的陆地样本 %d 个" % keep.sum())
print("")
print("=== |mlat| 阈值扫描：低半 / 高半 的 f，以及两者相等的似然比 ===")
print("  %-6s %-24s %-24s %s" % ("阈值", "低半", "高半", "似然比 p"))
for thr in (25, 30, 32, 35, 38, 40):
    mlo = keep & (ml < thr); mhi = keep & (ml >= thr)
    blo = bkeep & (bml < thr); bhi = bkeep & (bml >= thr)
    if mhi.sum() < 15:
        continue
    Blo = hist(blst[blo]); Bhi = hist(blst[bhi]) if bhi.sum() > 200 else hist(blst[bkeep])
    n1 = nllc(hist(lst[mlo]), T, Blo); n2 = nllc(hist(lst[mhi]), T, Bhi)
    d = 2 * ((n1 + n2).min() - n1.min() - n2.min())
    a1 = FS[n1.argmin()]; ok1 = FS[n1 <= n1.min() + 1.92]
    a2 = FS[n2.argmin()]; ok2 = FS[n2 <= n2.min() + 1.92]
    print("  %-6d N=%-5d f=%.3f [%.3f,%.3f]  N=%-4d f=%.3f [%.3f,%.3f]  Delta=%5.2f  p=%.1e"
          % (thr, int(mlo.sum()), a1, ok1.min(), ok1.max(),
             int(mhi.sum()), a2, ok2.min(), ok2.max(), d, np.exp(-d / 2)))

print("")
print("=== 未摘串版本（对照，看摘串把结论改了多少）===")
keep2 = cls == 'land'
for thr in (35,):
    mlo = keep2 & (ml < thr); mhi = keep2 & (ml >= thr)
    Blo = hist(blst[bkeep & (bml < thr)]); Bhi = hist(blst[bkeep & (bml >= thr)])
    n1 = nllc(hist(lst[mlo]), T, Blo); n2 = nllc(hist(lst[mhi]), T, Bhi)
    d = 2 * ((n1 + n2).min() - n1.min() - n2.min())
    print("  阈值 %d: N=%d f=%.3f / N=%d f=%.3f  Delta=%.2f p=%.1e"
          % (thr, int(mlo.sum()), FS[n1.argmin()], int(mhi.sum()), FS[n2.argmin()], d, np.exp(-d / 2)))

print("")
print("=== |mlat| >= 35 的那批是什么（摘串后 %d 个）===" % int((keep & (ml >= 35)).sum()))
k = keep & (ml >= 35)
yr = collections.Counter(c['date'][:4] for c, m in zip(sig, k) if m)
print("  年份 %s" % dict(sorted(yr.items())))
print("  |lat| 范围 %.1f–%.1f，经度 %s"
      % (np.abs(lat[k]).min(), np.abs(lat[k]).max(),
         "北美 %d / 欧亚 %d / 其它 %d" % (int(((lon[k] > 230) & (lon[k] < 300)).sum()),
                                       int(((lon[k] > 0) & (lon[k] < 150)).sum()),
                                       int((~(((lon[k] > 230) & (lon[k] < 300)) | ((lon[k] > 0) & (lon[k] < 150)))).sum()))))
print("  LST 直方 %s" % " ".join("%d" % v for v in hist(lst[k]).astype(int)))
print("  本底同段 LST 直方（归一后 ×%d）%s"
      % (int(k.sum()), " ".join("%.1f" % v for v in hist(blst[bkeep & (bml >= 35)]) / hist(blst[bkeep & (bml >= 35)]).sum() * k.sum())))
