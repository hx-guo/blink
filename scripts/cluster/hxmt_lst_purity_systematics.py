#!/usr/bin/env python3
"""三件事一起做：
  1. 模板的实测灵敏度（拿 f=1 和 f=0 各模拟一遍，看在这个 N 下分不分得开）——
     gbm 提醒的正对照，比看误差棒宽窄可靠。
  2. 亚秒多脉冲普查：|mlat| 高的那两个"最挤的日子"实际是 0.05 s / 0.1 s 内的 7 个脉冲。
     一个物理事件被数了 7 次，全样本里这种有多少，压成一个之后结论变不变。
  3. 用 gbm 新给的三版陆地模板 + gbm 自己的本底重拟一遍绝对纯度。
"""
import csv, sys, collections
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0
POLE_LAT, POLE_LON = 80.65, -72.68
rng = np.random.default_rng(20260911)

# gbm 给的三版陆地模板（原始整数计数）
T_FULL = np.array([201, 188, 75, 61, 197, 517, 375, 232], float)      # n=1846 A=0.988
T_NOTRAIN = np.array([172, 156, 68, 57, 164, 382, 288, 195], float)   # n=1482 A=0.877
T_2014 = np.array([36, 37, 15, 17, 36, 92, 70, 40], float)            # n=343  A=0.898
B_GBM = np.array([2945, 3162, 3015, 3047, 2920, 2927, 2990, 2845], float)  # n=23851 A=0.053


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


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


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


FS = np.linspace(-1.0, 2.0, 3001)


def fitf(obs, T, B):
    N = obs.sum()
    if N < 10:
        return float('nan'), float('nan'), float('nan'), float('nan')
    T = T / T.sum(); B = B / B.sum()
    M = N * (np.outer(FS, T) + np.outer(1 - FS, B))
    nll = -(obs * np.log(np.maximum(M, 1e-12)) - M).sum(axis=1)
    i = nll.argmin(); ok = FS[nll <= nll[i] + 1.92]
    mu = M[i]
    return FS[i], ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


def sens(N, T, B, ntrial=400):
    """真值 f=1 与 f=0 各模拟 ntrial 次，看拟出来的 f 分不分得开。"""
    Tn = T / T.sum(); Bn = B / B.sum()
    out = {}
    for ftrue in (1.0, 0.0):
        p = ftrue * Tn + (1 - ftrue) * Bn
        vals = [fitf(rng.multinomial(N, p).astype(float), T, B)[0] for _ in range(ntrial)]
        out[ftrue] = np.array(vals)
    return out


cat = list(csv.DictReader(open(sys.argv[1])))
sig = [c for c in cat if c['associated'] != '1']
lat = np.array([float(c['latitude']) for c in sig]); lon = np.array([float(c['longitude']) for c in sig])
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
ml = np.abs(mlat(lat, lon)); cls = np.array([classify_gbm(a, b) for a, b in zip(lat, lon)])
day = np.array([c['date'] for c in sig]); tsec = np.array([secs(c['start']) for c in sig])

bg = []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, assoc, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 6 == 0:
            bg.append((lst_hours(start, float(blon_)), float(blat_), float(blon_)))
blst = np.array([x[0] for x in bg]); blat = np.array([x[1] for x in bg]); blon = np.array([x[2] for x in bg])
bml = np.abs(mlat(blat, blon)); bcls = np.array([classify_gbm(x[1], x[2]) for x in bg])
B_HXMT_LAND = hist(blst[bcls == 'land'])

print("=== 1 模板在各个 N 下的实测灵敏度（真值 f=1 vs f=0 的拟合分布）===")
print("  用 gbm 全期陆地模板 + HXMT 陆地本底。分不开的档，f 拟出什么都不算数。")
for N in (1502, 1441, 120, 59, 41, 24):
    o = sens(N, T_FULL, B_HXMT_LAND)
    a1, a0 = o[1.0], o[0.0]
    print("  N=%-5d  真 f=1 -> %.3f +- %.3f（5 分位 %.3f） | 真 f=0 -> %.3f +- %.3f（95 分位 %.3f） | 两者重叠 %s"
          % (N, a1.mean(), a1.std(), np.percentile(a1, 5), a0.mean(), a0.std(), np.percentile(a0, 95),
             "是" if np.percentile(a1, 5) < np.percentile(a0, 95) else "否"))

print("")
print("=== 2 亚秒多脉冲普查：一个物理事件被数了几次 ===")
order = np.lexsort((tsec, day))
gid = np.empty(len(sig), int); g = -1
for j, idx in enumerate(order):
    if j == 0 or day[idx] != day[order[j - 1]] or tsec[idx] - tsec[order[j - 1]] > 1.0 \
            or abs(lat[idx] - lat[order[j - 1]]) > 0.5:
        g += 1
    gid[idx] = g
size = np.bincount(gid)
mult = size[gid]
print("  合并判据：同一天、时间间隔 <= 1 s、纬度差 <= 0.5 度 视为同一个物理事件")
print("  %d 行 -> %d 个物理事件；成员数分布 %s"
      % (len(sig), size.size, dict(sorted(collections.Counter(size).items()))))
print("  处在多脉冲组里的行 %d（%.1f%%）" % (int((mult > 1).sum()), 100 * (mult > 1).mean()))
for tag, m in (("陆地 |mlat| < 35", (cls == 'land') & (ml < 35)),
               ("陆地 |mlat| >= 35", (cls == 'land') & (ml >= 35))):
    print("  %-18s N=%-5d 其中在多脉冲组里 %d（%.1f%%）最大组 %d"
          % (tag, int(m.sum()), int((mult[m] > 1).sum()), 100 * (mult[m] > 1).mean(), int(size[gid[m]].max())))
first = np.zeros(len(sig), bool)
seen = set()
for idx in order:
    if gid[idx] not in seen:
        seen.add(gid[idx]); first[idx] = True
print("  -- 每组只留一行之后重拟 --")
for tag, m in (("陆地 |mlat| < 35", (cls == 'land') & (ml < 35)),
               ("陆地 |mlat| >= 35", (cls == 'land') & (ml >= 35))):
    for lbl, mm in (("全部行", m), ("每组一行", m & first)):
        v, lo, hi, c2 = fitf(hist(lst[mm]), T_FULL, B_HXMT_LAND)
        print("     %-18s %-8s N=%-5d f = %.3f [%.3f, %.3f] chi2 %4.1f/6" % (tag, lbl, int(mm.sum()), v, lo, hi, c2))

print("")
print("=== 3 用 gbm 三版模板 x 两套本底重拟绝对纯度 ===")
print("  模板 A：全期 %.3f / 摘串 %.3f / 只 2014 %.3f；本底 A：GBM %.3f / HXMT %.3f"
      % (amp(T_FULL), amp(T_NOTRAIN), amp(T_2014), amp(B_GBM), amp(B_HXMT_LAND)))
for tag, m in (("陆地 |lat| 0-26", (cls == 'land') & (np.abs(lat) < 26)),
               ("陆地 |mlat| < 35", (cls == 'land') & (ml < 35)),
               ("陆地 |mlat| >= 35", (cls == 'land') & (ml >= 35))):
    o = hist(lst[m])
    print("  %s  N=%d" % (tag, int(m.sum())))
    for tn, T in (("全期 1846", T_FULL), ("摘串 1482", T_NOTRAIN), ("只 2014 343", T_2014)):
        row = []
        for bn, B in (("GBM 本底", B_GBM), ("HXMT 本底", B_HXMT_LAND)):
            v, lo, hi, c2 = fitf(o, T, B)
            row.append("%s f = %.3f [%.3f, %.3f] chi2 %4.1f" % (bn, v, lo, hi, c2))
        print("     %-12s %s | %s" % (tn, row[0], row[1]))
