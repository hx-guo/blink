#!/usr/bin/env python3
"""GBM 口径下，陆地样本的纯度随 |lat| 怎么走。

上一步看到 f 从 0.936（|lat| <= 26）掉到 0.886（全纬度），差值只能来自高纬那一小撮。
高纬正是 REP 沉降的地方，所以这是一条可以直接验的推论。
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0
GBM_LAND = np.array([0.1089, 0.1018, 0.0406, 0.0330, 0.1067, 0.2801, 0.2031, 0.1257])


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


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


def fit(obs, T, B):
    N = obs.sum()
    if N == 0:
        return (float('nan'),) * 4
    T = T / T.sum(); B = B / B.sum()
    fs = np.linspace(-1.0, 2.0, 3001)
    nll = np.array([-(obs * np.log(np.maximum(N * (f * T + (1 - f) * B), 1e-12))
                      - N * (f * T + (1 - f) * B)).sum() for f in fs])
    i = nll.argmin(); ok = fs[nll <= nll[i] + 1.92]
    mu = N * (fs[i] * T + (1 - fs[i]) * B)
    return fs[i], ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


cat = list(csv.DictReader(open(sys.argv[1])))
sig = [c for c in cat if c['associated'] != '1']
lat = np.array([float(c['latitude']) for c in sig])
lon = np.array([float(c['longitude']) for c in sig])
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
cls = np.array([classify_gbm(a, b) for a, b in zip(lat, lon)])

bg = []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, (start, fpy, blon, blat, assoc, nb, train) in enumerate(r):
        if float(fpy) > 1 and train == '0' and i % 12 == 0:
            bg.append((lst_hours(start, float(blon)), float(blat), float(blon)))
bl = np.array([x[0] for x in bg])
bc = np.array([classify_gbm(x[1], x[2]) for x in bg])
B = hist(bl[bc == 'land'])

print("GBM 口径的陆地样本 %d 个，按 |lat| 分段拿 GBM 陆地模板拟：" % (cls == 'land').sum())
print("  %-14s %5s  %-28s %s" % ("|lat| 段", "N", "f 95%", "chi2/6"))
edges = [(0, 10), (10, 20), (20, 26), (26, 44), (0, 26), (0, 44)]
for a, b in edges:
    m = (cls == 'land') & (np.abs(lat) >= a) & (np.abs(lat) < b)
    v, lo, hi, c2 = fit(hist(lst[m]), GBM_LAND, B)
    tag = "  <- 与 GBM 覆盖带可比" if (a, b) == (0, 26) else ("  <- 高纬" if (a, b) == (26, 44) else "")
    print("  %-14s %5d  %.3f [%.3f, %.3f]%s %5.1f%s"
          % ("%d–%d°" % (a, b), int(m.sum()), v, lo, hi, " " * 6, c2, tag))
print("\n  高纬那一段若明显偏低，指向 REP 残留——高磁纬正是电子沉降的地方，")
print("  而 REP 的 LST 是平的（第 15 条负对照 f = 0.082），会被拟成本底。")
