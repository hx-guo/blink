#!/usr/bin/env python3
"""重做零分布，改两处错：
  1. 上一版把 1 sigma 和 95% 半区间直接比了。95% 半区间 ~ 1.96 sigma，要先换算。
  2. "按天整块抽"复制的是本底自己的日占用（一天十几个），而信号样本是 1087 天里 1069 个
     （一天约 1 个）。日尺度相关被夸大了。正确做法是**照抄信号自己的每日事件数分布**：
     信号某天有 k 个，就从本底里随机挑一天、取该天的 k 个。
"""
import csv, sys, collections
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0
T_NT = np.array([172, 156, 68, 57, 164, 382, 288, 195], float)
rng = np.random.default_rng(20260911)


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


FS = np.linspace(-1.5, 2.5, 4001)


def fitf(obs, T, B):
    N = obs.sum()
    Tn = T / T.sum(); Bn = B / B.sum()
    M = N * (np.outer(FS, Tn) + np.outer(1 - FS, Bn))
    nll = -(obs * np.log(np.maximum(M, 1e-12)) - M).sum(axis=1)
    i = nll.argmin(); ok = FS[nll <= nll[i] + 1.92]
    return FS[i], ok.min(), ok.max()


# --- 信号侧：口径对齐后的陆地样本，取它的每日事件数分布 ---
cat = list(csv.DictReader(open(sys.argv[1])))
lat_a = np.array([float(c['latitude']) for c in cat]); lon_a = np.array([float(c['longitude']) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])
day_a = np.array([c['date'] for c in cat])
t_a = np.array([daynum(c['date']) * 86400.0 + secs(c['start']) for c in cat])
o = np.argsort(t_a); ts = t_a[o]
nbr = np.zeros(len(cat), int)
nbr[o] = (np.searchsorted(ts, ts + 600.0, 'right') - np.searchsorted(ts, ts - 600.0, 'left')) - 1
cls = np.array([classify(a, b) for a, b in zip(lat_a, lon_a)])
keep = (~assoc) & (cls == 'land') & (nbr == 0)

shapes = {}
for tag, m in (("lo", keep & (np.abs(lat_a) < 26)), ("hi", keep & (np.abs(lat_a) >= 26))):
    c = collections.Counter(day_a[m])
    shapes[tag] = sorted(c.values(), reverse=True)
    print("信号 %s：N=%d，天数 %d，每日事件数分布 %s"
          % (tag, int(m.sum()), len(c), dict(sorted(collections.Counter(c.values()).items()))))

# --- 本底侧 ---
bgl, bgd, bgla = [], [], []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, ass, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 3 == 0:
            if classify(float(blat_), float(blon_)) == 'land':
                bgl.append(lst_hours(start, float(blon_))); bgd.append(start[:10]); bgla.append(abs(float(blat_)))
bgl = np.array(bgl); bgd = np.array(bgd); bgla = np.array(bgla)


def null(shape, latsel, ntrial=800):
    sel = latsel(bgla)
    x = bgl[sel]; d = bgd[sel]
    days = np.array(sorted(set(d)))
    byday = {k: x[d == k] for k in days}
    B = hist(x)
    vals, halfs = [], []
    for _ in range(ntrial):
        got = []
        for k in shape:
            for _try in range(30):
                dd = days[rng.integers(len(days))]
                pool = byday[dd]
                if len(pool) >= k:
                    got.append(rng.choice(pool, k, replace=False)); break
            else:
                got.append(rng.choice(x, k, replace=False))
        v, lo, hi = fitf(hist(np.concatenate(got)), T_NT, B)
        vals.append(v); halfs.append((hi - lo) / 2.0)
    return np.array(vals), np.array(halfs)


print("")
print("%-6s %-6s %-16s %-16s %-16s %s" % ("样本", "N", "似然 95% 半宽", "零分布 sigma", "零分布 95% 半宽", "该放大"))
for tag, latsel in (("lo", lambda a: a < 26), ("hi", lambda a: a >= 26)):
    shape = shapes[tag]
    v, h = null(shape, latsel)
    n95 = 1.96 * v.std()
    print("%-6s %-6d %-16.4f %-16.4f %-16.4f %.2f"
          % (tag, sum(shape), h.mean(), v.std(), n95, n95 / h.mean()))
    print("       真值 0，实测均值 %.4f；似然 95%% 区间的实际覆盖率 %.3f（名义 0.95）"
          % (v.mean(), float(np.mean(np.abs(v) <= h))))
