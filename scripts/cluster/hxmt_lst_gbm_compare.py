#!/usr/bin/env python3
"""按 GBM 的分类口径重分一次，再拿 GBM 的陆地模板拟 HXMT 的陆地样本。

为什么要重分：两边的"陆地"不是同一个人群。
  GBM：星下点单点判读 —— 陆 = 星下点在陆上；近岸 = 在海上但离陆 < 300 km；远洋 = > 300 km
  HXMT（第 16 条）：星下点 + 600 km 八方位九点全一致才算核 —— 陆核是纯内陆
后者把离海岸 100 km 的内陆也踢进近岸，所以 GBM 的"陆地"是 HXMT 的陆核掺近岸。
实测 HXMT 陆核 A ≈ 1.05、近岸 A ≈ 0.57，混合当然落在中间——**口径差能单独解释
A 的差，不需要任何纯度损失**。所以绝对纯度必须在同一口径下拟。

两套口径各拟一次：若两个 f 一致，结论不依赖分类办法；不一致说明分类在 f 里留了印子。

用法: hxmt_lst_gbm_compare.py <catalog_v6.csv> <pool_lst.csv>
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5
R = 6371.0
rng = np.random.default_rng(20260911)

# main 转来的 GBM 模板（8 格，格心 LST 1.5 … 22.5 h，归一）
GBM = {
    'land':  (1846, np.array([0.1089, 0.1018, 0.0406, 0.0330, 0.1067, 0.2801, 0.2031, 0.1257])),
    'coast': (1636, np.array([0.1516, 0.1638, 0.1125, 0.0648, 0.0990, 0.1510, 0.1375, 0.1198])),
    'ocean': (649,  np.array([0.1556, 0.1787, 0.1525, 0.0817, 0.0847, 0.1125, 0.1248, 0.1094])),
}


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def ring(lat, lon, km, n_az):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    d = np.degrees(km / R)
    la, lo = [], []
    for k in range(n_az):
        a = 2 * np.pi * k / n_az
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)) + 180) % 360) - 180)
    return np.array(la), np.array(lo)


def classify_gbm(lat, lon):
    """GBM 口径：单点判陆；海上再问离陆是否 < 300 km（16 个方位近似）。"""
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    if bool(globe.is_land(lat, lon180)):
        return 'land'
    la, lo = ring(lat, lon, 300.0, 16)
    return 'coast' if globe.is_land(la, lo).any() else 'ocean'


def classify_nine(lat, lon):
    """HXMT 口径：星下点 + 600 km 八方位，九点全一致才算核。"""
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    la, lo = ring(lat, lon, 600.0, 8)
    f = np.concatenate([[bool(globe.is_land(lat, lon180))], globe.is_land(la, lo)])
    return 'land' if f.all() else ('ocean' if not f.any() else 'coast')


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


def fit(obs, T, B):
    N = obs.sum(); T = T / T.sum(); B = B / B.sum()
    fs = np.linspace(-1.0, 2.0, 3001)
    nll = np.array([-(obs * np.log(np.maximum(N * (f * T + (1 - f) * B), 1e-12))
                      - N * (f * T + (1 - f) * B)).sum() for f in fs])
    i = nll.argmin(); ok = fs[nll <= nll[i] + 1.92]
    mu = N * (fs[i] * T + (1 - fs[i]) * B)
    return fs[i], ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


def main():
    cat = list(csv.DictReader(open(sys.argv[1])))
    sig = [c for c in cat if c['associated'] != '1']
    lat = np.array([float(c['latitude']) for c in sig])
    lon = np.array([float(c['longitude']) for c in sig])
    lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])

    bg = []
    with open(sys.argv[2]) as f:
        r = csv.reader(f); next(r)
        for i, (start, fpy, blon, blat, assoc, nb, train) in enumerate(r):
            if float(fpy) > 1 and train == '0' and i % 12 == 0:
                bg.append((lst_hours(start, float(blon)), float(blat), float(blon)))

    for cname, fn in (("GBM 口径（单点 + 300 km）", classify_gbm),
                      ("HXMT 口径（九点 600 km）", classify_nine)):
        cls = np.array([fn(a, b) for a, b in zip(lat, lon)])
        bcls = np.array([fn(x[1], x[2]) for x in bg])
        blst = np.array([x[0] for x in bg])
        print("\n=== %s ===" % cname)
        print("  %-7s %5s %-32s %6s | GBM 同类 %s" % ("类", "N", "每 3h 格 (%)", "A", "A"))
        for k in ('land', 'coast', 'ocean'):
            h = hist(lst[cls == k])
            g = GBM[k][1] * GBM[k][0]
            print("  %-7s %5d %-32s %6.2f | %.2f (n=%d)"
                  % (k, int(h.sum()), " ".join("%4.1f" % v for v in 100 * h / h.sum()),
                     amp(h), amp(g), GBM[k][0]))
        # 低纬陆地（可与 GBM 覆盖带比）拿 GBM 模板拟
        m = (cls == 'land') & (np.abs(lat) <= 26)
        obs = hist(lst[m])
        B = hist(blst[(bcls == 'land')]) if (bcls == 'land').sum() > 200 else hist(blst)
        b, lo, hi, c2 = fit(obs, GBM['land'][1], B)
        print("  陆地 |lat|<=26 N=%d 拿 GBM 陆地模板拟: f = %.3f 95%% [%.3f, %.3f] chi2 = %.1f / 6"
              % (int(m.sum()), b, lo, hi, c2))
        m2 = cls == 'land'
        b2, lo2, hi2, c22 = fit(hist(lst[m2]), GBM['land'][1], B)
        print("  陆地 全纬度 N=%d:                     f = %.3f 95%% [%.3f, %.3f] chi2 = %.1f / 6"
              % (int(m2.sum()), b2, lo2, hi2, c22))
        mc = cls == 'coast'
        b3, lo3, hi3, c23 = fit(hist(lst[mc]), GBM['coast'][1],
                                hist(blst[bcls == 'coast']) if (bcls == 'coast').sum() > 200 else hist(blst))
        print("  近岸 N=%d 拿 GBM 近岸模板拟:           f = %.3f 95%% [%.3f, %.3f] chi2 = %.1f / 6"
              % (int(mc.sum()), b3, lo3, hi3, c23))


if __name__ == "__main__":
    main()
