#!/usr/bin/env python3
"""高纬那一段纯度掉到 0.38：是 REP 残留，还是模板的相位在中纬对不上？

第 20 条量到 GBM 口径的陆地样本里，|lat| 26–44° 的纯度 0.380 [0.100, 0.652]，
与 0–26° 的 0.936 [0.861, 1.006] 区间不重叠。"高纬 = REP 沉降区"只是一个
合理的故事，不是证据。下面五条各自能把它证伪：

  A 几何纬 vs 地磁纬：REP 是地磁现象，若污染源是 REP，|mlat| 应当比 |lat| 排得更整齐。
  B 陆/近岸/远洋：REP 不挑陆海，掉纯度应当三类同掉；只有陆地掉 => 是陆地模板的问题。
  C 自由相位的一次谐波：中纬雷暴本来就比热带晚一两个小时，**相位对不上会被拟成纯度损失**。
    若高纬样本自己有强调制、只是峰相移了，那是模板相位不匹配，不是本底稀释。
  D 日期成团：REP 跟着地磁暴走，应当挤在少数几天里；真 TGF 摊在全任务上。
  E 事件本身的量（duration / count / neighbors_10min / n_bg）：REP 比 TGF 长、本底高。

用法: hxmt_lst_rep_residual.py <catalog_v6.csv> <pool_lst.csv>
"""
import csv, sys, collections
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0
GBM_LAND = np.array([0.1089, 0.1018, 0.0406, 0.0330, 0.1067, 0.2801, 0.2031, 0.1257])
GBM_COAST = np.array([0.1516, 0.1638, 0.1125, 0.0648, 0.0990, 0.1510, 0.1375, 0.1198])
GBM_OCEAN = np.array([0.1556, 0.1787, 0.1525, 0.0817, 0.0847, 0.1125, 0.1248, 0.1094])
# IGRF-13 的 2020 年偶极北极；HXMT 跨 2017–2025，极点每年漂 ~0.05°，
# 对 |mlat| 的影响远小于这里的分段宽度。
POLE_LAT, POLE_LON = 80.65, -72.68
rng = np.random.default_rng(20260911)


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def mlat(lat, lon):
    """中心偶极近似的地磁纬度。"""
    la, lo = np.radians(lat), np.radians(lon)
    pa, po = np.radians(POLE_LAT), np.radians(POLE_LON)
    s = np.sin(la) * np.sin(pa) + np.cos(la) * np.cos(pa) * np.cos(lo - po)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


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
    if N < 20:
        return (float('nan'),) * 4
    T = T / T.sum(); B = B / B.sum()
    fs = np.linspace(-1.0, 2.0, 3001)
    nll = np.array([-(obs * np.log(np.maximum(N * (f * T + (1 - f) * B), 1e-12))
                      - N * (f * T + (1 - f) * B)).sum() for f in fs])
    i = nll.argmin(); ok = fs[nll <= nll[i] + 1.92]
    mu = N * (fs[i] * T + (1 - fs[i]) * B)
    return fs[i], ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


def rayleigh(lst):
    """一次谐波的振幅与峰相（小时），外加瑞利 p。"""
    th = 2 * np.pi * np.asarray(lst) / 24.0
    n = len(th)
    c, s = np.cos(th).mean(), np.sin(th).mean()
    r = np.hypot(c, s)
    return 2 * r, (np.arctan2(s, c) % (2 * np.pi)) * 24 / (2 * np.pi), np.exp(-n * r * r), n


def main():
    cat = list(csv.DictReader(open(sys.argv[1])))
    sig = [c for c in cat if c['associated'] != '1']
    lat = np.array([float(c['latitude']) for c in sig])
    lon = np.array([float(c['longitude']) for c in sig])
    lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
    ml = np.abs(mlat(lat, lon))
    cls = np.array([classify_gbm(a, b) for a, b in zip(lat, lon)])
    dur = np.array([float(c['duration']) for c in sig])
    cnt = np.array([float(c['count']) for c in sig])
    nb10 = np.array([float(c['neighbors_10min']) for c in sig])
    nbg = np.array([float(c['n_bg']) for c in sig])
    day = np.array([c['date'] for c in sig])

    bg = []
    with open(sys.argv[2]) as f:
        r = csv.reader(f); next(r)
        for i, (start, fpy, blon, blat, assoc, nbn, train) in enumerate(r):
            if float(fpy) > 1 and train == '0' and i % 12 == 0:
                bg.append((lst_hours(start, float(blon)), float(blat), float(blon)))
    blst = np.array([x[0] for x in bg])
    blat = np.array([x[1] for x in bg])
    blon = np.array([x[2] for x in bg])
    bml = np.abs(mlat(blat, blon))
    bcls = np.array([classify_gbm(x[1], x[2]) for x in bg])

    land = cls == 'land'
    bland = bcls == 'land'
    print("陆地信号 %d 个（GBM 口径），陆地本底 %d 个" % (land.sum(), bland.sum()))

    print("\n=== A 几何纬 vs 地磁纬：哪个把纯度排得更整齐 ===")
    print("  每一段的本底都取同段的陆地本底，口径自洽。")
    for name, x, bx, edges in (
            ("|lat|", np.abs(lat), np.abs(blat), [(0, 10), (10, 20), (20, 26), (26, 44)]),
            ("|mlat|", ml, bml, [(0, 10), (10, 20), (20, 30), (30, 50)])):
        print("  -- 按 %s 分段 --" % name)
        for a, b in edges:
            m = land & (x >= a) & (x < b)
            bm = bland & (bx >= a) & (bx < b)
            B = hist(blst[bm]) if bm.sum() > 200 else hist(blst[bland])
            v, lo, hi, c2 = fit(hist(lst[m]), GBM_LAND, B)
            print("     %-10s N=%-5d 本底 %-6d f = %.3f [%.3f, %.3f]  chi2 = %4.1f/6"
                  % ("%d–%d°" % (a, b), int(m.sum()), int(bm.sum()), v, lo, hi, c2))

    print("\n=== B 高纬的纯度下降是不是只发生在陆地 ===")
    print("  REP 不挑陆海；只有陆地掉 => 是陆地模板的问题不是 REP。")
    for kname, T in (('land', GBM_LAND), ('coast', GBM_COAST), ('ocean', GBM_OCEAN)):
        for a, b in ((0, 26), (26, 44)):
            m = (cls == kname) & (np.abs(lat) >= a) & (np.abs(lat) < b)
            bm = (bcls == kname) & (np.abs(blat) >= a) & (np.abs(blat) < b)
            B = hist(blst[bm]) if bm.sum() > 200 else hist(blst[bcls == kname])
            v, lo, hi, c2 = fit(hist(lst[m]), T, B)
            print("  %-6s |lat| %-7s N=%-5d f = %.3f [%.3f, %.3f]  chi2 = %4.1f/6"
                  % (kname, "%d–%d°" % (a, b), int(m.sum()), v, lo, hi, c2))

    print("\n=== C 自由相位一次谐波：高纬是没有调制，还是调制移了相 ===")
    print("  模板相位对不上会被拟成纯度损失，所以要先量高纬样本自己的峰相。")
    for tag, m in (("陆地 |lat| 0–26", land & (np.abs(lat) < 26)),
                   ("陆地 |lat| 26–44", land & (np.abs(lat) >= 26)),
                   ("陆地本底 0–26", None), ("陆地本底 26–44", None)):
        if m is None:
            bm = bland & ((np.abs(blat) < 26) if "0–26" in tag else (np.abs(blat) >= 26))
            a, ph, p, n = rayleigh(blst[bm])
        else:
            a, ph, p, n = rayleigh(lst[m])
        print("  %-18s N=%-5d 一次谐波幅 %.3f 峰相 %5.2f h  瑞利 p = %.2e" % (tag, n, a, ph, p))
    # 低纬样本随机抽到与高纬同样的 N，看在这个 N 下期望的 p 是多少
    hi_n = int((land & (np.abs(lat) >= 26)).sum())
    lo_lst = lst[land & (np.abs(lat) < 26)]
    ps = [rayleigh(rng.choice(lo_lst, hi_n, replace=False))[2] for _ in range(2000)]
    print("  低纬样本降采样到 N=%d 时的瑞利 p：中位 %.2e，95 分位 %.2e" % (hi_n, np.median(ps), np.percentile(ps, 95)))
    print("  （这条回答「高纬的 p 大是不是纯粹因为只有 %d 个」。）" % hi_n)

    print("\n=== D 日期成团：REP 跟着地磁暴走，应挤在少数几天 ===")
    for tag, m in (("陆地 0–26", land & (np.abs(lat) < 26)), ("陆地 26–44", land & (np.abs(lat) >= 26))):
        d = day[m]
        c = collections.Counter(d)
        n, nd = len(d), len(c)
        top = c.most_common(5)
        # 同 N 下把日期从该类的整体日期池里随机抽，看"最大一天占比"的偶然线
        pool = day[land]
        share = [max(collections.Counter(rng.choice(pool, n, replace=False)).values()) / n for _ in range(1000)]
        print("  %-12s N=%-5d 天数 %-4d 每天 %.2f 个，最大一天 %d 个（%.1f%%），"
              "置换偶然线 %.1f%% [%.1f, %.1f]"
              % (tag, n, nd, n / nd, top[0][1], 100 * top[0][1] / n,
                 100 * np.median(share), 100 * np.percentile(share, 2.5), 100 * np.percentile(share, 97.5)))
        print("       最挤的 5 天：%s" % ", ".join("%s×%d" % t for t in top))

    print("\n=== E 事件本身的量：高纬那一撮跟低纬是不是同一种东西 ===")
    lo_m = land & (np.abs(lat) < 26)
    hi_m = land & (np.abs(lat) >= 26)
    for nm, v in (("duration (s)", dur), ("count", cnt), ("neighbors_10min", nb10), ("n_bg", nbg)):
        a, b = v[lo_m], v[hi_m]
        # Mann-Whitney U 的正态近似
        allv = np.concatenate([a, b]); r = allv.argsort().argsort().astype(float) + 1
        ra = r[:len(a)].sum(); na, nbn = len(a), len(b)
        u = ra - na * (na + 1) / 2.0
        mu, sd = na * nbn / 2.0, np.sqrt(na * nbn * (na + nbn + 1) / 12.0)
        z = (u - mu) / sd
        print("  %-16s 低纬中位 %-10.4g 高纬中位 %-10.4g  U 检验 z = %+5.2f" % (nm, np.median(a), np.median(b), z))


if __name__ == "__main__":
    main()
