#!/usr/bin/env python3
"""陆地 LST 模板随纬度变不变：跨仪器借模板之前必须先问这一句。

GBM 倾角 25.6°、HXMT 43°。更高的纬度更偏大陆性，陆地日变化本来就该更强，
所以两台仪器的陆地模板即使都"近乎纯"，幅度也可能本来就不一样。把 HXMT 的
陆核按 |lat| 切开，低纬那段与 GBM 直接可比；两段若有系统差，那是纬度效应，
要先扣掉再谈纯度。这个检验 GBM 做不了（它覆盖不到 43°）。

幅度 A = (峰 − 谷) / 2 / 均值。A 的误差用泊松自助法给，**不能拿两个 A 相除
就下结论**——n = 524 时 A 的误差在 ±0.1 量级。

用法: hxmt_lst_latitude.py <catalog_v6.csv>
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5
R = 6371.0
rng = np.random.default_rng(20260911)


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def classify(lat, lon, ring=600.0):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    d = np.degrees(ring / R)
    la = [lat]; lo = [lon180]
    for az in range(0, 360, 45):
        a = np.radians(az)
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)) + 180) % 360) - 180)
    f = globe.is_land(np.array(la), np.array(lo))
    return 'land' if f.all() else ('ocean' if not f.any() else 'coast')


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


def amp_err(h, n=2000):
    """泊松自助：每格按自身计数重抽，给 A 的 68/95 区间。"""
    s = np.array([amp(rng.poisson(h)) for _ in range(n)])
    return np.percentile(s, [2.5, 16, 84, 97.5])


def phase(h):
    ang = 2 * np.pi * CENTRES / 24.0
    return (np.degrees(np.arctan2((h * np.sin(ang)).sum(), (h * np.cos(ang)).sum())) % 360) * 24 / 360


def main():
    cat = list(csv.DictReader(open(sys.argv[1])))
    sig = [c for c in cat if c['associated'] != '1']
    lat = np.array([float(c['latitude']) for c in sig])
    lon = np.array([float(c['longitude']) for c in sig])
    lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
    cls = np.array([classify(a, b) for a, b in zip(lat, lon)])
    land = cls == 'land'
    print("陆核 %d 个，|lat| 中位 %.1f°，范围 %.1f–%.1f°"
          % (land.sum(), np.median(np.abs(lat[land])),
             np.abs(lat[land]).min(), np.abs(lat[land]).max()))

    print("\n=== 陆核按 |lat| 切开（26° 是 GBM 倾角 25.6° 的边界）===")
    print("  %-18s %5s  %-30s %-22s %s" % ("段", "N", "每 3h 格 (%)", "A（68% / 95%）", "谐波相位"))
    out = {}
    for nm, m in (("|lat| <= 26（可与 GBM 比）", land & (np.abs(lat) <= 26)),
                  ("|lat| > 26", land & (np.abs(lat) > 26)),
                  ("全部陆核", land)):
        h = hist(lst[m])
        e = amp_err(h)
        out[nm] = (h, amp(h), e)
        print("  %-18s %5d  %-30s A = %.2f [%.2f–%.2f] / [%.2f–%.2f]  %.1f h"
              % (nm, int(m.sum()), " ".join("%4.1f" % v for v in 100 * h / h.sum()),
                 amp(h), e[1], e[2], e[0], e[3], phase(h)))

    a1 = out["|lat| <= 26（可与 GBM 比）"]; a2 = out["|lat| > 26"]
    # 两段 A 之差的自助分布
    d = np.array([amp(rng.poisson(a2[0])) - amp(rng.poisson(a1[0])) for _ in range(4000)])
    print("\n  高纬 − 低纬的 A 之差 = %+.2f，95%% [%+.2f, %+.2f] -> %s"
          % (a2[1] - a1[1], np.percentile(d, 2.5), np.percentile(d, 97.5),
             "有纬度效应" if np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0
             else "看不出纬度效应，形状可迁移"))
    p1, p2 = phase(a1[0]), phase(a2[0])
    print("  两段的谐波相位 %.1f h vs %.1f h，差 %.1f h" % (p1, p2, (p2 - p1 + 12) % 24 - 12))

    print("\n=== 归一模板（给 GBM 对比用）===")
    for nm in ("|lat| <= 26（可与 GBM 比）", "|lat| > 26", "全部陆核"):
        h = out[nm][0]
        print("  %-18s N=%4d: %s" % (nm, int(h.sum()), ", ".join("%.4f" % v for v in h / h.sum())))


if __name__ == "__main__":
    main()
