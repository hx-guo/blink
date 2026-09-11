#!/usr/bin/env python3
"""按 GBM 单点口径重分 HXMT 候选，导出三类模板 + 同口径本底的原始计数。

供给别的仪器用，所以每个 N 都从数据现算并打印，不写字面量。
用法: gbm_caliber_templates.py <catalog_v6.csv> <pool_lst.csv>
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
R = 6371.0


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
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    if bool(globe.is_land(lat, lon180)):
        return 'land'
    la, lo = ring(lat, lon, 300.0, 16)
    return 'coast' if globe.is_land(la, lo).any() else 'ocean'


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def amp(h):
    return (h.max() - h.min()) / 2.0 / h.mean()


def show(tag, h):
    n = int(h.sum())
    print("  %-28s N=%-6d A=%.3f" % (tag, n, amp(h)))
    print("    原始计数 %s" % " ".join("%6d" % v for v in h.astype(int)))
    print("    归一化   %s" % " ".join("%6.4f" % v for v in h / h.sum()))
    return n


def main():
    cat = list(csv.DictReader(open(sys.argv[1])))
    sig = [c for c in cat if c['associated'] != '1']
    print("目录总行 %d，其中未关联 %d（模板样本）" % (len(cat), len(sig)))
    lat = np.array([float(c['latitude']) for c in sig])
    lon = np.array([float(c['longitude']) for c in sig])
    lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])

    bg = []
    n_pool = 0
    with open(sys.argv[2]) as f:
        r = csv.reader(f); next(r)
        for i, (start, fpy, blon, blat, assoc, nb, train) in enumerate(r):
            n_pool += 1
            if float(fpy) > 1 and train == '0' and i % 12 == 0:
                bg.append((lst_hours(start, float(blon)), float(blat), float(blon)))
    print("全池 %d 行，抽出 fa>1 且非列车的每 12 个取 1 个 -> 本底 %d 个" % (n_pool, len(bg)))

    cls = np.array([classify_gbm(a, b) for a, b in zip(lat, lon)])
    bcls = np.array([classify_gbm(x[1], x[2]) for x in bg])
    blst = np.array([x[0] for x in bg])

    print("\n=== GBM 单点口径：HXMT 显著未关联候选（信号）===")
    tot = 0
    for k in ('land', 'coast', 'ocean'):
        tot += show(k, hist(lst[cls == k]))
    print("  三类合计 %d（应等于未关联总数 %d）" % (tot, len(sig)))

    print("\n=== GBM 单点口径：HXMT 同口径本底（fa > 1，1/12 抽样）===")
    tot = 0
    for k in ('land', 'coast', 'ocean'):
        tot += show(k, hist(blst[bcls == k]))
    print("  三类合计 %d（应等于本底总数 %d）" % (tot, len(bg)))


if __name__ == "__main__":
    main()
