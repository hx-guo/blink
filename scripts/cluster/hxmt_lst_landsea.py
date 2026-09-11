#!/usr/bin/env python3
"""把不经闪电台网的 TGF 样本按星下点的陆/海拆开，各出一条 LST 模板。

为什么要拆：陆地雷暴峰在当地午后、海洋峰在凌晨，混在一起互相抵消、把模板的
调制幅度 A 压低，而 A 是"绝对纯度"那条路线的瓶颈。

星下点不是 TGF 的位置：500 km 高度上源通常在星下点 ~600 km 以内，所以单点的
陆/海判读带约 600 km 的模糊。这里不装作没有这件事——除了星下点本身，再在
600 km 半径的八个方位上各判一次：全陆才算"陆核"、全海才算"海核"，混的单列
"近岸"。结论只用两个核样本，近岸那一档单独报、不混进模板。

用法: hxmt_lst_landsea.py <catalog_v6.csv>
"""
import csv, sys
import numpy as np
from global_land_mask import globe

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5
RING_KM = 600.0
R_EARTH = 6371.0


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def hist(x):
    return np.histogram(x, bins=NBIN, range=(0, 24))[0].astype(float)


def classify(lat, lon):
    """星下点 + 600 km 八方位：全陆 'land'、全海 'ocean'、混的 'coast'。"""
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    pts_lat = [lat]
    pts_lon = [lon180]
    d = np.degrees(RING_KM / R_EARTH)
    for az in range(0, 360, 45):
        a = np.radians(az)
        dlat = d * np.cos(a)
        dlon = d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)
        pts_lat.append(np.clip(lat + dlat, -89.9, 89.9))
        pts_lon.append(((lon180 + dlon + 180.0) % 360.0) - 180.0)
    flags = globe.is_land(np.array(pts_lat), np.array(pts_lon))
    if flags.all():
        return 'land'
    if not flags.any():
        return 'ocean'
    return 'coast'


def main():
    cat = list(csv.DictReader(open(sys.argv[1])))
    sig = [c for c in cat if c['associated'] != '1']
    lat = np.array([float(c['latitude']) for c in sig])
    lon = np.array([float(c['longitude']) for c in sig])
    lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
    cls = np.array([classify(a, b) for a, b in zip(lat, lon)])

    print("不经台网的显著未关联候选 %d 个，星下点分类（±600 km 一致才算核）：" % len(sig))
    for k in ('land', 'coast', 'ocean'):
        print("  %-6s %5d  (%.1f%%)" % (k, (cls == k).sum(), 100 * (cls == k).mean()))
    # 曝光本身的陆海比：用同一批候选的位置无法代表曝光，改用均匀经纬采样
    rng = np.random.default_rng(7)
    slat = np.degrees(np.arcsin(rng.uniform(-1, 1, 200000))) * 43.0 / 90.0
    slat = rng.uniform(-43, 43, 200000)
    slon = rng.uniform(-180, 180, 200000)
    frac_land = globe.is_land(slat, slon).mean()
    print("  参照：|lat|<=43 带内随机点落在陆地的比例 %.1f%%（单点判读，不是核）"
          % (100 * frac_land))

    print("\n每 3 h 格 (%%)，格心 LST 1.5 … 22.5 h")
    out = {}
    for k in ('land', 'ocean', 'coast'):
        x = lst[cls == k]
        if len(x) < 50:
            continue
        h = hist(x)
        p = 100 * h / h.sum()
        A = (h.max() - h.min()) / 2.0 / h.mean()
        # 一阶谐波的相位（当地时小时）
        ang = 2 * np.pi * CENTRES / 24.0
        c = (h * np.cos(ang)).sum(); s = (h * np.sin(ang)).sum()
        ph = (np.degrees(np.arctan2(s, c)) % 360.0) * 24.0 / 360.0
        print("  %-6s N=%4d  %s" % (k, len(x), " ".join("%5.1f" % v for v in p)))
        print("  %-6s        峰格 %.1f h，一阶谐波相位 %.1f h，幅度 A = %.2f"
              % ("", CENTRES[p.argmax()], ph, A))
        out[k] = h

    if 'land' in out and 'ocean' in out:
        dl = CENTRES[(out['land'] / out['land'].sum()).argmax()]
        do = CENTRES[(out['ocean'] / out['ocean'].sum()).argmax()]
        print("\n  陆/海峰位差 = %.1f h" % ((dl - do) % 24.0))
        pl = out['land'] / out['land'].sum(); po = out['ocean'] / out['ocean'].sum()
        var = pl * (1 - pl) / out['land'].sum() + po * (1 - po) / out['ocean'].sum()
        print("  两形状 chi2 = %.1f (dof=7)，p99 = 18.5"
              % (((pl - po) ** 2 / np.maximum(var, 1e-12)).sum()))
        allh = hist(lst)
        print("\n  合起来的幅度 A = %.2f；拆开后 陆 %.2f / 海 %.2f"
              % ((allh.max() - allh.min()) / 2 / allh.mean(),
                 (out['land'].max() - out['land'].min()) / 2 / out['land'].mean(),
                 (out['ocean'].max() - out['ocean'].min()) / 2 / out['ocean'].mean()))

    print("\n=== 供给：归一模板 ===")
    for k in ('land', 'ocean', 'coast'):
        if k in out:
            h = out[k]
            print("  %-6s N=%d: %s" % (k, int(h.sum()), ", ".join("%.4f" % v for v in h / h.sum())))


if __name__ == "__main__":
    main()
