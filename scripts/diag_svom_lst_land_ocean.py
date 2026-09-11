"""陆海抵消：83 个证实 TGF 的 LST 平，会不会是陆地峰与海洋峰互相抵消？

陆地雷暴的日变化峰在午后（约 15–18 LST），海洋雷暴的峰在凌晨（约 03–06 LST）。
SVOM 倾角 30°、覆盖热带，海洋占比大，混在一起可能抵消成一条近似平的线。
所以拆开看**形状与峰位**——83 个拆两半统计力更弱，不做拟合，只看
两半的瑞利相位差是不是接近 180°。

同一件事在 WWLLN 落点上做一遍（上百万个，统计误差可忽略），既验证陆海相位差
确实存在，也给出「若真 TGF 按陆/海各自的分布，N 要多少才看得出来」。

陆海判定用 `global_land_mask`（1/100° 的全球掩模）。TGF 的位置取星下点，
落点若有（WWLLN 关联）就用落点。

用法: python3 diag_svom_lst_land_ocean.py <pool.csv> [incidence.csv] [AE 目录]
"""
import csv
import datetime as dt
import glob
import sys

import numpy as np
from global_land_mask import globe

NB = 8


def eot_hours(doy):
    b = 2 * np.pi * (doy - 81) / 364.0
    return (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0


def lst_of(iso, lon):
    t = dt.datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")
    utc = t.hour + t.minute / 60.0 + t.second / 3600.0
    return (utc + lon / 15.0 + eot_hours(t.timetuple().tm_yday)) % 24.0


def rayleigh(t, n_harm=1):
    """返回 (R, 相位小时, p)。相位是矢量和的方向，即分布的峰。"""
    z = np.exp(2j * np.pi * n_harm * np.asarray(t) / 24.0)
    m = z.mean()
    ph = (np.angle(m) / (2 * np.pi) * 24.0 / n_harm) % (24.0 / n_harm)
    return abs(m), ph, np.exp(-len(t) * abs(m) ** 2)


def show(tag, t):
    h, _ = np.histogram(t, bins=NB, range=(0, 24))
    r, ph, p = rayleigh(t)
    print("  %-14s N=%6d  " % (tag, len(t))
          + " ".join("%5.1f%%" % (100 * v / max(h.sum(), 1)) for v in h)
          + "   R=%.3f 峰 %.1f h  p=%.3f" % (r, ph, p))
    return ph, r


def main(pool, inc=None, wwlln=None):
    rows = [r for r in csv.DictReader(open(pool)) if abs(float(r["lon"])) <= 180]
    conf = [r for r in rows if float(r["fa"]) <= 1e-5 and r["in_cov"] == "1"
            and r["assoc"] == "1" and r["is_train"] == "0"]
    pos = {}
    if inc:
        for r in csv.DictReader(open(inc)):
            pos[r["start"]] = (float(r["lat_stroke"]), float(r["lon_stroke"]))
    lat, lon, t = [], [], []
    for r in conf:
        la, lo = pos.get(r["start"], (float(r["lat"]), float(r["lon"])))
        lat.append(la)
        lon.append(lo)
        t.append(lst_of(r["start"], lo))
    lat, lon, t = np.array(lat), np.array(lon), np.array(t)
    land = globe.is_land(lat, lon)
    print("83 个闪电证实 TGF（位置取 %s）：陆 %d，海 %d"
          % ("WWLLN 落点" if pos else "星下点", land.sum(), (~land).sum()))
    show("全部", t)
    ph_l, r_l = show("陆地", t[land])
    ph_o, r_o = show("海洋", t[~land])
    d = abs(ph_l - ph_o)
    print("  陆海峰位差 %.1f h（抵消要求接近 12 h）；两半的 R 分别 %.3f / %.3f"
          % (min(d, 24 - d), r_l, r_o))
    print("  判据：两半各自有对比度且峰位差接近 12 h → 是抵消；两半都平 → 是 N 不够")

    if not wwlln:
        return
    print("\nWWLLN 落点（|lat| ≤ 30°，与上面同样的日期）：")
    hh = {True: np.zeros(NB), False: np.zeros(NB)}
    ts = {True: [], False: []}
    for f in sorted(glob.glob(wwlln + "/AE*.loc")):
        day = dt.datetime.strptime(f.split("AE")[-1][:8], "%Y%m%d")
        eot = eot_hours(day.timetuple().tm_yday)
        hrs, lons, lats = [], [], []
        with open(f, errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i % 40:
                    continue
                p = line.split(",")
                if len(p) < 4:
                    continue
                try:
                    a, b, c = p[1].split(":")
                    la = float(p[2])
                    if abs(la) > 30:
                        continue
                    lo = float(p[3])
                except ValueError:
                    continue
                hrs.append(int(a) + int(b) / 60.0 + float(c) / 3600.0)
                lons.append(lo)
                lats.append(la)
        if not hrs:
            continue
        lo = np.array(lons)
        tt = (np.array(hrs) + lo / 15.0 + eot) % 24.0
        ld = globe.is_land(np.array(lats), lo)
        for k in (True, False):
            m = ld == k
            c, _ = np.histogram(tt[m], bins=NB, range=(0, 24))
            hh[k] += c
            ts[k].append(tt[m])
    for k, nm in ((True, "陆地"), (False, "海洋")):
        v = np.concatenate(ts[k])
        p = hh[k] / hh[k].sum()
        r, ph, _ = rayleigh(v)
        a = 2 * np.abs((p * np.exp(2j * np.pi * (np.arange(NB) + 0.5) * 3 / 24)).sum())
        print("  %-4s %8d  " % (nm, hh[k].sum())
              + " ".join("%5.1f%%" % (100 * x) for x in p)
              + "   峰 %.1f h，峰/谷 %.2f，A %.2f" % (ph, p.max() / p.min(), a))
        print("      若真 TGF 按这一半的分布，瑞利期望 p：" + "，".join(
            "N=%d %.3f" % (n, np.exp(-n * (a / 2) ** 2)) for n in (40, 83, 200)))
    pl = rayleigh(np.concatenate(ts[True]))[1]
    po = rayleigh(np.concatenate(ts[False]))[1]
    d = abs(pl - po)
    print("  WWLLN 的陆海峰位差 %.1f h" % min(d, 24 - d))
    np.savez("wwlln_lst_landsea.npz", land=hh[True], ocean=hh[False],
             weighted=hh[True] + hh[False])
    print("  → wwlln_lst_landsea.npz（陆/海两个模板，供两成分拟合用）")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
