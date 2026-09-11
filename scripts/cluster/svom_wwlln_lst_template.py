"""用 WWLLN 落点自己的当地太阳时分布当雷暴日变化模板（外部、大统计量）。

83 个闪电证实 TGF 只够给一个与均匀不可分的模板（N = 83 时，即使真按雷暴分布，
瑞利检验的期望 p 也只有 0.1）。落点库在同样的日期、同样的纬度带上有上百万个
落点，可以当外部模板用。

两个已知偏差要一起写：
  1. **全部闪电的日变化 ≠ 产生 TGF 的那类闪电的日变化**；
  2. **WWLLN 自己的探测效率有日变化**（VLF 夜间传播好，夜间效率高）。
     实测模板的峰在 15–21 LST、谷在 09–12，是典型的午后雷暴峰而不是夜间峰，
     说明效率那一项不是主导，但它会压低峰谷比。

为了让模板的地理构成与候选一致，按经度分三区分别统计，再按候选在三区的
占比加权（TGF 的地理分布见 OPEN-QUESTIONS 第 18 条：美洲 51%、亚洲–海洋大陆
40%、非洲–欧洲 7%）。陆地与海洋的雷暴日变化相位不同，这一步不能省。

用法: python3 svom_wwlln_lst_template.py <AE 文件目录> <out.npz> [pool.csv]
"""
import csv
import datetime as dt
import glob
import sys

import numpy as np

NB = 8
STEP = 20            # 每 20 行取 1 行；几十万个落点，统计误差可忽略
LATMAX = 30.0
SECTORS = [("美洲", -150.0, -30.0), ("非洲欧洲", -30.0, 60.0), ("亚洲海洋大陆", 60.0, 210.0)]


def eot_hours(doy):
    """时差方程（小时），幅度 ±0.29 h。"""
    b = 2 * np.pi * (doy - 81) / 364.0
    return (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0


def sector(lon):
    x = (lon + 360.0) % 360.0
    for k, (_, lo, hi) in enumerate(SECTORS):
        a, b = (lo + 360.0) % 360.0, (hi + 360.0) % 360.0
        if a < b and a <= x < b:
            return k
        if a > b and (x >= a or x < b):
            return k
    return 2


def amp(p):
    return 2 * np.abs((p * np.exp(2j * np.pi * (np.arange(NB) + 0.5) * 3 / 24)).sum())


def main(d, out, pool=None):
    h = np.zeros((len(SECTORS), NB))
    files = sorted(glob.glob(d + "/AE*.loc"))
    for f in files:
        day = dt.datetime.strptime(f.split("AE")[-1][:8], "%Y%m%d")
        eot = eot_hours(day.timetuple().tm_yday)
        hrs, lons, secs = [], [], []
        with open(f, errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i % STEP:
                    continue
                p = line.split(",")
                if len(p) < 4:
                    continue
                try:
                    hh, mm, ss = p[1].split(":")
                    lat = float(p[2])
                    if abs(lat) > LATMAX:
                        continue
                    lon = float(p[3])
                except ValueError:
                    continue
                hrs.append(int(hh) + int(mm) / 60.0 + float(ss) / 3600.0)
                lons.append(lon)
                secs.append(sector(lon))
        if not hrs:
            continue
        t = (np.array(hrs) + np.array(lons) / 15.0 + eot) % 24.0
        s = np.array(secs)
        for k in range(len(SECTORS)):
            c, _ = np.histogram(t[s == k], bins=NB, range=(0, 24))
            h[k] += c
    print("%d 天，|lat| ≤ %.0f°，抽样 1/%d，共 %d 个落点" % (len(files), LATMAX, STEP, h.sum()))
    for k, (nm, _, _) in enumerate(SECTORS):
        p = h[k] / h[k].sum()
        print("  %-12s %8d  " % (nm, h[k].sum())
              + " ".join("%.1f%%" % (100 * v) for v in p)
              + "   峰/谷 %.2f，A %.2f" % (p.max() / p.min(), amp(p)))

    w = np.array([1.0, 1.0, 1.0])
    if pool:
        rows = [r for r in csv.DictReader(open(pool))
                if abs(float(r["lon"])) <= 180 and float(r["fa"]) <= 1e-5
                and r["in_cov"] == "1" and r["assoc"] == "1"]
        w = np.zeros(len(SECTORS))
        for r in rows:
            w[sector(float(r["lon"]))] += 1
        print("  按 %d 个闪电证实 TGF 的经度分区加权：" % len(rows)
              + " ".join("%s %.0f%%" % (SECTORS[k][0], 100 * w[k] / w.sum())
                         for k in range(len(SECTORS))))
    frac = h / h.sum(1)[:, None]
    tot = (frac * (w / w.sum())[:, None]).sum(0)
    print("  加权模板：" + " ".join("%.1f%%" % (100 * v) for v in tot)
          + "   峰/谷 %.2f，A %.2f" % (tot.max() / tot.min(), amp(tot)))
    print("  若真 TGF 按加权模板分布，瑞利检验的期望 p："
          + "，".join("N=%d %.3f" % (n, np.exp(-n * (amp(tot) / 2) ** 2))
                     for n in (83, 200, 611, 2547)))
    np.savez(out, per_sector=h, weighted=tot, weights=w,
             sectors=np.array([s[0] for s in SECTORS]))
    print("→", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
