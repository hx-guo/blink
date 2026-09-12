"""候选率随 |偶极磁纬|，分母是曝光。**先升后降、峰在极光卵**是 B 角的形态签名。

`grid07` 在 GRID-07 上做出这张表：全部候选率在 |mlat| 50–60° 达峰（74.25 /100h）
然后掉到 39.35，显著候选率同样在 50–60 达峰（30.37）。**单调上升才是"本底越高越容易
出伪信号"该有的样子；先升后降对应极光卵**（50–60 是极光带，60 以上进极隙/极盖、
沉降弱）。而且方向逆着探测难度——高磁纬本底更高、过 `fa ≤ 1e-5` 更难。

本脚本在 GRID-04 上做同一张表，**分母直接用 `triple_rate_scan4.py` 产出的 10 s 片**
（每片带 GTI 内的实际时长与片中心的 |偶极磁纬|），所以曝光与候选用的是同一套位置口径，
也不经过 POSATT 采样率。

用法: python3 cand_rate_mlat.py <SAT> <tgfs.json> <trip_slice_*.csv> [更多片文件]
"""

import csv
import glob
import json
import sys

import numpy as np

EDGES = [0, 10, 20, 25, 30, 33, 40, 50, 60, 90]


def main():
    sat, jpath = sys.argv[1], sys.argv[2]
    paths = []
    for a in sys.argv[3:]:
        paths.extend(sorted(glob.glob(a)))
    # 曝光：逐片时长按片中心磁纬进带
    exp = np.zeros(len(EDGES) - 1)
    nosrc = 0.0
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["sat"] != sat:
                    continue
                try:
                    d = float(r["dur_s"])
                    m = abs(float(r["mlat"]))
                except (TypeError, ValueError):
                    nosrc += float(r["dur_s"]) if r.get("dur_s") else 0.0
                    continue
                i = np.searchsorted(EDGES, m, side="right") - 1
                if 0 <= i < len(exp):
                    exp[i] += d
    # 候选：全部与显著，按自带位置算磁纬
    import math
    POLE_LAT, POLE_LON = math.radians(80.7), math.radians(-72.7)

    def dlat(lat, lon):
        la, lo = math.radians(lat), math.radians(lon)
        s = (math.sin(la) * math.sin(POLE_LAT)
             + math.cos(la) * math.cos(POLE_LAT) * math.cos(lo - POLE_LON))
        return math.degrees(math.asin(max(-1, min(1, s))))

    allc = np.zeros(len(EDGES) - 1)
    sigc = np.zeros(len(EDGES) - 1)
    for c in json.load(open(jpath)):
        s = c["signal"]
        pos = s.get("position") or {}
        if "latitude" not in pos:
            continue
        m = abs(dlat(pos["latitude"], pos["longitude"]))
        i = np.searchsorted(EDGES, m, side="right") - 1
        if not (0 <= i < len(allc)):
            continue
        allc[i] += 1
        if s.get("false_positive_per_year", 1e9) <= 1e-5:
            sigc[i] += 1
    tot = exp.sum()
    print("%s  可定位曝光 %.1f h（无位置的片 %.1f h 未计入）  候选 %d（显著 %d）"
          % (sat, tot / 3600.0, nosrc / 3600.0, int(allc.sum()), int(sigc.sum())))
    print("%-9s %10s %8s %12s %8s %12s" % ("|mlat|", "曝光 h", "全部", "全部 /100h", "显著", "显著 /100h"))
    for i in range(len(exp)):
        if exp[i] <= 0:
            continue
        h = exp[i] / 3600.0
        print("%-9s %10.2f %8d %12.2f %8d %12.2f"
              % ("%d-%d" % (EDGES[i], EDGES[i + 1]), h, int(allc[i]), 100 * allc[i] / h,
                 int(sigc[i]), 100 * sigc[i] / h))
    # 二项检验：显著候选落在 |mlat| >= 40 的比例对曝光占比
    k40 = int(sigc[np.array(EDGES[:-1]) >= 40].sum())
    p40 = exp[np.array(EDGES[:-1]) >= 40].sum() / tot
    n = int(sigc.sum())
    from math import comb
    pv = sum(comb(n, i) * p40 ** i * (1 - p40) ** (n - i) for i in range(k40, n + 1))
    print("\n显著候选 |mlat| ≥ 40：%d/%d，曝光占比 %.4f ⇒ 期望 %.1f，单侧 p = %.2e"
          % (k40, n, p40, n * p40, pv))


if __name__ == "__main__":
    main()
