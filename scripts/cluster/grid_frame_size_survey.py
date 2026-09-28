"""读出缓冲的帧大小在任务期内是否恒定：每颗星均匀挑若干天，每天取计数率最高的过境，
在 1 s 计数 ≥ 阈值的秒里按空洞（> 0.5 ms 无事例）切帧，统计帧内事例数的众数与其占比。

不做能量准入（缓冲区装的是全部事例）。一行一天：
    卫星, 日期, 文件, 最高 1 s 计数, 成帧秒数, 帧总数(≥50 事例), 众数, 众数占比, 次众数

用法: python3 grid_frame_size_survey.py <卫星目录名> <第几天(0 起)> <共几天> <1 s 计数阈值>
"""
import collections, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
sat, k, nd, thr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
days = sorted(glob.glob(f"{G}/{sat}/fits7/20*/*/*"))
day = days[int(round(k * (len(days) - 1) / max(nd - 1, 1)))]
vs = sorted(os.listdir(day))
best = None
for f in sorted(glob.glob(f"{day}/{vs[-1]}/*.fits")):
    try:
        with fits.open(f, memmap=False) as h:
            t = np.sort(np.concatenate([np.asarray(h[f"EVENTS{i}"].data["TIME"], float) for i in range(4)]))
    except Exception:
        continue
    if t.size < 100:
        continue
    c = np.histogram(t, np.arange(np.floor(t[0]), np.ceil(t[-1]) + 1))[0]
    if best is None or c.max() > best[0]:
        best = (int(c.max()), f, t, c)
date = "/".join(day.split("/")[-3:])
if best is None:
    print(f"{sat},{date},-,0,0,0,-,-,-")
    sys.exit()
peak, f, t, c = best
edges = np.arange(np.floor(t[0]), np.ceil(t[-1]) + 1)
sizes = collections.Counter()
nsec = 0
for i in np.flatnonzero(c >= thr):
    w = t[(t >= edges[i]) & (t < edges[i] + 1)]
    fr = np.split(w, np.flatnonzero(np.diff(w) > 0.0005) + 1)
    n = [x.size for x in fr if x.size >= 50]
    if len(n) >= 2:
        nsec += 1
        sizes.update(n)
tot = sum(sizes.values())
mc = sizes.most_common(2)
if tot:
    print(f"{sat},{date},{os.path.basename(f)},{peak},{nsec},{tot},{mc[0][0]},{mc[0][1]/tot:.3f},{mc[1][0] if len(mc) > 1 else '-'}")
else:
    print(f"{sat},{date},{os.path.basename(f)},{peak},0,0,-,-,-")
