"""高计数率下的读出缓冲结构：逐秒把事例按空洞切成"帧"（> 0.5 ms 无事例）与"块"（> 50 ms 无事例），
报每帧/每块的事例数。缓冲区按个数装满就停时，帧内事例数是一个固定值。不做能量准入（缓冲区装的是全部事例）。

用法: python3 grid_buffer_frames.py <卫星目录名> <YYYY-MM-DD> [文件名，缺省取当天计数率最高的过境] [只报 1 s 计数 ≥ 该值的秒，缺省 5000]
"""
import collections, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"


def files(sat, day):
    y, m, d = day.split("-")
    base = f"{G}/{sat}/fits7/{y}/{m}/{d}"
    v = sorted(os.listdir(base))[-1]
    return sorted(glob.glob(f"{base}/{v}/*.fits"))


def load(f):
    with fits.open(f, memmap=False) as h:
        t = np.sort(np.concatenate([np.asarray(h[f"EVENTS{k}"].data["TIME"], float) for k in range(4)]))
        gs = float(h["GTI"].data["START"][0])
    return gs, t


sat, day = sys.argv[1], sys.argv[2]
fs = files(sat, day)
if len(sys.argv) > 3 and sys.argv[3] != "-":
    fs = [x for x in fs if os.path.basename(x) == sys.argv[3]]
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 5000
best = None
for f in fs:
    gs, t = load(f)
    if t.size < 100:
        continue
    peak = np.histogram(t, np.arange(np.floor(t[0]), np.ceil(t[-1]) + 1))[0].max()
    if best is None or peak > best[0]:
        best = (peak, f)
peak, f = best
gs, t = load(f)
print(sat, os.path.basename(f), "最高 1 s 计数", peak)
edges = np.arange(np.floor(t[0]), np.ceil(t[-1]) + 1)
cnt = np.histogram(t, edges)[0]
sizes = collections.Counter()
print("  秒(过境内)  1s计数  帧数  帧内事例中位  帧长中位ms  帧内率/ms  块内事例")
shown = 0
for i in np.flatnonzero(cnt >= thr):
    w = t[(t >= edges[i]) & (t < edges[i] + 1)]
    dt = np.diff(w)
    fr = np.split(w, np.flatnonzero(dt > 0.0005) + 1)
    n = np.array([x.size for x in fr])
    dur = np.array([(x[-1] - x[0]) * 1e3 for x in fr])
    big = n >= 50
    for x in n[big]:
        sizes[int(x)] += 1
    ob = [x.size for x in np.split(w, np.flatnonzero(dt > 0.05) + 1)]
    if shown < 25 and (shown < 8 or i % 5 == 0):
        print("  %6.0f %8d %5d %8s %9s %9s  %s" % (edges[i] - gs, w.size, len(fr),
              int(np.median(n[big])) if big.any() else "-",
              "%.2f" % np.median(dur[big]) if big.any() else "-",
              "%.0f" % np.median(n[big] / np.maximum(dur[big], 1e-3)) if big.any() else "-", ob[:4]))
        shown += 1
print("  高计数率秒数", int((cnt >= thr).sum()), "；帧内事例数最常见的值:", sizes.most_common(8))
