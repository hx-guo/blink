"""同一颗卫星上 GRID-03B 与 GRID-04 的相对时间差：以 03B 上同一时戳 ≥3 路的簇（带电粒子）为参照，
把 04 的事例对它们叠加，看峰落在哪。粒子两台可能同时打到，簇的数量远多于 TGF。

用法: python3 grid_pair_lag.py <day YYYY-MM-DD> [<day> ...]
每天取 03B 与 04 重叠的过境，输出 dt（04 − 03B 簇，微秒）在 ±3 ms 的 20 µs 直方。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files, load

TICK = 2.0 ** -22
edges = np.arange(-3000, 3000.1, 20)
tot = np.zeros(len(edges) - 1); ncl = 0
for day in sys.argv[1:]:
    f3 = pass_files("GRID-03B", day); f4 = pass_files("GRID-04", day)
    L4 = [load(p) for p in f4]
    for p in f3:
        gs, ge, t, d = load(p)
        k = np.round(t / TICK).astype(np.int64)
        u, cnt = np.unique(k, return_counts=True)
        cl = u[cnt >= 3] * TICK
        for gs4, ge4, t4, _ in L4:
            lo, hi = max(gs, gs4), min(ge, ge4)
            if hi - lo < 60: continue
            c = cl[(cl > lo + 1) & (cl < hi - 1)]
            ncl += c.size
            i0 = np.searchsorted(t4, c - 3e-3); i1 = np.searchsorted(t4, c + 3e-3)
            for a, b, x in zip(i0, i1, c):
                tot += np.histogram((t4[a:b] - x) * 1e6, edges)[0]
print("clusters", ncl)
for lo, n in zip(edges[:-1], tot):
    print("%6d %6d" % (lo, n))
