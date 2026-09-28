"""讲稿用缓冲区光变的数据：同一颗卫星上 GRID-03B 与 GRID-04 同一次过境。

导出：两颗星的逐秒计数（整次过境），以及 03B 最高计数率处前后 1.5 s 的逐事例时刻（两颗星都取），
画图在本地做（scripts/plot_grid_readout_slides.py）。事例准入与 grid_gap_lightcurve.py 相同。

用法: python3 grid_buffer_slide_data.py <day YYYY-MM-DD> <03B 文件名> <04 文件名> <中心 MET> <输出 npz>
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files, load

day, f3, f4, centre, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
res = {}
for tag, sat, name in (("g03b", "GRID-03B", f3), ("g04", "GRID-04", f4)):
    path = [p for p in pass_files(sat, day) if os.path.basename(p) == name][0]
    gs, ge, t, d = load(path)
    edges = np.arange(np.floor(gs), np.ceil(ge) + 1)
    res[tag + "_sec"] = edges[:-1]
    res[tag + "_cnt"] = np.histogram(t, edges)[0]
    w = (t > centre - 1.5) & (t < centre + 1.5)
    res[tag + "_t"] = t[w] - centre
    res[tag + "_gti"] = np.array([gs, ge])
res["centre"] = np.array([centre])
np.savez_compressed(out, **res)
print({k: v.shape for k, v in res.items()})
