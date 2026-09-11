"""核对新加载器要依赖的三件事：梯长由连续性定出来、梯顶 10053.5、能阈道 = channel_above(40)。"""
import glob, sys
import numpy as np
from astropy.io import fits

def newest(pattern):
    best = None
    for path in glob.glob(pattern):
        parts = path.split("/")[-1].removesuffix(".fits").split("_")
        if parts and parts[-1].startswith("v"):
            try:
                v = int(parts[-1][1:])
            except ValueError:
                continue
            if best is None or v > best[0]:
                best = (v, path)
    return best[1] if best else None

def ladder_len(e_min, e_max):
    for k in range(1, len(e_min)):
        if abs(e_min[k] - e_max[k - 1]) > e_max[k - 1] * 1e-4:
            return k
    return len(e_min)

def channel_above(e_max, energy, ladder):
    idx = np.flatnonzero(np.asarray(e_max[:ladder], float) > energy)
    return int(idx[0]) if idx.size else ladder

for path in sys.argv[1:]:
    p = newest(path) if "*" in path else path
    if p is None:
        print(f"{path}: 无文件"); continue
    with fits.open(p, memmap=True) as hd:
        eb = hd["EBOUNDS"].data
        e_min = np.asarray(eb["E_MIN"], float)
        e_max = np.asarray(eb["E_MAX"], float)
        L = ladder_len(e_min, e_max)
        tail = e_max[L - 1]
        sent = f"e_max[{L}]={e_max[L]:.1f} e_min[{L}]={e_min[L]:.1f}" if L < len(e_min) else "表尾，无溢出块"
        print(f"{p.split('/')[-1]:40s} 行 {len(e_min):4d} 梯长 {L:4d} "
              f"ch0={e_min[0]:.2f}-{e_max[0]:.3f} 梯顶={tail:.1f} "
              f"ch54={e_min[54]:.3f}-{e_max[54]:.3f} 能阈道={channel_above(e_max,40.0,L)} {sent}")
