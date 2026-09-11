"""GECAM-C：逐探头逐增益档的 PI 原始分布与 EBOUNDS 的行数，先把口径弄清楚。

`gc_edge.py` 第一轮出来的形状与 GECAM-A 完全不同（低增益档 85–99% 的事例 PI ≥ 448），
而 A 星上「`PI >= 448` 与 `EVT_TYPE == 2` 是同一个集合」是逐日逐位实测过的。两件事
对不上，说明在 C 星上要么增益语义不同、要么两档用的根本不是同一张表。先看原始分布
和 EBOUNDS 本身，不做任何解释。

用法: gc_pidist.py <YYYY-MM-DD> <hour>
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"

day, hour = sys.argv[1], int(sys.argv[2])
path = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits"))[-1]
print("文件", path)

with fits.open(path, memmap=True) as hdus:
    eb = hdus["EBOUNDS"].data
    emin = np.asarray(eb["E_MIN"], float)
    emax = np.asarray(eb["E_MAX"], float)
    print(f"\nEBOUNDS 行数 {emin.size}；道 447 = {emin[447]:.1f}–{emax[447]:.1f} keV")
    print("448 及以上各行（溢出段）：")
    for k in range(448, emin.size):
        print(f"   ch{k}  E_MIN {emin[k]:10.2f}  E_MAX {emax[k]:10.1f}")
    names = [c for c in eb.names]
    print("EBOUNDS 列", names)
    if "DET_ID" in names or "GAIN_TYPE" in names:
        for c in ("DET_ID", "GAIN_TYPE"):
            if c in names:
                print(f"  {c}[448:] =", np.asarray(eb[c])[448:])

    print("\n探头   档   事例数    PI 分位 0/1/25/50/75/99/100          >=448   <54")
    for hdu in hdus:
        if not hdu.name.startswith("EVENTS"):
            continue
        data = hdu.data
        if data is None or len(data) == 0:
            continue
        pi = np.asarray(data["PI"]).astype(int)
        gain = np.asarray(data["GAIN_TYPE"]).astype(int)
        evt = np.asarray(data["EVT_TYPE"]).astype(int)
        for g in (0, 1):
            for e in (1, 2):
                sel = (gain == g) & (evt == e)
                n = int(sel.sum())
                if n == 0:
                    continue
                p = np.percentile(pi[sel], [0, 1, 25, 50, 75, 99, 100])
                print(
                    f"{hdu.name} g{g} t{e} {n:9d}  "
                    + "/".join(f"{v:.0f}" for v in p)
                    + f"    {(pi[sel] >= 448).mean()*100:6.2f}%  {(pi[sel] < 54).mean()*100:6.2f}%"
                )
