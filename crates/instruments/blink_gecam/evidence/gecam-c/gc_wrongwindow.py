"""照搬道号阈值在 896 道纪元上具体错成什么样：把两套准入并排数一遍。

第 21 条说"照搬 ch54/448 会把能窗整个搬错位（14 keV 以上全收、514 keV 封顶）"
——这是从 EBOUNDS 推出来的，**没有量过它在事例上的后果有多大**。这里量：
同一个小时，用逐文件推出来的 (ch109, 896) 和照搬的 (ch54, 448) 各数一遍
通过准入的事例数与能量覆盖。

用法: gc_wrongwindow.py <YYYY-MM-DD> <hour> [...]
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_ENERGY_KEV = 40.0


def newest(pattern):
    best = None
    for path in glob.glob(pattern):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) >= 5 and parts[-1].startswith("v"):
            try:
                v = int(parts[-1][1:])
            except ValueError:
                continue
            if best is None or v > best[0]:
                best = (v, path)
    return best[1] if best else None


def ladder(e_min, e_max):
    broken = np.flatnonzero(np.abs(e_min[1:] - e_max[:-1]) > e_max[:-1] * 1e-4)
    length = int(broken[0]) + 1 if broken.size else e_min.size
    above = np.flatnonzero(e_max[:length] > MIN_ENERGY_KEV)
    return (int(above[0]) if above.size else length), length


def main():
    args = sys.argv[1:]
    for day, hour in zip(args[::2], args[1::2]):
        path = newest(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/gcg_evt_*_{hour}_v*.fits")
        if path is None:
            print(f"{day} {hour}h 无文件")
            continue
        with fits.open(path, memmap=True) as hdus:
            eb = hdus["EBOUNDS"].data
            e_min = np.asarray(eb["E_MIN"], float)
            e_max = np.asarray(eb["E_MAX"], float)
            derived = ladder(e_min, e_max)
            pis, evts = [], []
            for hdu in hdus:
                if not hdu.name.startswith("EVENTS"):
                    continue
                data = hdu.data
                if data is None or len(data) == 0:
                    continue
                pis.append(np.asarray(data["PI"]).astype(np.int32))
                evts.append(np.asarray(data["EVT_TYPE"]).astype(np.int8))
        pi = np.concatenate(pis)
        evt = np.concatenate(evts)
        normal = evt == 1
        print(f"\n{os.path.basename(path)}  EBOUNDS {e_min.size} 行  "
              f"事例 {pi.size}（EVT_TYPE==1 的 {int(normal.sum())}）")
        for label, (lo, hi) in (("逐文件推出来的", derived), ("照搬 470 道那套", (54, 448))):
            keep = normal & (pi >= lo) & (pi < hi)
            e_lo = e_min[lo] if lo < e_min.size else float("nan")
            e_hi = e_max[hi - 1] if hi - 1 < e_max.size else float("nan")
            print(f"  {label:14s} ch[{lo}, {hi})  能量 [{e_lo:.2f}, {e_hi:.1f}] keV  "
                  f"收 {int(keep.sum()):10d} 条 = {keep.sum() / normal.sum() * 100:6.2f}%")
        lo_d, hi_d = derived
        right = normal & (pi >= lo_d) & (pi < hi_d)
        wrong = normal & (pi >= 54) & (pi < 448)
        print(f"  只有照搬那套收的（14–40 keV 的软事例）  "
              f"{int((wrong & ~right).sum()):10d} 条")
        print(f"  只有正确那套收的（514 keV 以上的硬事例）"
              f"{int((right & ~wrong).sum()):10d} 条")


if __name__ == "__main__":
    main()
