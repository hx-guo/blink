"""审计一条"零例外"：`EVT_TYPE == 1` 的事例里 `PI >= 梯长` 到底有没有。

**这条先前是在 470 道的小时上量的，而 896 道那批的溢出段根本不在 448。**
所以要分梯长各量一遍，否则"零例外"可能只是没抽到细梯的小时——一个不覆盖
现象时间范围的样本给出的"零例外"是零信息。

同一趟顺带量 `EVT_TYPE == 2` 的 PI 落在哪里，确认它与梯外那一段是同一个集合。

用法: gc_overflow_audit.py <YYYY-MM-DD> <hour> [...]
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
    print("文件                        梯长  type1 事例   type1 且 PI>=梯长   占比      "
          "type2 的 PI 范围")
    for day, hour in zip(args[::2], args[1::2]):
        path = newest(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/gcg_evt_*_{hour}_v*.fits")
        if path is None:
            print(f"{day} {hour}h 无文件")
            continue
        with fits.open(path, memmap=True) as hdus:
            eb = hdus["EBOUNDS"].data
            _, length = ladder(
                np.asarray(eb["E_MIN"], float), np.asarray(eb["E_MAX"], float)
            )
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
        one = pi[evt == 1]
        two = pi[evt == 2]
        over = int((one >= length).sum())
        span = f"{two.min()}–{two.max()}" if two.size else "无"
        print(f"{os.path.basename(path):28s} {length:4d}  {one.size:10d}  "
              f"{over:14d}  {over / max(one.size, 1) * 100:8.5f}%  {span}")


if __name__ == "__main__":
    main()
