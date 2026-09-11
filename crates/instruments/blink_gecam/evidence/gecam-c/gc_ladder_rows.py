"""逐日点一遍 EBOUNDS 的行数与梯长，找细梯纪元。

只读表头（`NAXIS2`）和 EBOUNDS 一张表，不碰事例，所以整任务扫得动。

用法: gc_ladder_rows.py <a|b|c> [每天抽几个小时=1]
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits

ARCHIVE = {
    "a": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_A/GRD_evt", "gag"),
    "b": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_B/GRD_evt", "gbg"),
    "c": ("/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily", "GRD_EVT", "gcg"),
}


def ladder_len(e_min, e_max):
    broken = np.flatnonzero(np.abs(e_min[1:] - e_max[:-1]) > e_max[:-1] * 1e-4)
    return int(broken[0]) + 1 if broken.size else e_min.size


def main():
    root, subdir, prefix = ARCHIVE[sys.argv[1]]
    per_day = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    days = sorted(glob.glob(f"{root}/[12][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]"))
    tally = {}
    first = {}
    last = {}
    checked = 0
    for day in days:
        files = sorted(glob.glob(f"{day}/{subdir}/{prefix}_evt_*.fits"))
        if not files:
            continue
        step = max(1, len(files) // per_day)
        for path in files[::step][:per_day]:
            try:
                with fits.open(path, memmap=True) as hdus:
                    eb = hdus["EBOUNDS"]
                    rows = eb.header["NAXIS2"]
                    data = eb.data
                    length = ladder_len(
                        np.asarray(data["E_MIN"], float), np.asarray(data["E_MAX"], float)
                    )
            except Exception as error:  # 坏文件不该让整趟扫停下来
                key = ("error", type(error).__name__)
            else:
                key = (rows, length)
            checked += 1
            tally[key] = tally.get(key, 0) + 1
            stamp = day[-10:]
            first.setdefault(key, stamp)
            last[key] = stamp
    print(f"{sys.argv[1].upper()} 星：点了 {checked} 个文件，{len(days)} 天")
    for key in sorted(tally, key=lambda k: -tally[k]):
        print(f"  行 {key[0]} 梯长 {key[1]}：{tally[key]:5d} 个   {first[key]} .. {last[key]}")


if __name__ == "__main__":
    main()
