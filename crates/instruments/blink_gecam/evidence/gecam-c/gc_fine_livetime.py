"""896 道纪元那 1,121 个小时，逐小时实算活时间。

先前记的"1,121 小时 ≈ 703 小时活时间"是**按全任务占空比 0.627 折的**，不是
实算的。那个数要写进论文（+6% 可搜曝光），**得是逐小时 GTI 与本小时求交再累加**
的实测值——同一个毛病在第 25 条上已经栽过一次（"拿回 20 小时"实际只有
9.5 小时活时间）。

只读 EBOUNDS 与 GTI 两张表，不碰事例流。

用法: gc_fine_livetime.py [起 2022-08-03] [止 2022-10-15]
"""

import datetime as dt
import glob
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)


def ladder_len(e_min, e_max):
    broken = np.flatnonzero(np.abs(e_min[1:] - e_max[:-1]) > e_max[:-1] * 1e-4)
    return int(broken[0]) + 1 if broken.size else e_min.size


def newest_per_hour(day):
    best = {}
    for path in glob.glob(f"{ROOT}/{day}/GRD_EVT/gcg_evt_*.fits"):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) < 5 or not parts[-1].startswith("v"):
            continue
        try:
            version = int(parts[-1][1:])
        except ValueError:
            continue
        hour = parts[3][:2]
        if hour not in best or version > best[hour][0]:
            best[hour] = (version, path)
    return {h: p for h, (_, p) in best.items()}


def main():
    first = sys.argv[1] if len(sys.argv) > 1 else "2022-08-03"
    last = sys.argv[2] if len(sys.argv) > 2 else "2022-10-15"
    days = sorted(
        d for d in glob.glob(f"{ROOT}/2022/*/*")
        if first <= d[-10:].replace("/", "-") <= last
    )
    tally = {896: [0, 0.0], 448: [0, 0.0], 0: [0, 0.0]}
    zero_live = 0
    for day_dir in days:
        day = day_dir[-10:]
        base = (dt.datetime(int(day[:4]), int(day[5:7]), int(day[8:10]),
                            tzinfo=dt.timezone.utc) - EPOCH).total_seconds()
        for hour, path in sorted(newest_per_hour(day).items()):
            start = base + int(hour) * 3600.0
            stop = start + 3600.0
            try:
                with fits.open(path, memmap=True) as hdus:
                    eb = hdus["EBOUNDS"].data
                    length = ladder_len(
                        np.asarray(eb["E_MIN"], float), np.asarray(eb["E_MAX"], float)
                    )
                    gti = hdus["GTI"].data
                    a = np.asarray(gti["START"], float)
                    b = np.asarray(gti["STOP"], float)
            except Exception:
                tally[0][0] += 1
                continue
            live = float(np.clip(np.minimum(b, stop) - np.maximum(a, start), 0, None).sum())
            key = length if length in tally else 0
            tally[key][0] += 1
            tally[key][1] += live
            if key == 896 and live == 0.0:
                zero_live += 1

    print(f"{first} .. {last}，{len(days)} 天")
    for key, name in ((896, "896 道（细梯）"), (448, "470 道"), (0, "读不出来/其他")):
        n, live = tally[key]
        if n:
            print(f"  {name:16s} {n:5d} 小时   活时间 {live:10.1f} s = "
                  f"{live / 3600:7.2f} 小时 = {live / 86400:6.3f} 天   "
                  f"占空比 {live / (n * 3600):.3f}")
    print(f"  细梯里活时间恰好为 0 的小时：{zero_live}")


if __name__ == "__main__":
    main()
