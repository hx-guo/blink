"""那 20 个"被 CPD 空壳拖死"的小时，修好之后真能拿回多少活时间？

**"整小时能载入"与"这一小时有曝光"是两件事。** 2023-01-30 03h 修好之后确实
从 `corrupt_data` 变成 `searched`，但它的 GTI 与本小时**一秒都不重叠**，
活时间 0、候选 0。所以"20 小时白丢"这个数是**载入口径**的，拿回来的曝光要
另外算。

用法: gc_stub_gti.py <gc_stub.csv>
"""

import csv
import glob
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH_DAYS = 18628  # 2021-01-01 相对 1970-01-01 的天数


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


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    stub_hours = {(r["yymmdd"], r["hour"]) for r in rows if r["kind"] == "CPD"}
    print(f"CPD 空壳小时 {len(stub_hours)}")
    total = 0.0
    recovered = []
    for yymmdd, hour in sorted(stub_hours):
        day = f"20{yymmdd[:2]}/{yymmdd[2:4]}/{yymmdd[4:6]}"
        grd = newest(f"{ROOT}/{day}/GRD_EVT/gcg_evt_{yymmdd}_{hour}_v*.fits")
        if grd is None or os.path.getsize(grd) < 1_000_000:
            continue          # GRD 本身也是空壳或缺失，那不是白丢的
        import datetime as dt
        start = (dt.datetime(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]),
                             int(hour), tzinfo=dt.timezone.utc)
                 - dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)).total_seconds()
        stop = start + 3600.0
        try:
            with fits.open(grd, memmap=True) as hdus:
                gti = hdus["GTI"].data
                a = np.asarray(gti["START"], float)
                b = np.asarray(gti["STOP"], float)
        except Exception as error:
            print(f"  {day} {hour}h  GTI 读不出来：{error}")
            continue
        live = float(np.clip(np.minimum(b, stop) - np.maximum(a, start), 0, None).sum())
        total += live
        recovered.append((day, hour, os.path.getsize(grd) / 1e6, live))
    print(f"其中 GRD 是好文件（> 1 MB）的 {len(recovered)} 小时")
    print("日期        时  GRD(MB)   本小时内活时间(s)")
    for day, hour, mb, live in recovered:
        print(f"{day}  {hour}   {mb:8.1f}   {live:10.1f}")
    live_hours = [r for r in recovered if r[3] > 0]
    print(f"\n合计活时间 {total:.1f} s = {total / 3600:.2f} 小时，"
          f"其中 {len(live_hours)}/{len(recovered)} 个小时的活时间 > 0")


if __name__ == "__main__":
    main()
