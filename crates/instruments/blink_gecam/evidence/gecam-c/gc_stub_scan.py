"""GECAM-C：归档里有一批「空壳」事例文件——表在、行数 0、连列都没有。

起因：2023-01-30 03h 被搜索判 `corrupt_data`（`Cannot find column "TIME"`），而那一小时
的 GRD 是一个 269 MB、1,481 万事例的**好文件**。死在 CPD 上：`gcc_evt_230130_03_v00.fits`
只有 28,800 字节，两个 `EVENTS` 表**零行零列**。`from_epoch` 的规则是「只放过找不到
文件；文件在却读坏了仍旧报错」，空壳被算进了后一类，于是**一小时好的 GRD 数据因为
CPD 是占位文件而整个丢掉**。

空壳与真正的损坏不是一回事：空壳是管线在真数据到位之前写下的占位，形态明确
（零行、零列），可以稳妥地与「读到一半是乱码」分开。

这个脚本按「最大版本的 EVENTS 表有没有列」逐小时点一遍，把账算清楚：
**GRD 空壳必须照旧致命**（没有事例就没有搜索），**CPD 空壳应当降级成「这一小时没有
CPD 数据」**，与文件缺失同等对待。

用法: gc_stub_scan.py [<输出 CSV>]
"""

import collections
import csv
import glob
import os
import sys

from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"


def latest_per_hour(paths):
    best = {}
    for path in paths:
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) < 5 or not parts[-1].startswith("v"):
            continue
        try:
            version = int(parts[-1][1:])
        except ValueError:
            continue
        key = (parts[2], parts[3][:2])
        if key not in best or version > best[key][0]:
            best[key] = (version, path)
    return {k: v[1] for k, v in best.items()}


def first_events_columns(path):
    """最大版本的第一个 EVENTS 表的列数与行数。读不开返回 (-1, -1)。"""
    try:
        with fits.open(path, memmap=True) as hdus:
            for hdu in hdus:
                if not hdu.name.startswith("EVENTS"):
                    continue
                columns = len(hdu.columns.names) if hdu.columns else 0
                rows = len(hdu.data) if hdu.data is not None else 0
                return columns, rows
    except Exception:
        return -1, -1
    return 0, 0


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    grd = latest_per_hour(sorted(glob.glob(f"{ROOT}/*/*/*/GRD_EVT/*_v*.fits")))
    cpd = latest_per_hour(sorted(glob.glob(f"{ROOT}/*/*/*/CPD_EVT/*_v*.fits")))
    print(f"GRD 小时 {len(grd)}，CPD 小时 {len(cpd)}", flush=True)

    writer = None
    if out:
        writer = csv.writer(open(out, "w", newline=""), lineterminator="\n")
        writer.writerow(["yymmdd", "hour", "kind", "bytes", "columns", "rows"])

    tally = collections.Counter()
    stub_cpd_with_good_grd = []
    for i, (key, path) in enumerate(sorted(cpd.items())):
        size = os.path.getsize(path)
        # 只点小文件，大文件不可能是空壳；阈值取得很松
        if size > 200_000:
            tally["CPD 正常(按大小)"] += 1
            continue
        columns, rows = first_events_columns(path)
        if writer:
            writer.writerow([key[0], key[1], "CPD", size, columns, rows])
        if columns == 0:
            tally["CPD 空壳(零列)"] += 1
            gpath = grd.get(key)
            if gpath and os.path.getsize(gpath) > 1_000_000:
                stub_cpd_with_good_grd.append(key)
        elif columns < 0:
            tally["CPD 读不开"] += 1
        elif rows == 0:
            tally["CPD 有列但零行"] += 1
        else:
            tally["CPD 小但有数据"] += 1
        if (i + 1) % 2000 == 0:
            print(f"  CPD {i+1}/{len(cpd)}", flush=True)

    for key, path in sorted(grd.items()):
        size = os.path.getsize(path)
        if size > 200_000:
            tally["GRD 正常(按大小)"] += 1
            continue
        columns, rows = first_events_columns(path)
        if writer:
            writer.writerow([key[0], key[1], "GRD", size, columns, rows])
        if columns == 0:
            tally["GRD 空壳(零列)"] += 1
        elif columns < 0:
            tally["GRD 读不开"] += 1
        elif rows == 0:
            tally["GRD 有列但零行"] += 1
        else:
            tally["GRD 小但有数据"] += 1

    print("\n统计:")
    for key, count in tally.most_common():
        print(f"  {key}: {count}")
    print(f"\n**CPD 是空壳、而同一小时 GRD 是好文件（>1 MB）的小时数: "
          f"{len(stub_cpd_with_good_grd)}**")
    print("  这些就是「好 GRD 数据被 CPD 占位文件拖死」的小时，逐条：")
    for key in stub_cpd_with_good_grd[:60]:
        print(f"    20{key[0][:2]}-{key[0][2:4]}-{key[0][4:6]} {key[1]}h")
    if len(stub_cpd_with_good_grd) > 60:
        print(f"    …… 共 {len(stub_cpd_with_good_grd)} 个")


if __name__ == "__main__":
    main()
