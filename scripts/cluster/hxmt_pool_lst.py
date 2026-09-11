#!/usr/bin/env python3
"""从 tgfs.json 流式抽全池候选的 时间 + 位置 + 显著性 + 关联 + 列车 六列。

行扫描，与 extract_tgfs_v6.py 同一套约定（json 是缩进过的，每个字段独占一行）。
位置是 LST 检验唯一需要的额外量。
"""
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/v6run/tgfs.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/scratchfs2/gecam/guohx/v6run/pool_lst.csv"

n = 0
start = fpy = lon = lat = assoc = nb = None
in_position = False
with open(SRC) as f, open(OUT, "w") as out:
    out.write("start,fpy,lon,lat,assoc,neighbors_10min,is_train\n")
    for line in f:
        line = line.strip()
        if line.startswith('"start"'):
            start = line.split('"')[3]
        elif line.startswith('"false_positive_per_year"'):
            fpy = line.split(":")[1].strip().rstrip(",")
        elif line.startswith('"position"'):
            in_position = True
        elif in_position and line.startswith('"longitude"'):
            lon = line.split(":")[1].strip().rstrip(",")
        elif in_position and line.startswith('"latitude"'):
            lat = line.split(":")[1].strip().rstrip(",")
            in_position = False
        elif line.startswith('"associated"'):
            assoc = "true" in line
        elif line.startswith('"neighbors_10min"'):
            nb = line.split(":")[1].strip().rstrip(",")
        elif line.startswith('"is_train"'):
            n += 1
            out.write("%s,%s,%s,%s,%d,%s,%d\n"
                      % (start, fpy, lon, lat, int(assoc), nb, int("true" in line)))
            start = fpy = lon = lat = assoc = nb = None
print("wrote %d candidates -> %s" % (n, OUT))
