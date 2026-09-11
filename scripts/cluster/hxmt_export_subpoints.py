#!/usr/bin/env python3
"""导出一份星下点清单，让别组用自己的分类器跑同一批点。
分类器是唯一的变量：点一样、纬度截断一样，剩下的差别只能来自掩模与缓冲实现。
"""
import csv, sys
import numpy as np

rng = np.random.default_rng(11)
rows = []
cat = list(csv.DictReader(open(sys.argv[1])))
for c in cat:
    if abs(float(c['latitude'])) < 26:
        rows.append(("sig" if c['associated'] != '1' else "assoc",
                     float(c['latitude']), float(c['longitude'])))
bg = []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, ass, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 24 == 0 and abs(float(blat_)) < 26:
            bg.append(("bg", float(blat_), float(blon_)))
idx = rng.choice(len(bg), min(3000, len(bg)), replace=False)
rows += [bg[i] for i in idx]
with open(sys.argv[3], "w") as out:
    out.write("group,latitude,longitude\n")
    for g, la, lo in rows:
        out.write("%s,%.4f,%.4f\n" % (g, la, lo))
print("wrote %d points (%s)" % (len(rows), {g: sum(1 for r in rows if r[0] == g) for g in ("sig", "assoc", "bg")}))
