"""GECAM-C：`EVT_TYPE == 3` 是什么（OPEN-QUESTIONS 第 3 条）。

现在的 keep 只认 1，所以 3 被丢掉——安全，但不知道丢的是什么。
只读需要的三四列，不做 O(n log n) 的联合去重，几个小时能跑完。
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"

for spec in sys.argv[1:]:
    day, hour = spec.split("@")
    files = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{int(hour):02d}_v*.fits"))
    if not files:
        print("缺", spec)
        continue
    print(f"\n########## {spec}  {files[-1]}", flush=True)
    with fits.open(files[-1], memmap=True) as hdus:
        gti = None
        for hdu in hdus:
            if hdu.name == "GTI":
                gti = (np.asarray(hdu.data["START"], float), np.asarray(hdu.data["STOP"], float))
        tot = {}
        per_det = {}
        pi3, t3, g3, f3 = [], [], [], []
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            det = int(hdu.name[-2:])
            d = hdu.data
            evt = np.asarray(d["EVT_TYPE"]).astype(np.int8)
            for v, c in zip(*np.unique(evt, return_counts=True)):
                tot[int(v)] = tot.get(int(v), 0) + int(c)
            m = evt == 3
            per_det[det] = (int(m.sum()), int(evt.size))
            if m.any():
                pi3.append(np.asarray(d["PI"])[m].astype(int))
                t3.append(np.asarray(d["TIME"], float)[m])
                g3.append(np.asarray(d["GAIN_TYPE"])[m].astype(int))
                f3.append(np.asarray(d["FLAG"])[m].astype(int))
        n = sum(tot.values())
        print("  事例 %d；EVT_TYPE 分布: %s" % (n, {k: "%d (%.4f%%)" % (v, v / n * 100) for k, v in sorted(tot.items())}))
        if not pi3:
            print("  本小时没有 EVT_TYPE==3")
            continue
        pi = np.concatenate(pi3)
        tm = np.concatenate(t3)
        gg = np.concatenate(g3)
        ff = np.concatenate(f3)
        print("  EVT_TYPE==3 共 %d（%.5f%%）" % (pi.size, pi.size / n * 100))
        print("  逐路：", {d: "%d/%d (%.5f%%)" % (a, b, a / b * 100) for d, (a, b) in sorted(per_det.items()) if a})
        print("  PI：中位 %d，min %d max %d，>=448 占 %.1f%%，落在 keep 窗 [54,448) 的占 %.1f%%"
              % (np.median(pi), pi.min(), pi.max(), (pi >= 448).mean() * 100,
                 ((pi >= 54) & (pi < 448)).mean() * 100))
        print("  GAIN_TYPE:", dict(zip(*[a.tolist() for a in np.unique(gg, return_counts=True)])))
        print("  FLAG:", dict(zip(*[a.tolist() for a in np.unique(ff, return_counts=True)])))
        if gti is not None:
            inside = np.zeros(tm.size, bool)
            for s, e in zip(*gti):
                inside |= (tm >= s) & (tm <= e)
            print("  落在 GTI 内 %.1f%%" % (inside.mean() * 100))
        o = np.sort(tm)
        if o.size > 1:
            gaps = np.diff(o)
            print("  相邻间隔：中位 %.4g s，<1 ms 占 %.1f%%，<1 µs 占 %.1f%%，跨度 %.1f s"
                  % (np.median(gaps), (gaps < 1e-3).mean() * 100, (gaps < 1e-6).mean() * 100, o[-1] - o[0]))
