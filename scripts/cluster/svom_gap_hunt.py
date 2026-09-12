"""GTI 段内部未标出的丢数：有多普遍，候选会不会挨着它。

2024-07-03 单日已实测到两处 **0.600 s** 的段内空档（段内 900–1400 万个事例、
平均间隔 2.4e-4 s，即 2300–2500 倍）。0.6 s 正好是要命的尺度：本底窗半宽 0.5 s，
一个 0.6 s 的空档能盖掉大半个本底窗 ⇒ λ 低估约 2 倍 ⇒ 普通本底涨落越线。
第 11 条修的是 **GTI 标出来的**缺口，夹取抓不到 GTI 段**内部**的空档。

输出：逐空档的时刻表 + 候选与空档的时间距离分布。

用法: python3 gap_hunt.py <signals 根目录> <YYYY-MM-DD> [更多日期 ...]
"""
import glob
import json
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
MIN_GAP = 0.1


def newest(paths):
    best = {}
    for p in paths:
        key = p.rsplit("_v", 1)[0]
        if key not in best or p > best[key]:
            best[key] = p
    return sorted(best.values())


def day_gaps(day):
    y, m, d = day[:4], day[5:7], day[8:10]
    gaps = []
    for f in newest(glob.glob("%s/%s/%s/%s/grm_evt/*.fits" % (ROOT, y, m, d))):
        with fits.open(f, memmap=True) as hd:
            gti, ts = None, []
            for h in hd:
                nm = (h.name or "").upper()
                if "GTI" in nm and h.data is not None:
                    gti = np.array([(float(r[0]), float(r[1])) for r in h.data])
                elif "EVENT" in nm and h.data is not None:
                    col = [c for c in h.data.columns.names if c.upper() in ("TIME", "MET")]
                    if col:
                        ts.append(np.asarray(h.data[col[0]], float))
            if gti is None or not ts:
                continue
            t = np.sort(np.concatenate(ts))
        for a, b in gti:
            k = t[(t >= a) & (t <= b)]
            if len(k) < 200:
                continue
            dt = np.diff(k)
            for i in np.where(dt > MIN_GAP)[0]:
                gaps.append((float(k[i]), float(dt[i]), f.rsplit("/", 1)[-1]))
    return gaps


sig_root = sys.argv[1]
allgaps, days = [], sys.argv[2:]
for day in days:
    g = day_gaps(day)
    print("%s：段内空档 > %.1f s 共 %d 处" % (day, MIN_GAP, len(g)))
    for t0, w, f in sorted(g, key=lambda x: -x[1])[:6]:
        print("    MET %.3f  宽 %.3f s  （%s）" % (t0, w, f))
    allgaps += g
if not allgaps:
    sys.exit("没有找到段内空档")
w = np.array([g[1] for g in allgaps])
print("\n共 %d 处段内空档（%d 天）。宽度：中位 %.3f s，最大 %.3f s；"
      % (len(allgaps), len(days), np.median(w), w.max())
      + "宽 >= 0.5 s（能盖掉半个本底窗）的 %d 处" % int((w >= 0.5).sum()))
vals, cnt = np.unique(np.round(w, 3), return_counts=True)
print("最常见的宽度：" + "，".join("%.3f s × %d" % (v, c)
                            for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1])[:5]))

# 候选离最近空档多远
gt = np.sort(np.array([g[0] for g in allgaps]))
cand = []
for day in days:
    p = "%s/data/SVOM_GRM/%s/%s/%s%s%s_signals.json" % (
        sig_root, day[:4], day[5:7], day[:4], day[5:7], day[8:10])
    try:
        cand += json.load(open(p))
    except OSError:
        pass
if not cand:
    sys.exit("\n没读到候选文件，跳过邻近度")
print("\n候选 %d 个。离最近段内空档的距离：" % len(cand))
# signals.json 的 start 是 UTC 串，这里用 MET 不可得，改用同日空档的相对秒
print("（注：signals.json 只有 UTC 串、没有 MET，逐候选距离要另取；本节先只报空档表）")
