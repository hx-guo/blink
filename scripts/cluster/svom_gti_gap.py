"""GTI 段**内部**有没有未标出的数据丢失。

第 11 条把「本底窗伸进 GTI 缺口」修掉了（本底窗夹到候选所在的活时间段、分母只数
活时间）。但那只覆盖 **GTI 标出来的**缺口。HXMT 那边的缺陷是本底窗半边落在数据
缺口上 ⇒ λ 低估约 2 倍 ⇒ 几十毫秒内一串本底涨落同时越线。**若 SVOM 的 GTI 段内部
还有没标出来的丢数，夹取抓不到，同一类缺陷就仍然在。**

判据：逐 GTI 段量段内事例的**最大间隔**，与该段的平均事例间隔比。纯泊松下最大间隔
约是平均间隔的 ln(N) 倍；实测远大于它就是未标缺口。

用法: python3 gti_gap.py <YYYY> <MM> <DD> [更多 DD ...]
"""
import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"


def newest(paths):
    best = {}
    for p in paths:
        key = p.rsplit("_v", 1)[0]
        if key not in best or p > best[key]:
            best[key] = p
    return sorted(best.values())


def scan_day(y, m, d):
    pat = "%s/%s/%s/%s/grm_evt/*.fits" % (ROOT, y, m, d)
    files = newest(glob.glob(pat))
    if not files:
        print("  %s-%s-%s 没有文件" % (y, m, d))
        return []
    out = []
    for f in files:
        with fits.open(f, memmap=True) as hd:
            gti = None
            for h in hd:
                if h.name and "GTI" in h.name.upper():
                    gti = np.array([(float(r[0]), float(r[1])) for r in h.data])
            ts = []
            for h in hd:
                if h.name and "EVENT" in h.name.upper() and h.data is not None:
                    col = [c for c in h.data.columns.names if c.upper() in ("TIME", "MET")]
                    if col:
                        ts.append(np.asarray(h.data[col[0]], float))
            if gti is None or not ts:
                continue
            t = np.sort(np.concatenate(ts))
        for a, b in gti:
            k = t[(t >= a) & (t <= b)]
            if len(k) < 200 or (b - a) < 5.0:
                continue
            dt = np.diff(k)
            mx = dt.max()
            mean = (b - a) / len(k)
            out.append((f.rsplit("/", 1)[-1], a, b - a, len(k), mean, mx, mx / mean))
    return out


rows = []
y, m = sys.argv[1], sys.argv[2]
for d in sys.argv[3:]:
    r = scan_day(y, m, d)
    print("  %s-%s-%s：%d 个 GTI 段" % (y, m, d, len(r)))
    rows += r
if not rows:
    sys.exit("没有可用的 GTI 段")
ratio = np.array([r[6] for r in rows])
mx = np.array([r[5] for r in rows])
n = np.array([r[3] for r in rows])
expect = np.log(n)          # 纯泊松下最大间隔 / 平均间隔 的量级
print("\n共 %d 个 GTI 段（>= 200 事例且 >= 5 s）。" % len(rows))
print("段内最大事例间隔 / 段内平均间隔：中位 %.1f，90%% %.1f，最大 %.1f"
      % (np.median(ratio), np.percentile(ratio, 90), ratio.max()))
print("纯泊松期望（ln N）：中位 %.1f。实测/期望 中位 %.2f，最大 %.2f"
      % (np.median(expect), np.median(ratio / expect), (ratio / expect).max()))
print("段内最大间隔的绝对值：中位 %.4f s，90%% %.4f s，最大 %.4f s"
      % (np.median(mx), np.percentile(mx, 90), mx.max()))
bad = [r for r in rows if r[5] > 0.1]
print("\n段内最大间隔 > 0.1 s 的段（本底窗半宽 0.5 s，这个尺度才够压低 λ）：%d 个 = %.2f%%"
      % (len(bad), 100.0 * len(bad) / len(rows)))
for r in sorted(bad, key=lambda x: -x[5])[:8]:
    print("  %s  GTI 起 %.1f 长 %.1f s  N=%d  最大间隔 %.3f s（平均 %.2e s，%.0f 倍）"
          % (r[0], r[1], r[2], r[3], r[5], r[4], r[6]))
