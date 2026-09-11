"""逐 TGF 定位两档增益的交接道，供能谱拟合把交接处的坏道挡掉。

交接处不是"平滑过渡"：低增益支在某道戛然而止、高增益支从下一道开始，中间
留 1–3 个几乎空的道（2025 年起干脆整道为零）。CALDB 的响应不知道这件事，
于是折叠模型在那几道上系统性高估。交接道随时间漂，所以必须逐 TGF 定位。

输出每个 TGF 所在小时的：低增益支最高道、高增益支最低道、以及交接处
逐道的低/高增益计数（用整小时本底，统计量足够）。

用法: python3 svom_seam_per_tgf.py <incidence.csv> <out.csv> [WORKERS] [IDX]
"""
import csv
import datetime as dt
import glob
import sys

import numpy as np
from astropy.io import fits

D = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
REF = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
NCH = 256


def latest(pat):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    seen = {}
    for f in g:
        seen[f.rsplit("_v", 1)[0]] = f
    return sorted(seen.values())[-1]


def main(inc, out, workers=1, idx=0):
    rows = list(csv.DictReader(open(inc)))
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["start", "hour", "seam_lo", "seam_hi"]
               + ["r%d" % k for k in range(-6, 7)])
    for i, r in enumerate(rows):
        if i % workers != idx:
            continue
        s = r["start"]
        t = dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        f = latest("%s/%s/grm_evt/svom_grm_evt_%s_%s_v*.fits"
                   % (D, t.strftime("%Y/%m/%d"), t.strftime("%y%m%d"), t.strftime("%H")))
        if f is None:
            print("缺文件", s, flush=True)
            continue
        h0 = np.zeros(NCH, np.int64)
        h1 = np.zeros(NCH, np.int64)
        with fits.open(f) as hd:
            for j in (3, 4, 5):
                d = hd[j].data
                pi = np.asarray(d["PI"])
                k = ((np.asarray(d["EVT_TYPE"]) == 0) & (np.asarray(d["ANTI_COIN"]) == 0)
                     & (pi >= 0) & (pi < NCH))
                gt = np.asarray(d["GAIN_TYPE"])
                h0 += np.bincount(pi[k & (gt == 0)], minlength=NCH)[:NCH]
                h1 += np.bincount(pi[k & (gt == 1)], minlength=NCH)[:NCH]
        # 稳健的交接道：忽略零星越界事例，取累计 0.1% 分位
        c0 = np.cumsum(h0)
        c1 = np.cumsum(h1[::-1])[::-1]
        seam_lo = int(np.searchsorted(c0, c0[-1] * 0.99999))     # 低增益支实际末道
        seam_hi = int(NCH - 1 - np.searchsorted(c1[::-1], c1[0] * 0.99999))
        tot = (h0 + h1).astype(float)
        # 交接处逐道相对「两侧各 10 道的对数线性外推」的比值
        ref = np.zeros(13)
        for n, ch in enumerate(range(seam_lo - 6, seam_lo + 7)):
            if ch < 12 or ch > NCH - 13:
                ref[n] = -1
                continue
            k = np.array([c for c in range(ch - 12, ch + 13)
                          if abs(c - seam_lo) > 4 and 0 < c < NCH and tot[c] > 0])
            if len(k) < 8:
                ref[n] = -1
                continue
            p = np.polyfit(np.log(k.astype(float)), np.log(tot[k]), 1,
                           w=np.sqrt(tot[k]))
            ref[n] = tot[ch] / np.exp(np.polyval(p, np.log(float(ch))))
        w.writerow([s, t.strftime("%Y-%m-%dT%H"), seam_lo, seam_hi]
                   + ["%.3f" % v for v in ref])
        fh.flush()
        print("%3d %s seam %d/%d" % (i, s[:19], seam_lo, seam_hi), flush=True)
    fh.close()
    print("→", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0)
