"""dt 的来源：是"目录 UT 给起点"还是"匹配器挑错了候选"。

`gb_recall.py` 在 ±10 ms 内取**最显著**的候选。一个亮暴会产生一串候选，
"最显著"未必是最早的那个——若 dt 随亮度走只是因为亮暴的候选串更长，那它是
匹配口径造成的，不是物理。四种挑法并排量，让口径差自己显出来。
"""
import csv, json, os, datetime as dt
import numpy as np
from scipy import stats

EPOCH = (2019, 1, 1)
MS = 10.0


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    s = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (s - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


cat = list(csv.DictReader(open("/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")))
days = sorted({c["UT"][:10] for c in cat})
hours = {c["UT"][:13] for c in cat}
pool = {}
for d in days:
    stem = d.replace("-", "")
    p = "data/GECAM-B/%s/%s/%s_signals.json" % (d[:4], d[5:7], stem)
    if not os.path.exists(p):
        continue
    for s in json.load(open(p)):
        k = s["start"][:13]
        if k not in hours:
            continue
        t0 = met(s["start"]) + s["delay"]
        pool.setdefault(k, []).append((t0, s["false_positive_per_year"], s["count"], s["bin_size_best"]))

res = []
for c in cat:
    ut, k = met(c["UT"]), c["UT"][:13]
    near = [x for x in pool.get(k, []) if abs(x[0] - ut) * 1e3 <= MS]
    if not near:
        continue
    d = np.array([(x[0] - ut) * 1e6 for x in near])
    fa = np.array([x[1] for x in near])
    cnt = np.array([x[2] for x in near])
    res.append(dict(nnear=len(near), dt_sig=d[np.argmin(fa)], dt_near=d[np.argmin(np.abs(d))],
                    dt_first=d.min(), dt_cnt=d[np.argmax(cnt)],
                    dur=float(c["Duration_us"]), net=float(c["NetCounts"])))

A = lambda k: np.array([r[k] for r in res])
print("n =", len(res))
for k in ("dt_sig", "dt_near", "dt_first", "dt_cnt"):
    v = A(k)
    print(" %-8s 中位 %+8.1f us  5-95%% %+8.1f .. %+8.1f  >0 %5.1f%%  ~NetCounts %+.3f  ~Duration %+.3f" % (
        k, np.median(v), np.percentile(v, 5), np.percentile(v, 95), (v > 0).mean() * 100,
        stats.spearmanr(v, A("net")).statistic, stats.spearmanr(v, A("dur")).statistic))
n = A("nnear")
print(" 窗内候选数: 中位 %.0f  5-95%% %.0f..%.0f  ~NetCounts %+.3f" % (
    np.median(n), np.percentile(n, 5), np.percentile(n, 95), stats.spearmanr(n, A("net")).statistic))
