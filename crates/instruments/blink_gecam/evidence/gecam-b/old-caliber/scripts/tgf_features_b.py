"""147 个已发表 GECAM-B TGF 的事例级特征。

口径直接从统筹的 `gecam_features.py` 里 import（read_events / hour_file / met /
span），保证与 A、C 两星逐字一致：先准入、后按同探头同时戳合并双增益。
在它的基础上多算四样：

* `f3`   —— 落在 >=3 重同戳簇里的计数占 n_core 的比例
* `pi_ge_368` —— 窗内事例落在 ch368 以上的比例（低增益支路超量程堆积区）
* `cpd_env`   —— CPD 环境率（候选两侧各 1 s，秒计数）
* `gti_clip`  —— 基线窗 [t0-1s, t1+1s] 有没有被 GTI 切到，切掉多少秒

用法: python3 tgf_features_b.py <out.csv>
"""
import csv, json, sys
sys.path.insert(0, "/scratchfs2/gecam/guohx")
import numpy as np
from astropy.io import fits
from gecam_features import read_events, hour_file, met, span, CPD_HALF_WIDTHS

SAT = "GECAM-B"
RUN = "/scratchfs2/gecam/guohx/gecambrun"

def gti_of(path):
    with fits.open(path) as hdus:
        g = hdus["GTI"].data
        return np.asarray(g["START"], float), np.asarray(g["STOP"], float)

def gti_seconds(starts, stops, a, b):
    return float(np.clip(np.minimum(stops, b) - np.maximum(starts, a), 0, None).sum())

sigs = json.load(open(RUN + "/tgf_signals.json"))
# 分片：按所在小时取模，同一小时永远落在同一个分片里，事例只读一遍
if len(sys.argv) > 3:
    shard, nshard = int(sys.argv[2]), int(sys.argv[3])
    hours = sorted({x["start"][:13] for x in sigs})
    mine = {h for i, h in enumerate(hours) if i % nshard == shard}
    sigs = [x for x in sigs if x["start"][:13] in mine]
print("这一片的候选 %d 个" % len(sigs))
rows = []
grd = cpd = gti = None
key = None
mismatch = 0
for s in sorted(sigs, key=lambda x: x["start"]):
    iso = s["start"]
    t0 = met(iso, SAT) + s["delay"]
    t1 = t0 + s["bin_size_best"]
    k = iso[:13]
    if k != key:
        key = k
        p = hour_file(SAT, iso, "grd")
        grd = read_events(p, want_pi=True) if p else None
        gti = gti_of(p) if p else None
        pc = hour_file(SAT, iso, "cpd")
        cpd = read_events(pc, want_pi=False) if pc else None
    if grd is None: continue
    time, channel, detector = grd
    lo, hi = span(time, t0, t1)
    n_core = hi - lo
    if n_core != s["count"]: mismatch += 1
    if n_core == 0: continue
    tc, ch, dt_ = time[lo:hi], channel[lo:hi], detector[lo:hi]
    _, mult = np.unique(tc, return_counts=True)
    mf = float(mult[mult >= 2].sum()) / n_core
    f3 = float(mult[mult >= 3].sum()) / n_core
    _, dcnt = np.unique(dt_, return_counts=True)
    a, b = t0 - 1.0, t1 + 1.0
    cpd_counts = {}
    for half in CPD_HALF_WIDTHS:
        if cpd is None: cpd_counts[half] = ""; continue
        clo, chi = span(cpd[0], t0 - half, t1 + half)
        cpd_counts[half] = chi - clo
    if cpd is not None:
        elo, ehi = span(cpd[0], a, b)
        cpd_env = ehi - elo
    else:
        cpd_env = ""
    nominal = b - a
    live = gti_seconds(gti[0], gti[1], a, b) if gti else nominal
    base = s.get("detectors", {}).get("baseline") or []
    rows.append(dict(
        start=iso, fa="%.3e" % s["false_positive_per_year"], count=s["count"], n_core=n_core,
        bin_us="%.3f" % (s["bin_size_best"] * 1e6),
        n_det_hit=len(dcnt), det_frac_max="%.3f" % (dcnt.max() / n_core),
        max_mult=int(mult.max()), multiplet_frac="%.3f" % mf, f3="%.3f" % f3,
        pi_med=int(np.median(ch)), pi_ge_368="%.3f" % float((ch >= 368).mean()),
        n_acd=(s.get("acd") or {}).get("n_acd", ""),
        n_acd_multi=(s.get("acd") or {}).get("n_acd_multi", ""),
        n_acd_bg=(s.get("acd") or {}).get("n_acd_bg", ""),
        cpd_10us=cpd_counts.get(1e-5, ""), cpd_100us=cpd_counts.get(1e-4, ""),
        cpd_1ms=cpd_counts.get(1e-3, ""), cpd_10ms=cpd_counts.get(1e-2, ""),
        cpd_env=cpd_env, base_zero=sum(1 for x in base if x == 0), base_len=len(base),
        gti_live_s="%.3f" % live, gti_nominal_s="%.3f" % nominal))
print("对账: n_core != count 的 %d / %d" % (mismatch, len(rows)))
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=list(rows[0].keys()))
w.writeheader(); w.writerows(rows)

A = lambda k: np.array([float(r[k]) for r in rows])
def q(name, a):
    print("  %-14s 中位 %.3f  5%% %.3f  25%% %.3f  75%% %.3f  95%% %.3f  最大 %.3f" % (
        name, np.median(a), *np.percentile(a, [5, 25, 75, 95]), a.max()))
print("147 个真 TGF 的分布（%d 个匹配上）:" % len(rows))
q("bin_us", A("bin_us"))
q("multiplet_frac", A("multiplet_frac"))
print("    mf == 0 的占 %.1f%%,  mf <= 0.25 的占 %.1f%%,  mf <= 0.30 的占 %.1f%%" % (
    100*(A("multiplet_frac")==0).mean(), 100*(A("multiplet_frac")<=0.25).mean(), 100*(A("multiplet_frac")<=0.30).mean()))
q("f3", A("f3"))
print("    f3 == 0 的占 %.1f%%" % (100*(A("f3")==0).mean()))
q("det_frac_max", A("det_frac_max"))
print("    det_frac_max > 0.8 的占 %.1f%%" % (100*(A("det_frac_max")>0.8).mean()))
q("pi_ge_368", A("pi_ge_368"))
q("n_det_hit", A("n_det_hit"))
nacd = A("n_acd"); ncore = A("n_core")
print("  CPD: 窗内有计数的占 %.1f%%,  n_acd/n_core 中位 %.4f" % (
    100*(nacd>0).mean(), np.median(nacd/ncore)))
print("  基线窗被 GTI 切到的候选 %d / %d, 活时间中位 %.3f s (标称 %.3f s)" % (
    int((A("gti_live_s") < A("gti_nominal_s") - 1e-6).sum()), len(rows),
    np.median(A("gti_live_s")), np.median(A("gti_nominal_s"))))
print("  detectors.baseline 有零格的候选 %d / %d" % (
    int((A("base_zero") > 0).sum()), len(rows)))
