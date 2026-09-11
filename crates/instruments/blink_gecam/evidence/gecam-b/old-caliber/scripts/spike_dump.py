"""12 个 `multiplet_frac > 0.3` 的真 TGF：我们的最佳格到底抓到了什么？

两种可能，结论完全不同：

* **同戳堆积就是这个 TGF 本身**（亮而短，光子挤进同一个时戳格）——那么
  `multiplet_frac <= 0.3` 这条判据真的会切掉 8.6% 的已发表 TGF。
* **最佳格落在 TGF 旁边的一个粒子尖峰上，不是 TGF 本身**——那么暗端的真实
  完备性比 96.6% 低，而判据切掉的是粒子不是 TGF。

分辨办法：回事例流，把**目录窗** `[UT, UT+Duration_us]` 与**我们的最佳格**
分别摊开，看目录窗里除了尖峰有没有一团铺开的、低同戳的超出。

对照组：`mf <= 0.3` 里目录 NetCounts 最接近的 12 个，同一套量。

用法: python3 spike_dump.py <out.json>
"""
import csv, json, sys, datetime as dt
import numpy as np

sys.path.insert(0, "/scratchfs2/gecam/guohx")
from gecam_features import read_events, hour_file, met, span

RUN = "/scratchfs2/gecam/guohx/gecambrun"
SAT = "GECAM-B"
TOL = 0.01


def utc_seconds(iso):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    s = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=dt.timezone.utc).timestamp()
    return s + (float("0." + frac) if frac else 0.0)


cat = []
for row in csv.DictReader(open(RUN + "/gecam_tgf_catalog.csv")):
    cat.append(dict(ut=row["UT"], t=utc_seconds(row["UT"]),
                    dur=float(row["Duration_us"]) * 1e-6,
                    net=float(row["NetCounts"]),
                    cpd=float(row["CPDtoGRDcountsRatio"])))
cat_t = np.array([c["t"] for c in cat])

sigs = json.load(open(RUN + "/tgf_signals.json"))
items = []
for s in sigs:
    t_best = utc_seconds(s["start"]) + s["delay"]
    i = int(np.argmin(np.abs(cat_t - t_best)))
    if abs(cat_t[i] - t_best) > TOL:
        continue
    items.append((s, cat[i], t_best - cat_t[i]))
print("匹配上 %d 个" % len(items))

# 只读要用的小时：12 个 mf>0.3 的，加 12 个目录 NetCounts 相近的对照
mf_by_start = {r["start"]: float(r["multiplet_frac"])
               for r in csv.DictReader(open(RUN + "/tgf_features_all.csv"))}
susp = [it for it in items if mf_by_start.get(it[0]["start"], 0.0) > 0.30]
rest = [it for it in items if mf_by_start.get(it[0]["start"], 0.0) <= 0.30
        and it[0]["start"] in mf_by_start]
target_net = float(np.median([it[1]["net"] for it in susp]))
ctrl = sorted(rest, key=lambda it: abs(it[1]["net"] - target_net))[:12]
items = susp + ctrl
print("待查 %d 个 (mf>0.3), 对照 %d 个 (net 中位 %.1f 附近)" %
      (len(susp), len(ctrl), target_net))


def mf_of(t):
    if len(t) == 0:
        return 0.0, 0, 0
    _, c = np.unique(t, return_counts=True)
    return float(c[c >= 2].sum()) / len(t), int(c.max()), len(c)


# 先算每个的 mf（用最佳格），据此分组
cache_key, cache_val = None, None


def events_of(iso):
    global cache_key, cache_val
    k = iso[:13]
    if k != cache_key:
        p = hour_file(SAT, iso, "grd")
        cache_val = read_events(p, want_pi=True) if p else None
        cache_key = k
    return cache_val


rows = []
for s, c, dtt in sorted(items, key=lambda x: x[0]["start"]):
    iso = s["start"]
    ev = events_of(iso)
    if ev is None:
        continue
    time, ch, det = ev
    t_best = met(iso, SAT) + s["delay"]
    b = s["bin_size_best"]
    # 目录 UT 的 MET = 最佳格 MET − (最佳格 UTC − 目录 UTC)
    t_cat = t_best - dtt
    lo, hi = span(time, t_best, t_best + b)
    mf_best, mx_best, _ = mf_of(time[lo:hi])
    n_best = hi - lo
    # 目录窗
    clo, chi = span(time, t_cat, t_cat + c["dur"])
    mf_cat, mx_cat, nts_cat = mf_of(time[clo:chi])
    n_cat = chi - clo
    # 目录窗里把同戳簇整簇摘掉之后还剩多少（单戳计数）
    tc = time[clo:chi]
    if len(tc):
        u, cc = np.unique(tc, return_counts=True)
        n_single = int(cc[cc == 1].sum())
        n_multi = int(cc[cc >= 2].sum())
    else:
        n_single = n_multi = 0
    # 本底：目录窗两侧各 1 s，挖掉 ±10 ms
    a0, a1 = t_cat - 1.0, t_cat + c["dur"] + 1.0
    blo, bhi = span(time, a0, a1)
    hlo, hhi = span(time, t_cat - 0.01, t_cat + c["dur"] + 0.01)
    n_bg = (bhi - blo) - (hhi - hlo)
    bg_s = 2.0 - 0.02
    rate = n_bg / bg_s
    exp_cat = rate * c["dur"]
    # 同戳簇摘掉后的单戳超出，与本底的单戳期望比
    tb = time[blo:bhi]
    if len(tb):
        _, cb = np.unique(tb, return_counts=True)
        frac_single_bg = float(cb[cb == 1].sum()) / len(tb)
    else:
        frac_single_bg = 1.0
    rows.append(dict(
        start=iso, ut=c["ut"], dt_ms=round(dtt * 1e3, 4),
        dur_us=round(c["dur"] * 1e6, 1), net=c["net"], cat_cpd=c["cpd"],
        bin_us=round(b * 1e6, 3), n_best=n_best, mf_best=round(mf_best, 3),
        max_best=mx_best,
        n_cat=n_cat, mf_cat=round(mf_cat, 3), max_cat=mx_cat,
        n_single_cat=n_single, n_multi_cat=n_multi,
        bg_rate=round(rate, 1), exp_cat=round(exp_cat, 2),
        exp_single_cat=round(exp_cat * frac_single_bg, 2),
        fa=s["false_positive_per_year"]))

json.dump(rows, open(sys.argv[1], "w"), indent=1)
print("%-27s %7s %6s %6s %6s | %5s %6s %5s | %5s %6s %5s %5s %6s %6s" % (
    "start", "dt_ms", "dur_us", "net", "bin_us",
    "n_bst", "mf_bst", "mxb", "n_cat", "mf_cat", "sing", "mult", "exp", "expS"))
for r in sorted(rows, key=lambda x: -x["mf_best"]):
    print("%-27s %7.3f %6.1f %6.1f %6.1f | %5d %6.3f %5d | %5d %6.3f %5d %5d %6.2f %6.2f" % (
        r["start"][:26], r["dt_ms"], r["dur_us"], r["net"], r["bin_us"],
        r["n_best"], r["mf_best"], r["max_best"],
        r["n_cat"], r["mf_cat"], r["n_single_cat"], r["n_multi_cat"],
        r["exp_cat"], r["exp_single_cat"]))
