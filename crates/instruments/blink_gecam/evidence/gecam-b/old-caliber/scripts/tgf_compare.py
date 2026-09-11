"""把我们的 GECAM-B 候选与 147 个已发表 TGF（Zhao et al. 2023 GRL）逐个匹配。

两边都是 UTC ISO 字符串，直接按 UTC 秒比，不经过 MET，也就不涉及历元和闰秒。
小数秒整取（GECAM 时戳 0.03 µs，截到微秒会错位几十个窗宽）。

对账窗用最佳格：t0 = start + delay，t1 = t0 + bin_size_best。

用法: python3 tgf_compare.py <out.csv>
"""
import csv, glob, json, sys, datetime as dt
import numpy as np

RUN = "/scratchfs2/gecam/guohx/gecambrun"
CAT = RUN + "/gecam_tgf_catalog.csv"
TOL_S = 0.01

def utc_seconds(iso):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return stamp.timestamp() + (float("0." + frac) if frac else 0.0)

cat = []
for row in csv.DictReader(open(CAT)):
    cat.append(dict(ut=row["UT"], t=utc_seconds(row["UT"]),
                    lon=float(row["Longitude_deg"]), lat=float(row["Latitude_deg"]),
                    dur_us=float(row["Duration_us"]), net=float(row["NetCounts"]),
                    hr=float(row["HardnessRatio_200keV"]), cpd_ratio=float(row["CPDtoGRDcountsRatio"])))
print("目录 %d 个 TGF" % len(cat))

ours = []
for f in sorted(glob.glob(RUN + "/data/GECAM-B/*/*/*_signals.json")):
    for s in json.load(open(f)):
        t0 = utc_seconds(s["start"]) + s["delay"]
        ours.append(dict(raw=s, t0=t0, t1=t0 + s["bin_size_best"], start=utc_seconds(s["start"]),
                         fa=s["false_positive_per_year"], count=s["count"], mean=s["mean"],
                         bin_us=s["bin_size_best"] * 1e6,
                         lon=s["position"]["longitude"], lat=s["position"]["latitude"],
                         acd=s.get("acd") or {}, det=s.get("detectors") or {}))
print("我们的候选 %d 个" % len(ours))

# 逐小时账本：这个小时到底搜没搜，以及诊断量
searched, excluded, metrics_sum = 0, {}, {}
sec = 0.0
n_sig_total = 0
for f in sorted(glob.glob(RUN + "/data/GECAM-B/*/*/*_hours.json")):
    rep = json.load(open(f))
    for h in rep["hours"]:
        if h["status"] == "searched":
            searched += 1; sec += h["searched_seconds"]; n_sig_total += h["n_signals"]
            for k, v in (h.get("metrics") or {}).items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        else:
            r = h.get("reason") or "?"
            excluded[str(r)] = excluded.get(str(r), 0) + 1
print("账本: searched %d 小时, 曝光 %.0f s (%.2f 天); 候选 %d 个" % (searched, sec, sec/86400, n_sig_total))
print("  excluded 原因:", excluded)
print("  诊断量合计:", {k: round(v, 1) for k, v in sorted(metrics_sum.items())})
if metrics_sum.get("n_events"):
    print("  双增益重复计数占比: %.4f%%" % (100 * metrics_sum.get("merged_gain_duplicates", 0) / metrics_sum["n_events"]))
if sec > 0:
    print("  候选率: %.0f 个/天(曝光)" % (n_sig_total / (sec / 86400)))

if not ours:
    print("没有候选"); sys.exit(0)

t0s = np.array([o["t0"] for o in ours]); order = np.argsort(t0s)
t0s = t0s[order]; ours = [ours[i] for i in order]

rows = []
ours_matched = []
for c in cat:
    i = np.searchsorted(t0s, c["t"]); best = None
    for j in range(max(0, i - 3), min(len(t0s), i + 4)):
        if abs(t0s[j] - c["t"]) <= TOL_S:
            if best is None or ours[j]["fa"] < ours[best]["fa"]: best = j
    o = ours[best] if best is not None else None
    if o is not None: ours_matched.append(o)
    acd = o["acd"] if o else {}
    rows.append(dict(ut=c["ut"], dur_us="%.1f" % c["dur_us"], net="%.1f" % c["net"],
                     hr="%.3f" % c["hr"], cat_cpd="%.3f" % c["cpd_ratio"],
                     lat="%.2f" % c["lat"], lon="%.2f" % c["lon"],
                     found=int(o is not None),
                     our_fa=("%.3e" % o["fa"]) if o else "",
                     our_count=o["count"] if o else "",
                     our_bin_us=("%.1f" % o["bin_us"]) if o else "",
                     dt_ms=("%.4f" % ((o["t0"] - c["t"]) * 1e3)) if o else "",
                     n_acd=acd.get("n_acd", "") if o else "",
                     n_acd_multi=acd.get("n_acd_multi", "") if o else "",
                     n=acd.get("n", "") if o else ""))
# 把匹配上的候选原样存一份，供后续特征脚本用
matched = [o["raw"] for o in ours_matched]
json.dump(matched, open(RUN + "/tgf_signals.json", "w"))
print("匹配上的候选原样存了 %d 个 -> tgf_signals.json" % len(matched))

out = sys.argv[1]
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0].keys()))
w.writeheader(); w.writerows(rows)

found = np.array([r["found"] for r in rows], bool)
print("完备性: %d/%d = %.1f%%" % (found.sum(), len(rows), 100 * found.mean()))
fa = np.array([float(r["our_fa"]) if r["our_fa"] else np.nan for r in rows])
for thr, name in ((1e-5, "fa<=1e-5"), (1.0, "fa<=1"), (20.0, "fa<=20")):
    n = int(np.nansum(fa <= thr))
    print("  达显著 %-9s %d/%d 找回的 = %.1f%%   占全部 147 = %.1f%%" % (
        name, n, found.sum(), 100 * n / max(found.sum(), 1), 100 * n / len(rows)))
dt_ms = np.array([float(r["dt_ms"]) for r in rows if r["dt_ms"]])
if dt_ms.size:
    print("时刻差 dt (我们的最佳格起点 − 目录 UT), ms: 中位 %.4f  5%% %.4f  95%% %.4f" % (
        np.median(dt_ms), np.percentile(dt_ms, 5), np.percentile(dt_ms, 95)))
miss = [r for r in rows if not r["found"]]
print("没找回的 %d 个:" % len(miss))
for r in miss[:30]:
    print("   %s  dur %s us  net %s  |lat| %.1f" % (r["ut"], r["dur_us"], r["net"], abs(float(r["lat"]))))
