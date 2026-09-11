"""17 天的样本上重做两件事：天格 B 角同型的 CPD 判定，和 ±30 s 成串阈的验证。

只用 tgfs.json 里已有的字段（position / acd / detectors / bin_size_best），不回事例流，
所以谱那一维缺席——用"窗长顶到 1 ms 上限 + 高磁纬 + 多路全亮 + 亮"这几条代替。
"""

import datetime as dt
import json
import sys

import numpy as np

rows = json.load(open(sys.argv[1]))
sig = [r["signal"] for r in rows]
lat = np.array([s["position"]["latitude"] for s in sig], float)
lon = np.array([s["position"]["longitude"] for s in sig], float)
cnt = np.array([s["count"] for s in sig], float)
width = np.array([s["bin_size_best"] for s in sig], float) * 1e6
fa = np.array([s["false_positive_per_year"] for s in sig], float)
nacd = np.array([s["acd"]["n_acd"] for s in sig], float)
nacdbg = np.array([s["acd"]["n_acd_bg"] for s in sig], float)
nbg = np.array([s["acd"]["n_bg"] for s in sig], float)
n = np.array([s["acd"]["n"] for s in sig], float)
W = np.array([s["detectors"]["window"] for s in sig], float)
assoc = np.array([r["lightning"]["associated"] for r in rows])
prob = np.array([r["lightning"]["coincidence_probability"] for r in rows], float)
nlit = (W > 0).sum(axis=1)
share = W.max(axis=1) / np.maximum(W.sum(axis=1), 1)
rate = nacdbg / 2.0

P1, P2 = np.radians(80.65), np.radians(-72.68)
la, lo = np.radians(lat), np.radians(lon)
amlat = np.abs(np.degrees(np.arcsin(np.clip(np.sin(la) * np.sin(P1)
                                            + np.cos(la) * np.cos(P1) * np.cos(lo - P2), -1, 1))))

start = [s["start"] for s in sig]
ref = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)


def secs(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    st = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (st - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


t = np.array([secs(s) for s in start])
order = np.argsort(t)
ts = t[order]

print("候选 %d，跨 %d 天" % (len(rows), len({s[:10] for s in start})))

print("\n=== 天格 B 角同型（17 天）===")
print("  CPD 窗内命中率的全样本基准 %.1f%%" % ((nacd > 0).mean() * 100))
for label, m in (("窗长顶上限 >950 µs", width > 950),
                 ("+ |mlat| >= 33°", (width > 950) & (amlat >= 33)),
                 ("+ 计数 >= 20", (width > 950) & (amlat >= 33) & (cnt >= 20)),
                 ("计数>=20 且 |mlat|>=33（不限窗长）", (cnt >= 20) & (amlat >= 33)),
                 ("计数>=20 且 |mlat|<33", (cnt >= 20) & (amlat < 33)),
                 ("计数 8–12 且 |mlat|>=33", (cnt >= 8) & (cnt <= 12) & (amlat >= 33))):
    if m.sum() < 3:
        print("  %-36s n=%4d 样本不足" % (label, m.sum()))
        continue
    # episode 数（10 s 聚簇）
    tt = np.sort(t[m])
    eps = 1 + int((np.diff(tt) > 10).sum()) if tt.size > 1 else 1
    print("  %-36s n=%5d (%3d episode)  CPD命中 %5.1f%%  CPD率中位 %6.0f  亮路数中位 %4.1f/12  "
          "单路占比 %.2f  闪电 %d/%d(期望 %.1f)"
          % (label, m.sum(), eps, (nacd[m] > 0).mean() * 100, np.median(rate[m]),
             np.median(nlit[m]), np.median(share[m]), assoc[m].sum(), m.sum(), prob[m].sum()))

print("\n=== 亮度扫描：CPD 命中率随窗内计数（|mlat| >= 33）===")
print("  %-14s %6s %8s %10s %10s %8s" % ("计数", "n", "CPD命中", "CPD率中位", "亮路数", "窗长中位"))
for lo_, hi_ in ((8, 10), (10, 15), (15, 20), (20, 30), (30, 60), (60, 100000)):
    m = (cnt >= lo_) & (cnt < hi_) & (amlat >= 33)
    if m.sum() < 5:
        continue
    print("  %3d–%-9d %6d %7.1f%% %10.0f %10.1f %8.1f"
          % (lo_, hi_, m.sum(), (nacd[m] > 0).mean() * 100, np.median(rate[m]),
             np.median(nlit[m]), np.median(width[m])))
print("  低磁纬对照 |mlat| < 33：")
for lo_, hi_ in ((8, 10), (10, 15), (15, 20), (20, 30), (30, 100000)):
    m = (cnt >= lo_) & (cnt < hi_) & (amlat < 33)
    if m.sum() < 5:
        continue
    print("  %3d–%-9d %6d %7.1f%% %10.0f %10.1f %8.1f"
          % (lo_, hi_, m.sum(), (nacd[m] > 0).mean() * 100, np.median(rate[m]),
             np.median(nlit[m]), np.median(width[m])))

print("\n=== ±30 s 成串阈在 17 天池子上的表现 ===")
density = len(ts) / (ts[-1] - ts[0])
print("  跨度 %.1f 天，池率 %.4f /s（单日 0.0392）" % ((ts[-1] - ts[0]) / 86400, density))
for Wsec in (0.5, 30.0, 60.0, 600.0):
    lo_i = np.searchsorted(ts, ts - Wsec, "left")
    hi_i = np.searchsorted(ts, ts + Wsec, "right")
    nb_s = hi_i - lo_i - 1
    nb = np.empty_like(nb_s)
    nb[order] = nb_s
    # 池率按天算（跨天的空隙会把全局密度算低）
    print("  ±%-6g 邻居数中位 %4.0f  95分位 %4.0f  最大 %5d" % (Wsec, np.median(nb), np.percentile(nb, 95), nb.max()))
    for name, thr in (("泊松+5σ(按当天池率)", None), ("全池95分位", np.percentile(nb, 95))):
        if thr is None:
            continue
        m = nb > thr
        print("      阈 %-6.0f(%s) 摘 %5d (%.1f%%)  被摘者 CPD命中 %.1f%% vs 留下 %.1f%%  "
              "闪电 摘走 %d / 留下 %d"
              % (thr, name, m.sum(), m.mean() * 100, (nacd[m] > 0).mean() * 100,
                 (nacd[~m] > 0).mean() * 100, assoc[m].sum(), assoc[~m].sum()))
