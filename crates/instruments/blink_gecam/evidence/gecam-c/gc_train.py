"""GECAM-C：列车密度判据（neighbors_10min）的离线评估。

`blink_wwlln::train_neighbor_counts` 数的是每个候选 ±600 s 内、**全量候选池**里的邻居数。
现行阈是绝对值 34。GECAM-C 一天出 3385 个候选，池率比 HXMT 高得多，先看阈坐在哪。

检验样本：2023-06-15T14:08:35–14:08:59 那一串 22 个候选（同一次 0.63 s 粒子增强），
以及 22:51 那一串。好判据应当把它们整串摘掉，同时不碰高显著的干净幸存者。
"""

import csv
import datetime as dt

import numpy as np

base = "/Users/skyair/Developer/ihep/blink/scratch_gbm/gecam/"
cpd = list(csv.DictReader(open(base + "cpd_norm.csv")))
xd = list(csv.DictReader(open(base + "xdet.csv")))


def col(src, name):
    return np.array([float(r[name]) if r[name] not in ("", None) else np.nan for r in src])


binus, rate, pir, cnt = col(cpd, "bin_us"), col(cpd, "cpd_rate"), col(cpd, "pi_ratio"), col(cpd, "count")
lat = col(cpd, "lat")
obs10 = col(cpd, "obs_10us")
mf, maxd, fa = col(xd, "multiplet_frac"), col(xd, "max_d"), col(xd, "fa")
delay = col(xd, "delay_us") * 1e-6
ndet = col(xd, "n_det_hit")

ref = dt.datetime(2023, 6, 15, tzinfo=dt.timezone.utc)
t = np.array([(dt.datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S.%f")
               .replace(tzinfo=dt.timezone.utc) - ref).total_seconds() for r in cpd]) + delay
order = np.argsort(t)
ts = t[order]

W = 600.0
lo = np.searchsorted(ts, ts - W, "left")
hi = np.searchsorted(ts, ts + W, "right")
nb_sorted = hi - lo - 1
nb = np.empty_like(nb_sorted)
nb[order] = nb_sorted

print("候选池 %d 个 / 1 天。neighbors_10min 分布：" % len(t))
print("  中位 %d，均值 %.1f，四分位 %d–%d，5–95%% %d–%d，最大 %d"
      % (np.median(nb), nb.mean(), *np.percentile(nb, [25, 75]).astype(int),
         *np.percentile(nb, [5, 95]).astype(int), nb.max()))
print("  均匀分布下的期望邻居数 = 池率 × 1200 s = %.1f" % (len(t) / 86400.0 * 1200))
print("  现行绝对阈 34：会摘掉 %d 个（%.1f%%）—— 阈坐在分布的第 %.0f 百分位"
      % ((nb > 34).sum(), (nb > 34).mean() * 100, (nb <= 34).mean() * 100))
new_thr = max(34, round(3.78 * np.median(nb)))
print("  按池中位缩放 threshold = max(34, round(3.78 × 中位 %d)) = %d：摘掉 %d 个（%.1f%%）"
      % (np.median(nb), new_thr, (nb > new_thr).sum(), (nb > new_thr).mean() * 100))

print("\n=== 检验样本一：14:08:28–14:09:01 那一串 ===")
burst = (t > 14 * 3600 + 8 * 60 + 25) & (t < 14 * 3600 + 9 * 60 + 5)
print("  串内候选 %d 个；它们的 neighbors_10min: 中位 %d，范围 %d–%d"
      % (burst.sum(), np.median(nb[burst]), nb[burst].min(), nb[burst].max()))
print("  绝对阈 34 摘掉其中 %d/%d；新阈 %d 摘掉 %d/%d"
      % ((nb[burst] > 34).sum(), burst.sum(), new_thr, (nb[burst] > new_thr).sum(), burst.sum()))
burst2 = (t > 22 * 3600 + 50 * 60 + 55) & (t < 22 * 3600 + 51 * 60 + 25)
print("=== 检验样本二：22:50:55–22:51:25 那一串 ===")
print("  串内候选 %d 个；neighbors_10min 中位 %d，范围 %d–%d；阈 34 摘 %d/%d；新阈 %d 摘 %d/%d"
      % (burst2.sum(), np.median(nb[burst2]), nb[burst2].min(), nb[burst2].max(),
         (nb[burst2] > 34).sum(), burst2.sum(), new_thr, (nb[burst2] > new_thr).sum(), burst2.sum()))

print("\n=== 代价：干净的高显著幸存者会不会被误摘 ===")
surv = mf <= 0.3
clean = surv & (rate <= 500) & (obs10 == 0)
for label, mask in (("幸存 mf<=0.3", surv), ("幸存+CPD双否决", clean),
                    ("幸存+CPD双否决+fa<=1", clean & (fa <= 1)),
                    ("幸存+CPD双否决+fa<=0.01", clean & (fa <= 0.01))):
    if mask.sum() == 0:
        continue
    print("  %-24s n=%4d  邻居中位 %3d  阈34 摘 %3d (%.0f%%)  新阈%d 摘 %3d (%.0f%%)"
          % (label, mask.sum(), np.median(nb[mask]),
             (mask & (nb > 34)).sum(), (nb[mask] > 34).mean() * 100,
             new_thr, (mask & (nb > new_thr)).sum(), (nb[mask] > new_thr).mean() * 100))

print("\n=== 邻居数与候选性质的关系（邻居多 = 粒子？）===")
qs = np.percentile(nb, [0, 25, 50, 75, 90, 100])
for i in range(len(qs) - 1):
    m = (nb >= qs[i]) & (nb <= qs[i + 1])
    if m.sum() < 20:
        continue
    print("  邻居 %3d–%-3d n=%4d  mf<=0.3 占 %4.1f%%  CPD±10µs 命中 %4.1f%%  "
          "CPD率中位 %5.0f  |lat| 中位 %4.1f°  谱比中位 %.2f"
          % (qs[i], qs[i + 1], m.sum(), (mf[m] <= 0.3).mean() * 100, (obs10[m] > 0).mean() * 100,
             np.median(rate[m]), np.median(abs(lat[m])), np.nanmedian(pir[m])))

print("\n=== 天格 B 角形态在 GECAM-C 上有没有同型（T90≈3 ms、谱≈本底、高磁纬、多路均分）===")
POLE_LAT, POLE_LON = np.radians(80.65), np.radians(-72.68)
lon = col(cpd, "lon")
la, lo_ = np.radians(lat), np.radians(lon)
amlat = np.abs(np.degrees(np.arcsin(np.clip(np.sin(la) * np.sin(POLE_LAT)
                                            + np.cos(la) * np.cos(POLE_LAT) * np.cos(lo_ - POLE_LON), -1, 1))))
topped = binus > 950  # 顶到 1 ms 窗上限
print("  窗长顶到上限（>950 µs）的候选 %d 个（%.1f%%）" % (topped.sum(), topped.mean() * 100))
m = topped
print("    这批：|磁纬| 中位 %.1f°，谱比中位 %.2f，点亮探头中位 %.0f/12，"
      "CPD±10µs 命中 %.0f%%，CPD 本底率中位 %.0f c/s，计数中位 %.0f"
      % (np.median(amlat[m]), np.nanmedian(pir[m]), np.median(ndet[m]),
         (obs10[m] > 0).mean() * 100, np.median(rate[m]), np.median(cnt[m])))
grid_like = topped & (amlat >= 33) & (pir < 1.15)
print("  再加 |磁纬|>=33° 且 谱比<1.15（天格 B 角判据）：%d 个" % grid_like.sum())
if grid_like.sum() > 0:
    print("    这批 CPD±10µs 命中 %.0f%%，CPD 本底率中位 %.0f c/s，mf<=0.3 占 %.0f%%，邻居中位 %d"
          % ((obs10[grid_like] > 0).mean() * 100, np.median(rate[grid_like]),
             (mf[grid_like] <= 0.3).mean() * 100, np.median(nb[grid_like])))
