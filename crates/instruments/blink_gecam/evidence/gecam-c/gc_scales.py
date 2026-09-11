"""GECAM-C：成串判据的多尺度扫描。

天格把成串窗从 ±600 s 换成 ±30 s，两个人群立刻分开（高磁纬软暴中位 182 个邻居，
短硬暴 4 个）。他们早先在 ±600 s 上量到"不成串"，据此削弱过微暴假说——同一批数据、
同一个问题，换个尺度结论完全相反。所以这里报整条曲线，不只报某一个尺度。

两个人群：
  * 粒子增强串：14:08 那次（36 个候选散在 0.63 s）、22:51 那次（17 个）
  * 干净幸存者：mf<=0.3 且 CPD 本底率<=500 c/s 且 ±10 µs 无 CPD 计数
分离度用中位比和 AUC（Mann-Whitney）两种。
"""

import csv
import datetime as dt

import numpy as np

base = "/Users/skyair/Developer/ihep/blink/scratch_gbm/gecam/"
cpd = list(csv.DictReader(open(base + "cpd_norm.csv")))
xd = list(csv.DictReader(open(base + "xdet.csv")))


def col(src, name):
    return np.array([float(r[name]) if r[name] not in ("", None, "inf") else np.nan for r in src])


binus, rate, cnt = col(cpd, "bin_us"), col(cpd, "cpd_rate"), col(cpd, "count")
lat, lon = col(cpd, "lat"), col(cpd, "lon")
obs10 = col(cpd, "obs_10us")
mf, fa = col(xd, "multiplet_frac"), col(xd, "fa")
delay = col(xd, "delay_us") * 1e-6

ref = dt.datetime(2023, 6, 15, tzinfo=dt.timezone.utc)
t = np.array([(dt.datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S.%f")
               .replace(tzinfo=dt.timezone.utc) - ref).total_seconds() for r in cpd]) + delay
order = np.argsort(t)
ts = t[order]
span = ts[-1] - ts[0]
density = len(ts) / span

P1, P2 = np.radians(80.65), np.radians(-72.68)
la, lo = np.radians(lat), np.radians(lon)
amlat = np.abs(np.degrees(np.arcsin(np.clip(np.sin(la) * np.sin(P1)
                                            + np.cos(la) * np.cos(P1) * np.cos(lo - P2), -1, 1))))

burst = ((t > 14 * 3600 + 8 * 60 + 25) & (t < 14 * 3600 + 9 * 60 + 5)) | \
        ((t > 22 * 3600 + 50 * 60 + 55) & (t < 22 * 3600 + 51 * 60 + 25))
clean = (mf <= 0.3) & (rate <= 500) & (obs10 == 0)
print("粒子增强串 %d 个候选；干净幸存者 %d 个；候选池 %d 个，池率 %.4f /s"
      % (burst.sum(), clean.sum(), len(t), density))


def auc(a, b):
    """Mann-Whitney：随机取一个 a、一个 b，a > b 的概率（并列算一半）。"""
    both = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable").astype(float) + 1
    # 处理并列
    order_ = np.argsort(both, kind="stable")
    sorted_ = both[order_]
    i = 0
    while i < sorted_.size:
        j = i
        while j + 1 < sorted_.size and sorted_[j + 1] == sorted_[i]:
            j += 1
        if j > i:
            ranks[order_[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    ra = ranks[:a.size].sum()
    return (ra - a.size * (a.size + 1) / 2) / (a.size * b.size)


SCALES = (0.1, 0.5, 2.0, 5.0, 30.0, 60.0, 300.0, 600.0)
print("\n%-9s %8s | %-28s | %-28s | %7s %7s" %
      ("窗半宽", "泊松期望", "粒子串 邻居数 中位(5–95%)", "干净幸存者 中位(5–95%)", "中位比", "AUC"))
store = {}
for W in SCALES:
    lo_i = np.searchsorted(ts, ts - W, "left")
    hi_i = np.searchsorted(ts, ts + W, "right")
    nb_sorted = hi_i - lo_i - 1
    nb = np.empty_like(nb_sorted)
    nb[order] = nb_sorted
    store[W] = nb
    a, b = nb[burst].astype(float), nb[clean].astype(float)
    expect = density * 2 * W
    ratio = (np.median(a) + 1) / (np.median(b) + 1)
    print("±%-8g %8.2f | %10.0f (%3.0f–%3.0f)  n=%3d | %10.0f (%3.0f–%3.0f)  n=%3d | %7.2f %7.3f"
          % (W, expect, np.median(a), *np.percentile(a, [5, 95]), a.size,
             np.median(b), *np.percentile(b, [5, 95]), b.size, ratio, auc(a, b)))

print("\n各尺度下：把阈定在'干净幸存者的 95 分位'，能摘掉粒子串的多少")
for W in SCALES:
    nb = store[W]
    thr = np.percentile(nb[clean], 95)
    print("  ±%-6g 阈=%5.1f（干净者 95 分位）→ 摘掉粒子串 %3d/%3d (%5.1f%%)，"
          "误摘干净者 %2d/%3d (%4.1f%%)，全池摘 %4d (%4.1f%%)"
          % (W, thr, (nb[burst] > thr).sum(), burst.sum(), (nb[burst] > thr).mean() * 100,
             (nb[clean] > thr).sum(), clean.sum(), (nb[clean] > thr).mean() * 100,
             (nb > thr).sum(), (nb > thr).mean() * 100))

print("\n各尺度下：邻居数与'是不是粒子'的相关（全池，用 CPD ±10 µs 命中率当标签）")
for W in SCALES:
    nb = store[W]
    hi = nb > np.percentile(nb, 90)
    lo_ = nb <= np.percentile(nb, 50)
    print("  ±%-6g 邻居数最高 10%% 的 CPD 命中 %5.1f%%，最低 50%% 的 %5.1f%%，比 %.2f"
          % (W, (obs10[hi] > 0).mean() * 100, (obs10[lo_] > 0).mean() * 100,
             (obs10[hi] > 0).mean() / max((obs10[lo_] > 0).mean(), 1e-9)))

print("\n高磁纬人群（|mlat| >= 40）vs 低磁纬（< 30）的邻居数（对照天格的磁纬分裂）")
hi_m, lo_m = amlat >= 40, amlat < 30
print("  n = %d / %d" % (hi_m.sum(), lo_m.sum()))
for W in SCALES:
    nb = store[W]
    print("  ±%-6g 高磁纬中位 %4.0f，低磁纬中位 %4.0f，比 %.2f"
          % (W, np.median(nb[hi_m]), np.median(nb[lo_m]),
             (np.median(nb[hi_m]) + 1) / (np.median(nb[lo_m]) + 1)))

print("\n通过全链的那 2 个候选在各尺度上的邻居数（该是孤立的）")
final = (fa <= 7e-7) & (mf <= 0.3) & (rate <= 500) & (obs10 == 0)
for i in np.flatnonzero(final):
    print("  %s |mlat|=%.1f°  " % (cpd[i]["start"], amlat[i])
          + "  ".join("±%g:%d" % (W, store[W][i]) for W in SCALES))
