"""两段式摘除（±0.5 s 先摘密的，再用 ±30/60 s）到底省不省。

`main` 的推论：第一段几乎不动干净幸存者（0/405），所以是"免费"的，先摘掉三分之二，
第二段总代价该更低。这个推论的前提是"第一段给第二段腾出了裕度"——只有当**剩下的
粒子串在第二段尺度上仍远高于幸存者**时才成立。这里直接验它。

小样本区间一律用 Clopper–Pearson，`0/405` 不写成 0.0%。
"""

import csv
import datetime as dt

import numpy as np
from scipy.stats import beta

base = "/Users/skyair/Developer/ihep/blink/scratch_gbm/gecam/"
cpd = list(csv.DictReader(open(base + "cpd_norm.csv")))
xd = list(csv.DictReader(open(base + "xdet.csv")))


def col(src, name):
    return np.array([float(r[name]) if r[name] not in ("", None, "inf") else np.nan for r in src])


rate, obs10 = col(cpd, "cpd_rate"), col(cpd, "obs_10us")
mf = col(xd, "multiplet_frac")
delay = col(xd, "delay_us") * 1e-6
ref = dt.datetime(2023, 6, 15, tzinfo=dt.timezone.utc)
t = np.array([(dt.datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S.%f")
               .replace(tzinfo=dt.timezone.utc) - ref).total_seconds() for r in cpd]) + delay
order = np.argsort(t)
ts = t[order]

burst = ((t > 14 * 3600 + 8 * 60 + 25) & (t < 14 * 3600 + 9 * 60 + 5)) | \
        ((t > 22 * 3600 + 50 * 60 + 55) & (t < 22 * 3600 + 51 * 60 + 25))
clean = (mf <= 0.3) & (rate <= 500) & (obs10 == 0)


def neighbors(W):
    lo = np.searchsorted(ts, ts - W, "left")
    hi = np.searchsorted(ts, ts + W, "right")
    nb_sorted = hi - lo - 1
    nb = np.empty_like(nb_sorted)
    nb[order] = nb_sorted
    return nb


def ci(k, n):
    """Clopper–Pearson 95% 区间，返回 (下限%, 上限%)。"""
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo * 100, hi * 100


nb = {W: neighbors(W) for W in (0.5, 30.0, 60.0)}

print("=== 1. 第一段（±0.5 s，阈 1）摘掉的与剩下的，在第二段尺度上的分布 ===")
stage1 = nb[0.5] > 1
removed = burst & stage1
left = burst & ~stage1
print("  粒子串 %d 个：第一段摘掉 %d，剩下 %d" % (burst.sum(), removed.sum(), left.sum()))
for W in (30.0, 60.0):
    print("  ±%g s 邻居数：" % W)
    print("    第一段摘掉的 %d 个：中位 %.0f，范围 %d–%d"
          % (removed.sum(), np.median(nb[W][removed]), nb[W][removed].min(), nb[W][removed].max()))
    print("    第一段剩下的 %d 个：中位 %.0f，范围 %d–%d，**最小值 %d**"
          % (left.sum(), np.median(nb[W][left]), nb[W][left].min(), nb[W][left].max(), nb[W][left].min()))
    print("    干净幸存者 %d 个：中位 %.0f，95 分位 %.0f，最大 %d"
          % (clean.sum(), np.median(nb[W][clean]), np.percentile(nb[W][clean], 95), nb[W][clean].max()))

print("\n=== 2. 第二段阈能放到多松 ===")
for W in (30.0, 60.0):
    lo_left = nb[W][left].min()
    p95 = np.percentile(nb[W][clean], 95)
    print("  ±%g s：剩余粒子串最小邻居数 %d → 阈最松可放到 %d（单段时是幸存者 95 分位 %.0f）"
          % (W, lo_left, lo_left - 1, p95))

print("\n=== 3/4. 三种方案逐条比 ===")
print("  %-30s %-22s %-24s %s" % ("方案", "摘粒子串", "误摘干净者", "全池扰动"))


def report(label, removed_mask, pool_mask):
    k1, n1 = int((burst & removed_mask).sum()), int(burst.sum())
    k2, n2 = int((clean & removed_mask).sum()), int(clean.sum())
    a1, b1 = ci(k1, n1)
    a2, b2 = ci(k2, n2)
    print("  %-30s %3d/%-3d %5.1f%% (%.1f–%.1f) %3d/%-3d %5.2f%% (%.2f–%.2f)  %4d (%.1f%%)"
          % (label, k1, n1, k1 / n1 * 100, a1, b1, k2, n2, k2 / n2 * 100, a2, b2,
             int(pool_mask.sum()), pool_mask.mean() * 100))


for W in (30.0, 60.0):
    thr = np.percentile(nb[W][clean], 95)
    m = nb[W] > thr
    report("单段 ±%g s，阈 %.0f" % (W, thr), m, m)
thr05 = 1
m05 = nb[0.5] > thr05
report("单段 ±0.5 s，阈 1", m05, m05)
for W in (30.0, 60.0):
    lo_left = nb[W][left].min()
    thr2 = lo_left - 1
    m = m05 | (nb[W] > thr2)
    report("两段 ±0.5 s(阈1) → ±%g s(阈 %d)" % (W, thr2), m, m)
    thr_same = np.percentile(nb[W][clean], 95)
    m2 = m05 | (nb[W] > thr_same)
    report("两段 ±0.5 s(阈1) → ±%g s(阈 %.0f，不放松)" % (W, thr_same), m2, m2)

print("\n=== 不依赖幸存者样本的阈（避免循环）===")
print("  %-34s %-16s %-18s %s" % ("规则", "阈", "摘粒子串", "全池扰动"))
for W in (0.5, 30.0, 60.0):
    density = len(ts) / (ts[-1] - ts[0])
    expect = density * 2 * W
    for name, thr in (("全池 95 分位", np.percentile(nb[W], 95)),
                      ("全池 99 分位", np.percentile(nb[W], 99)),
                      ("泊松期望 × 3", 3 * expect),
                      ("泊松期望 + 5σ", expect + 5 * np.sqrt(expect))):
        m = nb[W] > thr
        k1 = int((burst & m).sum())
        k2 = int((clean & m).sum())
        print("  ±%-5g %-26s %6.1f  %3d/%-3d (%5.1f%%)  %4d (%.1f%%)  误摘干净者 %d (%.2f%%)"
              % (W, name, thr, k1, burst.sum(), k1 / burst.sum() * 100,
                 int(m.sum()), m.mean() * 100, k2, k2 / clean.sum() * 100))
