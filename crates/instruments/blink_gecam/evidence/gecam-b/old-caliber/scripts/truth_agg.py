"""合并 truth_*.json：构造无关的 mf + CPD–GRD 时戳差。"""
import glob, json, math
import numpy as np

rows = []
fs = sorted(glob.glob("/scratchfs2/gecam/guohx/gecambrun/truth_[0-9].json"))
for f in fs:
    rows += json.load(open(f))
tgf = [r for r in rows if r["tag"] == "tgf"]
ctrl = [r for r in rows if r["tag"] == "ctrl"]
print("分片 %d 个；目录 TGF %d 个，对照候选 %d 个" % (len(fs), len(tgf), len(ctrl)))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


print("\n=== 一、构造无关的 mf（只锚目录 UT，±3 ms 内固定 100 µs 滑窗找峰）===")
n = np.array([r["n_peak"] for r in tgf])
mf = np.array([r["mf_peak"] for r in tgf])
off = np.array([r["peak_off_us"] for r in tgf])
exp = np.array([r["exp_peak"] for r in tgf])
net = np.array([r["net"] for r in tgf])
nd = np.array([r["ndet_peak"] for r in tgf])
print("  峰相对目录 UT 的偏移 µs: 中位 %+.1f  5%% %+.1f  25%% %+.1f  75%% %+.1f  95%% %+.1f"
      % (np.median(off), *np.percentile(off, [5, 25, 75, 95])))
print("    偏移 > 0 的 %d/%d = %.1f%%  —— 目录 UT 系统性偏早"
      % (int((off > 0).sum()), len(off), 100 * (off > 0).mean()))
print("  峰内计数 n: 中位 %.0f（目录 NetCounts 中位 %.1f），本底期望中位 %.2f"
      % (np.median(n), np.median(net), np.median(exp)))
print("  n/NetCounts 中位 %.2f" % np.median(n / net))
print("  构造无关 mf: 中位 %.3f  75%% %.3f  95%% %.3f  最大 %.3f"
      % (np.median(mf), *np.percentile(mf, [75, 95]), mf.max()))
for thr in (0.25, 0.30, 0.40, 0.50):
    k = int((mf <= thr).sum())
    lo, hi = wilson(k, len(mf))
    print("    mf <= %.2f 放行 %3d/%3d = %5.1f%% [%4.1f%%, %4.1f%%]"
          % (thr, k, len(mf), 100 * k / len(mf), 100 * lo, 100 * hi))
print("  点亮探头数 中位 %.0f" % np.median(nd))
# 分亮度
print("  按目录 NetCounts 分档的 mf<=0.3 放行率:")
for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 1e9)):
    m = (net >= lo) & (net < hi)
    if m.sum() == 0:
        continue
    k = int((mf[m] <= 0.3).sum())
    a, b = wilson(k, int(m.sum()))
    print("    net∈[%3d,%5s)  %3d 个  %5.1f%% [%4.1f,%4.1f]  峰内 n 中位 %4.1f"
          % (lo, "inf" if hi > 1e8 else int(hi), m.sum(), 100 * k / m.sum(),
             100 * a, 100 * b, np.median(n[m])))

print("\n=== 二、对照组的 mf（我们搜索选出的最佳格）===")
cmf = np.array([r["mf_peak"] for r in ctrl])
cb = np.array([r["bin_us"] for r in ctrl])
cn = np.array([r["n_peak"] for r in ctrl])
print("  n=%d  mf 中位 %.3f  mf<=0.3 放行 %.1f%%  bin 中位 %.3f µs  窗内 n 中位 %.0f"
      % (len(cmf), np.median(cmf), 100 * (cmf <= 0.3).mean(), np.median(cb), np.median(cn)))
for lo, hi in ((0, 1), (1, 10), (10, 50), (50, 1e9)):
    m = (cb >= lo) & (cb < hi)
    if m.sum() == 0:
        continue
    print("    bin∈[%4.0f,%5s) µs  %6d 个 (%5.1f%%)  mf 中位 %.3f  mf<=0.3 放行 %5.1f%%"
          % (lo, "inf" if hi > 1e8 else int(hi), m.sum(), 100 * m.mean(),
             np.median(cmf[m]), 100 * (cmf[m] <= 0.3).mean()))

print("\n=== 三、CPD–GRD 时戳差（判据该不该换成同戳）===")
for name, group in (("目录 TGF", tgf), ("对照候选", ctrl)):
    d = np.array([x for r in group for x in r["cpd_dts"] if x >= 0])
    if len(d) == 0:
        print("  %s: 窗内没有 CPD 计数" % name)
        continue
    k_ex = sum(1 for r in group if r["cpd_exact"] > 0)
    lo, hi = wilson(k_ex, len(group))
    print("  %-8s 候选 %6d 个, 窗内 CPD 计数 %6d 个" % (name, len(group), len(d)))
    print("     |Δt| µs: 中位 %8.3f  5%% %7.3f  25%% %7.3f  75%% %8.3f"
          % (np.median(d), *np.percentile(d, [5, 25, 75])))
    for t in (0.0, 0.05, 0.1, 0.5, 1.0, 10.0):
        print("       |Δt| <= %6.2f µs 的 CPD 计数占 %6.2f%%" % (t, 100 * (d <= t).mean()))
    print("     **同戳（Δt 恰为 0）**: 有同戳对的候选 %d/%d = %.2f%% [%.2f%%, %.2f%%]"
          % (k_ex, len(group), 100 * k_ex / len(group), 100 * lo, 100 * hi))
