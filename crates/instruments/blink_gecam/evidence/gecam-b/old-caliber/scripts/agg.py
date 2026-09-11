"""合并 10 片 feat_tgf_*.csv，出 147 个已发表 GECAM-B TGF 的全局分布。

比例一律带 Wilson 95% 区间。CPD 的 obs/exp 用每个候选自己的环境率
（cpd_env / 活时间）算期望，不用全局率。
"""
import csv, glob, math, json
import numpy as np

D = "/private/tmp/claude-501/-Users-skyair-Developer-ihep-blink/b546f3ca-0717-495a-9f8e-929a4423a036/scratchpad/gb"

rows = []
for f in sorted(glob.glob(D + "/feat_tgf_*.csv")):
    rows += list(csv.DictReader(open(f)))
rows.sort(key=lambda r: r["start"])
print("合并后 %d 行" % len(rows))

# 与 tgf_match.csv 对账
match = list(csv.DictReader(open(D + "/tgf_match.csv")))
found = [m for m in match if m["found"] == "1"]
print("目录 %d 行, found=1 的 %d 行, 特征表 %d 行" % (len(match), len(found), len(rows)))
mm = {m["ut"]: m for m in found}

A = lambda k: np.array([float(r[k]) for r in rows])
I = lambda k: np.array([int(r[k]) for r in rows])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def frac(name, mask):
    n = len(mask)
    k = int(mask.sum())
    lo, hi = wilson(k, n)
    print("  %-34s %3d/%3d = %5.1f%%   95%% CI [%4.1f%%, %4.1f%%]"
          % (name, k, n, 100 * k / n, 100 * lo, 100 * hi))


def q(name, a, unit=""):
    print("  %-16s 中位 %8.3f  5%% %8.3f  25%% %8.3f  75%% %8.3f  95%% %8.3f  最小 %8.3f 最大 %8.3f %s"
          % (name, np.median(a), *np.percentile(a, [5, 25, 75, 95]), a.min(), a.max(), unit))


print("\n=== 一、bin_size_best（最佳格宽）===")
bin_us = A("bin_us")
q("bin_us", bin_us, "µs")
for lim in (1, 10, 50, 100, 200, 500, 1000):
    frac("bin_us < %d µs" % lim, bin_us < lim)
# 与目录 BB 块长对比
dur = np.array([float(mm[r["start"][:26]]["dur_us"]) if r["start"][:26] in mm else np.nan
                for r in rows])
# start 的格式是 ISO 带 Z，对不上 ut；用 tgf_match 的 our_bin_us 反查
byfa = {m["our_fa"]: m for m in found}
dur2, ourbin = [], []
for m in found:
    dur2.append(float(m["dur_us"]))
    ourbin.append(float(m["our_bin_us"]))
dur2, ourbin = np.array(dur2), np.array(ourbin)
print("\n  目录 Duration_us (BB 块长, 142 个找回的):")
q("Duration_us", dur2, "µs")
r = ourbin / dur2
print("  bin_best / Duration_us 比:")
q("ratio", r)
print("  bin_best < Duration 的 %d/%d" % (int((ourbin < dur2).sum()), len(dur2)))

print("\n=== 二、multiplet_frac 与 f3（同戳判据）===")
mf = A("multiplet_frac")
f3 = A("f3")
q("multiplet_frac", mf)
frac("mf == 0", mf == 0)
frac("mf <= 0.25 (判据通过)", mf <= 0.25)
frac("mf <= 0.30 (判据通过)", mf <= 0.30)
frac("mf <= 0.40", mf <= 0.40)
print("  mf 的最大值 %.3f -> 阈 0.3 的余量 %.3f" % (mf.max(), 0.30 - mf.max()))
q("f3", f3)
frac("f3 == 0", f3 == 0)
frac("f3 <= 0.05", f3 <= 0.05)
print("  f3 的最大值 %.3f" % f3.max())
mx = I("max_mult")
print("  max_mult 分布:", dict(zip(*np.unique(mx, return_counts=True))))

print("\n=== 三、单路占比 det_frac_max（现役判据 0.8）===")
dfm = A("det_frac_max")
q("det_frac_max", dfm)
frac("det_frac_max > 0.8 (会被现役判据否决)", dfm > 0.8)
frac("det_frac_max > 0.5", dfm > 0.5)
nd = I("n_det_hit")
q("n_det_hit", nd.astype(float), "路")

print("\n=== 四、能谱 pi_ge_368 ===")
p368 = A("pi_ge_368")
q("pi_ge_368", p368)
frac("pi_ge_368 > 0.10", p368 > 0.10)
frac("pi_ge_368 > 0.30", p368 > 0.30)

print("\n=== 五、CPD 符合 ===")
n_core = A("n_core")
nacd = A("n_acd")
nacd_bg = A("n_acd_bg")
cpd_env = A("cpd_env")
has_cpd = cpd_env > 0
print("  有 CPD 文件的候选 %d/%d" % (int(has_cpd.sum()), len(rows)))
for key, half in (("cpd_10us", 1e-5), ("cpd_100us", 1e-4), ("cpd_1ms", 1e-3), ("cpd_10ms", 1e-2)):
    c = np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])
    ok = ~np.isnan(c)
    # 期望: 当地率 = cpd_env / 2 s(标称, 已核 GTI 未切), 窗宽 = bin + 2*half
    rate = cpd_env[ok] / 2.0
    w = bin_us[ok] * 1e-6 + 2 * half
    mu = rate * w
    exp = (1 - np.exp(-mu)).sum()
    obs = int((c[ok] > 0).sum())
    lo, hi = wilson(obs, int(ok.sum()))
    print("  ±%-7s 命中 %3d/%3d = %5.1f%% [%4.1f%%,%4.1f%%]   偶然期望 %5.1f 个 (%4.1f%%)   obs/exp = %.2f"
          % (key[4:], obs, int(ok.sum()), 100 * obs / ok.sum(), 100 * lo, 100 * hi,
             exp, 100 * exp / ok.sum(), obs / exp if exp > 0 else float("nan")))

print("\n  CPD 环境率 (cpd_env/2 s):")
q("cpd_rate", cpd_env / 2.0, "c/s")
frac("cpd 环境率 > 500 c/s (C 星那条第二判据会否决)", cpd_env / 2.0 > 500)

print("\n  n_acd/n_core (窗内 CPD/GRD 比) vs 目录 CPDtoGRDcountsRatio:")
ratio = nacd / n_core
q("n_acd/n_core", ratio)
frac("n_acd == 0 (窗内无 CPD 计数)", nacd == 0)
catr = np.array([float(m["cat_cpd"]) for m in found])
q("目录 CPDtoGRD", catr)
print("  目录 <= 0 的 %d/%d = %.1f%%" % (int((catr <= 0).sum()), len(catr),
                                       100 * (catr <= 0).mean()))

print("\n=== 六、逐路基线零格（探头级停机）===")
bz = I("base_zero")
bl = I("base_len")
print("  base_len 取值:", dict(zip(*np.unique(bl, return_counts=True))))
frac("基线向量有零格的候选", bz > 0)
print("  零格数分布:", dict(zip(*np.unique(bz, return_counts=True))))

print("\n=== 七、GTI 切割 ===")
live = A("gti_live_s")
nom = A("gti_nominal_s")
frac("基线窗被 GTI 切到", live < nom - 1e-6)
q("gti_live_s", live, "s")

print("\n=== 八、显著性 ===")
fa = A("fa")
q("log10(fa)", np.log10(np.maximum(fa, 1e-300)))
for lim in (1e-5, 1e-7, 7e-7, 1e-10):
    frac("fa <= %.0e" % lim, fa <= lim)

# 落盘合并表
out = D + "/tgf_features_all.csv"
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0].keys()))
w.writeheader()
w.writerows(rows)
print("\n合并表 ->", out)
