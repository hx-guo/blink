"""细看会被现有/备选判据切掉的真 TGF，以及跨星阈值可迁移性。"""
import csv, math
import numpy as np

D = "/private/tmp/claude-501/-Users-skyair-Developer-ihep-blink/b546f3ca-0717-495a-9f8e-929a4423a036/scratchpad/gb"
rows = list(csv.DictReader(open(D + "/tgf_features_all.csv")))
F = lambda r, k: float(r[k]) if r[k] != "" else float("nan")

print("=== mf > 0.30 的真 TGF（会被 C 星那条同戳判据切掉）===")
print("%-30s %8s %6s %8s %6s %6s %6s %6s %6s %8s" %
      ("start", "bin_us", "n", "fa", "mf", "f3", "maxm", "ndet", "cpd10", "cpdenv"))
bad = [r for r in rows if F(r, "multiplet_frac") > 0.30]
for r in sorted(bad, key=lambda x: -F(x, "multiplet_frac")):
    print("%-30s %8.1f %6d %8.1e %6.3f %6.3f %6d %6d %6s %8s" %
          (r["start"][:26], F(r, "bin_us"), int(r["n_core"]), F(r, "fa"),
           F(r, "multiplet_frac"), F(r, "f3"), int(r["max_mult"]),
           int(r["n_det_hit"]), r["cpd_10us"], r["cpd_env"]))
print("共 %d 个" % len(bad))

print("\n=== mf 随窗内计数的稀释（真 TGF）===")
n = np.array([int(r["n_core"]) for r in rows])
mf = np.array([F(r, "multiplet_frac") for r in rows])
bu = np.array([F(r, "bin_us") for r in rows])
for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 120), (120, 10000)):
    m = (n >= lo) & (n < hi)
    if m.sum() == 0:
        continue
    print("  n∈[%3d,%5d)  %3d 个   mf 中位 %.3f   mf<=0.3 通过 %5.1f%%   bin 中位 %7.1f µs" %
          (lo, hi, m.sum(), np.median(mf[m]), 100 * (mf[m] <= 0.3).mean(), np.median(bu[m])))

print("\n=== bin_us 随计数 ===")
for lo, hi in ((0, 1), (1, 10), (10, 50), (50, 200), (200, 2000)):
    m = (bu >= lo) & (bu < hi)
    if m.sum() == 0:
        continue
    print("  bin∈[%4d,%5d) µs  %3d 个 (%4.1f%%)  n 中位 %5.1f  mf 中位 %.3f" %
          (lo, hi, m.sum(), 100 * m.mean(), np.median(n[m]), np.median(mf[m])))

print("\n=== CPD 环境率：跨星阈值可迁移性 ===")
env = np.array([F(r, "cpd_env") for r in rows]) / 2.0
print("  B 星 8 路 CPD：总率中位 %.0f c/s -> 每路 %.0f c/s" % (np.median(env), np.median(env) / 8))
print("  C 星 2 路 CPD：那条阈 500 c/s -> 每路 250 c/s")
for thr in (500, 1000, 2000, 4000):
    k = int((env > thr).sum())
    print("    B 星 > %5d c/s (每路 %4.0f)  %3d/%d = %5.1f%% 的真 TGF 被否决" %
          (thr, thr / 8, k, len(env), 100 * k / len(env)))

print("\n=== CPD 逐档 obs/exp，按窗内计数分箱 ===")
for key, half in (("cpd_10us", 1e-5), ("cpd_100us", 1e-4)):
    print("  ±%s:" % key[4:])
    c = np.array([F(r, key) for r in rows])
    for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 10000)):
        m = (n >= lo) & (n < hi) & ~np.isnan(c) & (env > 0)
        if m.sum() == 0:
            continue
        mu = env[m] * (bu[m] * 1e-6 + 2 * half)
        exp = (1 - np.exp(-mu)).sum()
        obs = int((c[m] > 0).sum())
        print("    n∈[%3d,%5d) %3d 个  命中 %3d (%5.1f%%)  期望 %5.1f  obs/exp %5.2f" %
              (lo, hi, m.sum(), obs, 100 * obs / m.sum(), exp, obs / exp))

print("\n=== 「窗内 CPD 计数数」的总量对账 ===")
nacd = np.array([F(r, "n_acd") for r in rows])
nbg = np.array([F(r, "n_acd_bg") for r in rows])
ncore = np.array([int(r["n_core"]) for r in rows])
# 期望窗内 CPD 计数 = 环境率 × 窗宽
mu_all = env * (bu * 1e-6)
print("  窗内 CPD 计数合计 %d,  按环境率算的期望合计 %.1f,  超出 %.1f 个 = %.2f 倍" %
      (nacd.sum(), mu_all.sum(), nacd.sum() - mu_all.sum(), nacd.sum() / mu_all.sum()))
print("  净 CPD/GRD 比 = (Σn_acd − Σμ)/Σn_core = %.4f   （目录 CPDtoGRDcountsRatio 中位 0.044）" %
      ((nacd.sum() - mu_all.sum()) / ncore.sum()))

print("\n=== 显著性最弱的 10 个真 TGF（暗端在哪）===")
fa = np.array([F(r, "fa") for r in rows])
idx = np.argsort(-fa)[:10]
print("%-30s %9s %5s %8s %6s %6s" % ("start", "fa", "n", "bin_us", "mf", "cpd10"))
for i in idx:
    r = rows[i]
    print("%-30s %9.1e %5d %8.1f %6.3f %6s" %
          (r["start"][:26], F(r, "fa"), int(r["n_core"]), F(r, "bin_us"),
           F(r, "multiplet_frac"), r["cpd_10us"]))
