#!/usr/bin/env python3
"""绕开 WWLLN 夜间偏差：用 pre-2025 的显著未关联候选当模板，拟 post-2024。

两边是同一类样本、时间上互不重叠，模板不经过闪电台网，因此不带台网的
昼夜探测效率偏差。这对回答"WWLLN 到期之后那一段还剩什么独立验证"正好合适。
"""
import csv, sys, collections
import numpy as np

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5
rng = np.random.default_rng(20260911)


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def hist(x):
    return np.histogram(x, bins=NBIN, range=(0, 24))[0].astype(float)


def fit(obs, T, B):
    N = obs.sum(); T = T / T.sum(); B = B / B.sum()
    fs = np.linspace(-1.0, 2.0, 3001)
    nll = np.array([-(obs * np.log(np.maximum(N * (f * T + (1 - f) * B), 1e-12))
                      - N * (f * T + (1 - f) * B)).sum() for f in fs])
    i = nll.argmin(); b = fs[i]
    ok = fs[nll <= nll[i] + 1.92]
    mu = N * (b * T + (1 - b) * B)
    return b, ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


def dayboot(x, d, T, B, n=400):
    by = collections.defaultdict(list)
    for a, k in zip(x, d):
        by[k].append(a)
    ks = list(by)
    out = [fit(hist(np.concatenate([by[ks[j]] for j in rng.choice(len(ks), len(ks))])), T, B)[0]
           for _ in range(n)]
    return np.percentile(out, [2.5, 50, 97.5])


rows = []
with open(sys.argv[1]) as f:
    r = csv.reader(f); next(r)
    for start, fpy, lon, lat, assoc, nb, train in r:
        rows.append((start[:10], float(fpy), lst_hours(start, float(lon)),
                     assoc == '1', train == '1'))

bg = np.array([x[2] for x in rows if x[1] > 1 and not x[4]])
sig = [x for x in rows if x[1] < 1e-5 and not x[4] and not x[3]]
pre = np.array([x[2] for x in sig if x[0] < '2025'])
pre_d = np.array([x[0] for x in sig if x[0] < '2025'])
post = np.array([x[2] for x in sig if x[0] >= '2025'])
post_d = np.array([x[0] for x in sig if x[0] >= '2025'])
tgf = np.array([x[2] for x in rows if x[3]])
rep = np.array([x[2] for x in rows if x[1] < 1e-5 and x[4] and not x[3]])
rep_d = np.array([x[0] for x in rows if x[1] < 1e-5 and x[4] and not x[3]])
print("pre-2025 显著未关联 %d（%d 天）；post-2024 %d（%d 天）；本底 %d"
      % (len(pre), len(set(pre_d)), len(post), len(set(post_d)), len(bg)))

B = hist(bg)
print("\n模板形状（%/3h，格心 LST 1.5…22.5 h）")
for nm, h in (("pre-2025 显著未关联（新模板）", hist(pre)),
              ("闪电关联 TGF（旧模板）", hist(tgf))):
    print("  %-26s %s  峰 %.1f h" % (nm, " ".join("%5.1f" % x for x in 100 * h / h.sum()),
                                     CENTRES[(h / h.sum()).argmax()]))

print("\n=== 用 pre-2025 模板拟 post-2024（两边都不经过闪电台网）===")
b, lo, hi, c2 = fit(hist(post), hist(pre), B)
q = dayboot(post, post_d, hist(pre), B)
print("  f_TGF = %.3f  95%% [%.3f, %.3f]  chi2 = %.1f / 6 dof" % (b, lo, hi, c2))
print("  按天 bootstrap: 中位 %.3f，95%% [%.3f, %.3f]" % (q[1], q[0], q[2]))
print("  -> post-2024 的 %d 个显著未关联候选里，真 TGF 约 %.0f 个（95%% 下限 %.0f）"
      % (len(post), b * len(post), max(lo, q[0]) * len(post)))

print("\n=== 同一模板做负对照：列车/REP 候选 ===")
b2, lo2, hi2, c22 = fit(hist(rep), hist(pre), B)
q2 = dayboot(rep, rep_d, hist(pre), B)
print("  N=%d（%d 天）f_TGF = %.3f  95%% [%.3f, %.3f]  chi2 = %.1f / 6"
      % (len(rep), len(set(rep_d)), b2, lo2, hi2, c22))
print("  按天 bootstrap: 中位 %.3f，95%% [%.3f, %.3f]  <- 日期太少，区间打开" % (q2[1], q2[0], q2[2]))
print("  LST 直方: %s" % " ".join("%5.1f" % x for x in 100 * hist(rep) / len(rep)))

print("\n=== 系统底：把 pre-2025 模板拿去拟本底（真值 0）===")
odd = np.array([int(d.replace('-', '')) % 2 == 1 for d in
                np.array([x[0] for x in rows if x[1] > 1 and not x[4]])])
b3, lo3, hi3, c23 = fit(hist(bg[odd]), hist(pre), hist(bg[~odd]))
print("  f = %+.4f  95%% [%+.4f, %+.4f]  chi2 = %.1f / 6" % (b3, lo3, hi3, c23))
