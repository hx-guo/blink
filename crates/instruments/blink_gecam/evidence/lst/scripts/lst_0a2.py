"""0a 的三件后续：跨仪器互校、模板稳不稳、以及诚实的功效分析。

1. **跨仪器互校**：GECAM-B 的 147 个已发表 TGF（触发目录）与 HXMT 的 4677 个
   显著未关联候选，两个独立仪器、不同倾角（29° vs 43°）、不同选样（触发目录
   vs 我们的搜索），LST 形状对不对得上。对得上就同时说明三件事：我的 LST 算法
   没错、HXMT 那批确实以 TGF 为主、以及我们可以借 HXMT 那个大得多的模板。
2. **模板稳不稳**：亮暗两半的形状一致吗。不一致就说明模板有亮度依赖，
   拿它去拟一批亮度分布不同的候选会有系统偏差。
3. **功效分析**：把模板自身只有 147 个事件这件事算进去。做法是蒙特卡洛——
   按去偏后的真形状生成一份 147 个的"模板样本"和一份 N 个的"数据样本"，
   用前者拟后者，看 f̂ 的偏差与散布。只报 Σ(p−1/8)² 会高估杠杆。
"""
import csv, math, datetime as dt
import numpy as np
from global_land_mask import globe

CAT = "/Users/skyair/Developer/ihep/blink/scratch_gbm/gecam/gecam_tgf_catalog.csv"
# HXMT 报的两条 LST 分布（格心 1.5, 4.5, ..., 22.5 h）
HXMT_UNASSOC = np.array([0.1321, 0.1237, 0.0793, 0.0571, 0.1087, 0.1863, 0.1716, 0.1410])
HXMT_ASSOC = np.array([0.1725, 0.1341, 0.0983, 0.0763, 0.1018, 0.1203, 0.1428, 0.1540])
N_HXMT_UNASSOC, N_HXMT_ASSOC = 4677, 2547


def utc_seconds(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    return (dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S")
            .replace(tzinfo=dt.timezone.utc).timestamp()
            + (float("0." + f) if f else 0.0))


rows = list(csv.DictReader(open(CAT)))
t = np.array([utc_seconds(r["UT"]) for r in rows])
lon = np.array([float(r["Longitude_deg"]) for r in rows])
lat = np.array([float(r["Latitude_deg"]) for r in rows])
net = np.array([float(r["NetCounts"]) for r in rows])
lst = (((t % 86400.0) / 3600.0) + lon / 15.0) % 24.0
land = np.array([bool(globe.is_land(a, b)) for a, b in zip(lat, lon)])
EDGES = np.linspace(0, 24, 9)
h147, _ = np.histogram(lst, bins=EDGES)
p147 = h147 / h147.sum()

print("=== 一、GECAM-B 147 个 vs HXMT 4677 个未关联 ===")
print("  格 (h)      GECAM-B          HXMT 未关联   HXMT 闪电关联")
for i in range(8):
    print("  %02d-%02d      %3d  %6.4f      %6.4f        %6.4f"
          % (EDGES[i], EDGES[i + 1], h147[i], p147[i], HXMT_UNASSOC[i], HXMT_ASSOC[i]))
for name, q, nq in (("HXMT 未关联", HXMT_UNASSOC, N_HXMT_UNASSOC),
                    ("HXMT 闪电关联", HXMT_ASSOC, N_HXMT_ASSOC)):
    e = q / q.sum() * h147.sum()
    # 两样本：模板一侧的统计噪声也算进去
    var = e + (h147.sum() / nq) ** 2 * (q * nq)
    c2 = float(((h147 - e) ** 2 / var).sum())
    print("  与 %-12s 的一致性: χ² = %.2f / 7 自由度  →  %s"
          % (name, c2, "形状一致" if c2 < 14.07 else "形状不一致 (p<0.05)"))

print("\n=== 二、模板稳不稳：亮暗两半 ===")
med = np.median(net)
hb, _ = np.histogram(lst[net >= med], bins=EDGES)
hf, _ = np.histogram(lst[net < med], bins=EDGES)
tot = hb.sum() + hf.sum()
c2 = 0.0
for i in range(8):
    n_i = hb[i] + hf[i]
    if n_i == 0:
        continue
    eb = n_i * hb.sum() / tot
    ef = n_i * hf.sum() / tot
    c2 += (hb[i] - eb) ** 2 / eb + (hf[i] - ef) ** 2 / ef
print("  亮半 %s" % list(hb))
print("  暗半 %s" % list(hf))
print("  2×8 列联表 χ² = %.2f / 7 自由度  →  %s"
      % (c2, "两半形状一致（模板没有亮度依赖的证据）" if c2 < 14.07 else "两半形状不一致"))

print("\n=== 三、功效分析（把模板自身的 147 个统计噪声算进去）===")
# 去偏：观测到的 Σ(p−1/8)² 里有一份纯粹的多项式涨落
u = 1.0 / 8
obs_dev = ((p147 - u) ** 2).sum()
noise_dev = 7 / (8 * 147)
true_dev = max(obs_dev - noise_dev, 1e-6)
# 构造一个"真形状"：把观测偏离按 sqrt(true/obs) 缩回去
p_true = u + (p147 - u) * math.sqrt(true_dev / obs_dev)
p_true = np.clip(p_true, 1e-4, None)
p_true /= p_true.sum()
print("  观测 Σ(p−1/8)² = %.5f, 纯涨落期望 %.5f, 去偏后真偏离 %.5f"
      % (obs_dev, noise_dev, true_dev))
print("  去偏后的真模板:", np.round(p_true, 4))

rng = np.random.default_rng(20260911)


def fit_f(counts, tmpl):
    """两成分极大似然：q = f·tmpl + (1−f)/8，对 f 做一维扫描。"""
    fs = np.linspace(-0.5, 1.5, 801)
    best, bf = -1e18, 0.0
    for f in fs:
        q = f * tmpl + (1 - f) * u
        if (q <= 0).any():
            continue
        ll = float((counts * np.log(q)).sum())
        if ll > best:
            best, bf = ll, f
    return bf


for N in (1000, 3000, 10000, 100000):
    for f_true in (0.05, 0.20, 0.50):
        est = []
        for _ in range(300):
            tm = rng.multinomial(147, p_true)                 # 模板样本
            tmpl = np.clip(tm / 147.0, 1e-4, None)
            tmpl /= tmpl.sum()
            q = f_true * p_true + (1 - f_true) * u
            dat = rng.multinomial(N, q)                        # 数据样本
            est.append(fit_f(dat, tmpl))
        est = np.array(est)
        print("  N=%6d  f_真=%.2f  →  f̂ 中位 %+.3f  偏差 %+.3f  σ %.3f  "
              "[5%%,95%%] = [%+.2f, %+.2f]"
              % (N, f_true, np.median(est), np.median(est) - f_true, est.std(),
                 *np.percentile(est, [5, 95])))

print("\n  同样的模拟，但模板换成 HXMT 未关联那 4677 个（模板噪声小 5.6 倍）:")
p_h = HXMT_UNASSOC / HXMT_UNASSOC.sum()
for N in (3000, 10000, 100000):
    for f_true in (0.05, 0.20):
        est = []
        for _ in range(300):
            tm = rng.multinomial(N_HXMT_UNASSOC, p_h)
            tmpl = np.clip(tm / N_HXMT_UNASSOC, 1e-4, None)
            tmpl /= tmpl.sum()
            q = f_true * p_h + (1 - f_true) * u
            dat = rng.multinomial(N, q)
            est.append(fit_f(dat, tmpl))
        est = np.array(est)
        print("  N=%6d  f_真=%.2f  →  f̂ 中位 %+.3f  σ %.3f  [5%%,95%%] = [%+.2f, %+.2f]"
              % (N, f_true, np.median(est), est.std(), *np.percentile(est, [5, 95])))
