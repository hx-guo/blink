"""LST 成分拟合的第 0a 步：模板有没有对比度。

统筹的判据：真 TGF 模板若本身就是平的，拟合就没有杠杆（SVOM 就塌在这一步，
83 个闪电证实 TGF 的瑞利 p = 0.82）。GECAM 的真值样本是 147 个已发表
GECAM-B TGF（Zhao et al. 2023 GRL），本段量它的对比度，并按星下点落在陆地
还是海洋拆开——陆地雷暴日变化峰在午后、海洋雷暴峰在凌晨，混在一起会互相
抵消成一条近似平的线。

**一个必须跟着结论走的口径**：目录给的经纬度是**星下点**，不是 TGF 源位置，
两者相差几百公里。对 LST 影响不大（800 km 经向约 0.5 h，远小于 3 h 的格），
但**对陆海分类影响不小**——沿海的会分错。所以陆海拆分是"倾向性"证据，
不是逐个事件的真分类。
"""
import csv, math, datetime as dt
import numpy as np
from global_land_mask import globe

CAT = "/Users/skyair/Developer/ihep/blink/scratch_gbm/gecam/gecam_tgf_catalog.csv"


def utc_seconds(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    return (dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S")
            .replace(tzinfo=dt.timezone.utc).timestamp()
            + (float("0." + f) if f else 0.0))


def rayleigh(lst):
    """瑞利检验。返回 (R, Z, p)。Z = nR²，p 用 Wilkie 的修正式。"""
    n = len(lst)
    a = np.asarray(lst) / 24.0 * 2 * np.pi
    C, S = np.cos(a).mean(), np.sin(a).mean()
    R = math.hypot(C, S)
    Z = n * R * R
    p = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n)
                        - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n * n))
    mu = (math.atan2(S, C) / (2 * math.pi) * 24) % 24
    return R, Z, max(min(p, 1.0), 0.0), mu


def chi2_flat(lst, nb=8):
    h, _ = np.histogram(lst, bins=np.linspace(0, 24, nb + 1))
    e = len(lst) / nb
    return float(((h - e) ** 2 / e).sum()), nb - 1, h


rows = list(csv.DictReader(open(CAT)))
t = np.array([utc_seconds(r["UT"]) for r in rows])
lon = np.array([float(r["Longitude_deg"]) for r in rows])
lat = np.array([float(r["Latitude_deg"]) for r in rows])
net = np.array([float(r["NetCounts"]) for r in rows])
lst = (((t % 86400.0) / 3600.0) + lon / 15.0) % 24.0
land = np.array([bool(globe.is_land(a, b)) for a, b in zip(lat, lon)])

print("=== 0a 模板对比度：147 个已发表 GECAM-B TGF ===")
R, Z, p, mu = rayleigh(lst)
c2, dof, h = chi2_flat(lst)
print("  瑞利: R = %.3f, Z = nR² = %.2f, p = %.4f, 圆均值 LST = %.2f h" % (R, Z, p, mu))
print("  卡方（8 格 3 h）: χ² = %.1f / %d 自由度" % (c2, dof))
print("  直方:", list(h))
print("  对照 SVOM（83 个闪电证实）: R = 0.048, p = 0.82 —— 那边没有对比度")
print("  → GECAM-B 的模板 %s" % ("**有**对比度（p < 0.05）" if p < 0.05 else "没有对比度"))

print("\n=== 陆海拆分（星下点，注意几百公里的定位模糊）===")
print("  陆 %d 个 (%.1f%%), 海 %d 个 (%.1f%%)"
      % (land.sum(), 100 * land.mean(), (~land).sum(), 100 * (~land).mean()))
for name, m in (("陆地", land), ("海洋", ~land)):
    R2, Z2, p2, mu2 = rayleigh(lst[m])
    c22, _, h2 = chi2_flat(lst[m])
    print("  %-4s n=%3d  R = %.3f  Z = %.2f  p = %.4f  圆均值 %.2f h  χ² = %.1f/7"
          % (name, m.sum(), R2, Z2, p2, mu2, c22))
    print("       直方:", list(h2), " 12-18h 占 %.1f%%, 00-06h 占 %.1f%%"
          % (100 * ((lst[m] >= 12) & (lst[m] < 18)).mean(),
             100 * (lst[m] < 6).mean()))
print("  物理预期：陆地峰在 15-18 h，海洋峰在 03-06 h。两个子模板的圆均值差 %.2f h"
      % ((rayleigh(lst[land])[3] - rayleigh(lst[~land])[3] + 12) % 24 - 12))

print("\n=== 亮暗失配（模板要与被检验样本的亮度分布对齐）===")
q33 = np.percentile(net, 33.3)
for name, m in (("全样本", np.ones(len(lst), bool)),
                ("亮 2/3 net>%.1f" % q33, net > q33),
                ("暗 1/3 net<=%.1f" % q33, net <= q33)):
    R3, Z3, p3, mu3 = rayleigh(lst[m])
    print("  %-18s n=%3d  R = %.3f  p = %.4f  圆均值 %.2f h  陆地占 %.0f%%"
          % (name, m.sum(), R3, p3, mu3, 100 * land[m].mean()))

print("\n=== 拟合杠杆：这个对比度能测多准 ===")
pv = h / h.sum()
u = 1.0 / 8
fisher_unit = 8 * ((pv - u) ** 2).sum()      # f→0 处，每个候选贡献的费歇尔信息
print("  模板对 flat 的偏离 Σ(p_i−1/8)² = %.5f  →  σ_f ≈ %.2f / √N" %
      (((pv - u) ** 2).sum(), 1 / math.sqrt(fisher_unit)))
for N in (235, 1000, 3000, 10000, 100000):
    print("    N = %6d 个候选  →  σ_f ≈ %.3f" % (N, 1 / math.sqrt(fisher_unit * N)))
print("  **但模板自身也只有 147 个事件**：每格的统计噪声 √(147/8)/147 = %.4f，"
      % (math.sqrt(147 / 8) / 147))
print("  而模板偏离的 rms 只有 %.4f —— 两者同量级，模板噪声要一起传进 σ_f。"
      % math.sqrt(((pv - u) ** 2).mean()))

# 模板噪声的自举：每次从 147 个里有放回重抽，看 Σ(p−1/8)² 的分布
rng = np.random.default_rng(20260911)
boot = []
for _ in range(4000):
    s = rng.choice(lst, size=len(lst), replace=True)
    hb, _ = np.histogram(s, bins=np.linspace(0, 24, 9))
    pb = hb / hb.sum()
    boot.append(((pb - u) ** 2).sum())
boot = np.array(boot)
print("  自举 4000 次: Σ(p−1/8)² 中位 %.5f, 5%% %.5f, 95%% %.5f"
      % (np.median(boot), *np.percentile(boot, [5, 95])))
print("  扣掉纯统计涨落的期望 (8−1)/(8·147) = %.5f 之后，真实偏离 ≈ %.5f"
      % (7 / (8 * 147), ((pv - u) ** 2).sum() - 7 / (8 * 147)))
