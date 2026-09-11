"""LST 判别量第一步：公式自检 + 真 TGF 模板。

147 个已发表 GECAM-B TGF 带微秒精度 UT 与卫星经纬度，拿它们做两件事：

1. **核 LST 公式**。目录没有自带 LST 列，所以不能像 GBM 那样直接对一列数。
   改用两条独立的检验：
   * 平太阳时 vs 真太阳时（加时差方程）——两者之差应在 ±0.27 h 以内、
     随年内日期走一条已知的双峰曲线。这验的是公式的量级与季节相位。
   * 真 TGF 的 LST 分布应当长成雷暴的样子（陆地午后峰）。**公式若错，
     这个峰会被抹平或整体平移**——这是外部物理给的最硬的一把尺子。
2. **做出模板**。这批就是成分拟合要用的真 TGF 模板。

模板的口径限定一并量出来：亮/暗两半的 LST 分布差多少（模板失配的上界）。
"""
import csv, math, datetime as dt
import numpy as np

CAT = "/Users/skyair/Developer/ihep/blink/scratch_gbm/gecam/gecam_tgf_catalog.csv"


def utc_seconds(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    return (dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S")
            .replace(tzinfo=dt.timezone.utc).timestamp()
            + (float("0." + f) if f else 0.0))


def equation_of_time_hours(t_unix):
    """时差方程（真太阳时 − 平太阳时），小时。低阶展开，幅度 ±0.27 h。"""
    d = t_unix / 86400.0 - 10957.5          # 自 J2000.0 起的天数
    g = math.radians((357.529 + 0.98560028 * d) % 360)       # 平近点角
    q = math.radians((280.459 + 0.98564736 * d) % 360)       # 平黄经
    lam = q + math.radians(1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 3.6e-7 * d)
    y = math.tan(eps / 2) ** 2
    eot = (y * math.sin(2 * q) - 2 * 0.0167 * math.sin(g)
           + 4 * 0.0167 * y * math.sin(g) * math.cos(2 * q)
           - 0.5 * y * y * math.sin(4 * q) - 1.25 * 0.0167 ** 2 * math.sin(2 * g))
    return math.degrees(eot) / 15.0


def lst_mean(t_unix, lon_deg):
    """平当地太阳时（小时，0..24）。"""
    utc_h = (t_unix % 86400.0) / 3600.0
    return (utc_h + lon_deg / 15.0) % 24.0


rows = list(csv.DictReader(open(CAT)))
t = np.array([utc_seconds(r["UT"]) for r in rows])
lon = np.array([float(r["Longitude_deg"]) for r in rows])
lat = np.array([float(r["Latitude_deg"]) for r in rows])
net = np.array([float(r["NetCounts"]) for r in rows])
lstm = np.array([lst_mean(a, b) for a, b in zip(t, lon)])
eot = np.array([equation_of_time_hours(a) for a in t])
lsta = (lstm + eot) % 24.0

print("=== 一、公式自检 ===")
print("  样本 %d 个（已发表 GECAM-B TGF 全表）" % len(rows))
print("  时差方程 真−平: 中位 %+.3f h, 范围 %+.3f .. %+.3f h（教科书幅度 ±0.27 h）"
      % (np.median(eot), eot.min(), eot.max()))
print("  平 vs 真太阳时的最大分歧 %.3f h —— 对 3 h 一格的直方图无影响" % np.abs(eot).max())
print("  经度范围 %.1f .. %.1f°, 纬度 %.1f .. %.1f°（B 星倾角 29°）"
      % (lon.min(), lon.max(), lat.min(), lat.max()))

print("\n=== 二、真 TGF 的 LST 分布（模板）===")
edges = np.arange(0, 25, 3)
h, _ = np.histogram(lstm, bins=edges)
print("  3 h 一格:")
for i in range(len(h)):
    bar = "#" * int(round(40 * h[i] / max(h.max(), 1)))
    print("    %02d-%02d h  %3d 个  %5.1f%%  %s" % (edges[i], edges[i + 1], h[i],
                                                   100 * h[i] / len(lstm), bar))
print("  卡方检验「平」: chi2 = %.1f, dof = %d" %
      (((h - len(lstm) / 8.0) ** 2 / (len(lstm) / 8.0)).sum(), 7))
# 峰位：按小时的圆均值
ang = lstm / 24.0 * 2 * np.pi
cx, cy = np.cos(ang).mean(), np.sin(ang).mean()
print("  圆均值 LST = %.2f h, 集中度 R = %.3f（R=0 完全均匀，R=1 全挤一点）"
      % ((math.atan2(cy, cx) / (2 * math.pi) * 24) % 24, math.hypot(cx, cy)))

print("\n=== 三、模板失配上界：亮一半 vs 暗一半 ===")
med = np.median(net)
for name, m in (("亮半 net>=%.1f" % med, net >= med), ("暗半 net<%.1f" % med, net < med)):
    hh, _ = np.histogram(lstm[m], bins=edges)
    a = lstm[m] / 24.0 * 2 * np.pi
    r = math.hypot(np.cos(a).mean(), np.sin(a).mean())
    mu = (math.atan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * math.pi) * 24) % 24
    print("  %-18s %3d 个  圆均值 %.2f h  R = %.3f  12-18h 占 %.1f%%"
          % (name, m.sum(), mu, r, 100 * ((lstm[m] >= 12) & (lstm[m] < 18)).mean()))
# 最暗三分之一
q33 = np.percentile(net, 33.3)
m = net <= q33
a = lstm[m] / 24.0 * 2 * np.pi
print("  最暗三分之一 (net<=%.1f) %d 个  圆均值 %.2f h  R = %.3f  12-18h 占 %.1f%%"
      % (q33, m.sum(),
         (math.atan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * math.pi) * 24) % 24,
         math.hypot(np.cos(a).mean(), np.sin(a).mean()),
         100 * ((lstm[m] >= 12) & (lstm[m] < 18)).mean()))

np.savez("/private/tmp/claude-501/-Users-skyair-Developer-ihep-blink/"
         "b546f3ca-0717-495a-9f8e-929a4423a036/scratchpad/gb/lst_template.npz",
         lst=lstm, net=net, lat=lat, lon=lon, t=t)
print("\n模板落盘 lst_template.npz")
