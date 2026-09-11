"""LST 第 0b 步：本底候选的 LST 分布平不平，逐磁纬带。

本底样本取 `fa > 1`（与被检验的显著端不重叠，照 GBM 的留出法）。磁纬用
中心偶极近似（2020 年代地磁北极 80.65°N, 72.68°W）——分带够用，不需要 IGRF。

用法: python3 lst_0b.py <signals 根目录> <标签>
"""
import glob, json, math, sys, datetime as dt
import numpy as np

POLE_LAT, POLE_LON = math.radians(80.65), math.radians(-72.68)


def utc_seconds(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    return (dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S")
            .replace(tzinfo=dt.timezone.utc).timestamp()
            + (float("0." + f) if f else 0.0))


def maglat(lat_deg, lon_deg):
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    s = (math.sin(la) * math.sin(POLE_LAT)
         + math.cos(la) * math.cos(POLE_LAT) * math.cos(lo - POLE_LON))
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


root, tag = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(root + "/*/*/*/*_signals.json")) or \
     sorted(glob.glob(root + "/*/*/*_signals.json"))
print("%s: %d 个日文件" % (tag, len(fs)))

lst, fa, ml, glat = [], [], [], []
for f in fs:
    for s in json.load(open(f)):
        p = s.get("position") or {}
        if "longitude" not in p:
            continue
        t = utc_seconds(s["start"])
        lst.append((((t % 86400.0) / 3600.0) + p["longitude"] / 15.0) % 24.0)
        fa.append(s["false_positive_per_year"])
        ml.append(maglat(p["latitude"], p["longitude"]))
        glat.append(p["latitude"])
lst = np.array(lst); fa = np.array(fa); ml = np.array(ml); glat = np.array(glat)
print("候选 %d 个；地理纬度 %.1f..%.1f°，|磁纬| 中位 %.1f°"
      % (len(lst), glat.min(), glat.max(), np.median(np.abs(ml))))

E = np.linspace(0, 24, 9)


def report(name, m):
    if m.sum() < 30:
        print("  %-22s n=%d 太少，跳过" % (name, m.sum()))
        return
    h, _ = np.histogram(lst[m], bins=E)
    e = m.sum() / 8
    c2 = float(((h - e) ** 2 / e).sum())
    a = lst[m] / 24 * 2 * np.pi
    C, S = np.cos(a).mean(), np.sin(a).mean()
    R = math.hypot(C, S)
    Z = m.sum() * R * R
    print("  %-22s n=%7d  每格 %s  χ²=%9.1f/7  R=%.4f  Z=%.1f"
          % (name, m.sum(), " ".join("%5.2f%%" % (100 * x / m.sum()) for x in h), c2, R, Z))


bg = fa > 1
print("\n=== 本底样本 (fa > 1)，%d 个 ===" % bg.sum())
report("全部", bg)
for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 55), (55, 90)):
    report("|磁纬| %d-%d°" % (lo, hi), bg & (np.abs(ml) >= lo) & (np.abs(ml) < hi))

print("\n=== 对照：显著端 (fa <= 1e-5)，%d 个 ===" % (fa <= 1e-5).sum())
report("全部", fa <= 1e-5)
print("\n=== 对照：全池 ===")
report("全部", np.ones(len(lst), bool))
