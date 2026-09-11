"""「我们有而目录没有」的那批到底是不是 TGF：拿 WWLLN 闪电关联当判别。

思路：2014 年的目录给了两个人群——目录匹配上的那批是**已认证的真 TGF**，
是天然的对照组。真 TGF 在 ±5 ms / 800 km 内应当高比例地对上一次闪电；
纯假信号应当只落在偶然符合的水平上。两组用同一套判据、同一个库，直接比。

偶然水平不靠估算，靠**时间平移对照**：把候选时刻整体移开若干分钟再查一遍，
位置不动、闪电活动的地理与季节结构都保住，只有真实的时间符合被破坏。

判据与 `blink_lightning::algorithms::associated` 逐行对应：
  dist   = haversine(候选星下点, 闪电)，R_EARTH = 6371 km
  toa    = |卫星位置 − 闪电位置| / c，闪电取 15 km 高
  关联   <=> |t_闪电 − (t_峰 − toa)| <= 5 ms 且 dist <= 800 km

用法: python3 gbm_extras_assoc.py <in.csv> <label> [shift_seconds ...]
      in.csv 需要列 start,delay,bin_best,lon,lat,alt
"""
import csv
import math
import sys
from datetime import datetime, timedelta, timezone

# 库用了 STRICT 表（需要 SQLite >= 3.37），而集群自带的 python3 只有 3.34，
# 连读 schema 都会报 "malformed database schema"。pysqlite3-binary 自带新版。
try:
    import pysqlite3 as sqlite3
except ImportError:  # 本地开发机上标准库通常够新
    import sqlite3

DB = "/gecamfs/Exchange/GSDC/missions/AEfiles/WWLLN.db"
R_EARTH = 6_371_000.0
C = 299_792_458.0
LIGHTNING_ALT = 15_000.0
WINDOW_S = 0.005
MAX_KM = 800.0


def peak_time(row):
    """与 UnifiedSignal::peak_time 一致：start + delay + bin_size_best/2。

    注意 `timedelta` 只到微秒，所以这里的峰时刻量化在 1 µs 上。关联窗是
    ±5 ms，1 µs 是它的 2e-4，可以忽略；但**要做更紧的时刻比对就不能用这个
    实现**——那时得照 HXMT 的做法走整数纳秒（start 的 9 位小数直接转整数
    纳秒，delay 与 bin_size_best 各 round(x*1e9)，相加后再格回 ISO）。
    """
    b = row["start"].rstrip("Z")
    head, _, frac = b.partition(".")
    t = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t += timedelta(seconds=float("0." + frac) if frac else 0.0)
    t += timedelta(seconds=float(row["delay"]) + float(row["bin_best"]) / 2.0)
    return t


def great_circle(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def time_of_arrival(dist, h1, h2):
    alpha = dist / R_EARTH
    d2 = (R_EARTH + h1) ** 2 + (R_EARTH + h2) ** 2 - 2 * (R_EARTH + h1) * (R_EARTH + h2) * math.cos(alpha)
    return math.sqrt(d2) / C


def associated(conn, t_peak, lat, lon, alt):
    lo = (t_peak - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S.%f")
    hi = (t_peak + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S.%f")
    rows = conn.execute(
        "SELECT time, lat, lon FROM lightning WHERE time BETWEEN ? AND ?", (lo, hi)
    ).fetchall()
    for ts, llat, llon in rows:
        dist = great_circle(lat, lon, llat, llon)
        if dist > MAX_KM * 1000:
            continue
        lt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        fixed = t_peak - timedelta(seconds=time_of_arrival(dist, alt, LIGHTNING_ALT))
        if abs((lt - fixed).total_seconds()) <= WINDOW_S:
            return True
    return False


def main():
    path, label = sys.argv[1], sys.argv[2]
    shifts = [float(x) for x in sys.argv[3:]] or [0.0, 300.0, -300.0, 900.0]
    rows = list(csv.DictReader(open(path)))
    conn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    print("%s: %d 个候选" % (label, len(rows)))
    for shift in shifts:
        hits = 0
        for r in rows:
            t = peak_time(r) + timedelta(seconds=shift)
            if associated(conn, t, float(r["lat"]), float(r["lon"]), float(r["alt"])):
                hits += 1
        tag = "真时刻" if shift == 0 else "平移 %+.0f s" % shift
        print("  %-14s 关联 %4d / %4d = %5.2f%%" % (tag, hits, len(rows), 100 * hits / len(rows)))
    print("  （k/N 单独看不含信息，必须与平移对照并排读：平移保住位置与季节结构，"
          "只破坏真实的时间符合，所以它给出的就是这批候选的偶然水平。）")


if __name__ == "__main__":
    main()
