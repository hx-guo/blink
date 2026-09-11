"""逐行输出闪电关联标记与「雷暴邻近度」，给第二层判别用。

`gbm_extras_assoc.py` 只报人群的总关联率。要往下分一层——**没关联的那批里
还有没有 TGF**——需要两样它没有的东西：

1. **逐行的关联标记**，才能把「没关联」的子集单独拎出来分析。
2. **一个比「同一次闪电」宽松的量**：真 TGF 必定发生在活跃雷暴之上，即使
   它自己的母闪电没被 WWLLN 探到。所以数候选星下点 R 公里、±T 秒内的
   闪电总数，记作雷暴邻近度。

偶然水平不能再用 ±300 s 平移——雷暴持续几小时，平移几分钟根本没破坏它。
这里改用**整日平移**（±1 / ±3 / ±7 天）：位置一动不动、季节不变，只换掉
那一天的天气。热带雷暴逐日复现，所以这个对照是**偏保守的**（会高估偶然
水平），偏保守的方向正好是我们要的。

用法: python3 gbm_storm_proximity.py <in.csv> <out.csv> [半径 km] [半窗 s]
      in.csv 需要列 start,delay,bin_best,lon,lat,alt
"""
import csv
import math
import sys
from datetime import datetime, timedelta, timezone

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

DB = "/gecamfs/Exchange/GSDC/missions/AEfiles/WWLLN.db"
R_EARTH = 6_371_000.0
C = 299_792_458.0
LIGHTNING_ALT = 15_000.0
ASSOC_WINDOW_S = 0.005
ASSOC_MAX_KM = 800.0
DAY_SHIFTS = [1, -1, 3, -3, 7, -7]


def peak_time(row):
    """与 UnifiedSignal::peak_time 一致：start + delay + bin_size_best/2。"""
    body = row["start"].rstrip("Z")
    head, _, frac = body.partition(".")
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
    d2 = ((R_EARTH + h1) ** 2 + (R_EARTH + h2) ** 2
          - 2 * (R_EARTH + h1) * (R_EARTH + h2) * math.cos(alpha))
    return math.sqrt(d2) / C


def fetch(conn, lo, hi):
    return conn.execute(
        "SELECT time, lat, lon FROM lightning WHERE time BETWEEN ? AND ?",
        (lo.strftime("%Y-%m-%d %H:%M:%S.%f"), hi.strftime("%Y-%m-%d %H:%M:%S.%f")),
    ).fetchall()


def associated(conn, t_peak, lat, lon, alt):
    """与 blink_lightning::algorithms::associated 逐行对应。"""
    for ts, llat, llon in fetch(conn, t_peak - timedelta(seconds=1), t_peak + timedelta(seconds=1)):
        dist = great_circle(lat, lon, llat, llon)
        if dist > ASSOC_MAX_KM * 1000:
            continue
        lt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        fixed = t_peak - timedelta(seconds=time_of_arrival(dist, alt, LIGHTNING_ALT))
        if abs((lt - fixed).total_seconds()) <= ASSOC_WINDOW_S:
            return 1
    return 0


def storm_count(conn, t_peak, lat, lon, radius_km, half_window_s):
    lo = t_peak - timedelta(seconds=half_window_s)
    hi = t_peak + timedelta(seconds=half_window_s)
    n = 0
    for _ts, llat, llon in fetch(conn, lo, hi):
        if great_circle(lat, lon, llat, llon) <= radius_km * 1000:
            n += 1
    return n


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    radius_km = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    half_window_s = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0
    rows = list(csv.DictReader(open(in_path)))
    conn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

    fields = ["start", "fa", "count", "bin_best", "lon", "lat", "assoc", "storm"]
    fields += ["storm_d%+d" % d for d in DAY_SHIFTS]
    out = csv.writer(open(out_path, "w", newline=""))
    out.writerow(fields)

    for i, r in enumerate(rows):
        t = peak_time(r)
        lat, lon, alt = float(r["lat"]), float(r["lon"]), float(r["alt"])
        rec = [r["start"], r["fa"], r["count"], r["bin_best"], r["lon"], r["lat"],
               associated(conn, t, lat, lon, alt),
               storm_count(conn, t, lat, lon, radius_km, half_window_s)]
        for d in DAY_SHIFTS:
            rec.append(storm_count(conn, t + timedelta(days=d), lat, lon,
                                   radius_km, half_window_s))
        out.writerow(rec)
        if (i + 1) % 50 == 0:
            print("  %d / %d" % (i + 1, len(rows)), flush=True)
    print("done -> %s  (半径 %.0f km, 半窗 %.0f s)" % (out_path, radius_km, half_window_s))


if __name__ == "__main__":
    main()
