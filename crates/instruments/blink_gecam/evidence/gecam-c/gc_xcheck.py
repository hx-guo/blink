"""已发表 GECAM-B TGF 里落在 GECAM-C 在轨之后的 4 个：C 星当时在哪、有没有数据、GTI 内不内。

TGF 的地面足迹半径约 800 km，两颗星要同时看到必须都在这个圈里。C 星在 SATech-01 上，
与 A/B 不同轨，共视是碰运气，但 4 次机会不查白不查。
"""

import datetime as dt
import glob

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
A = 6378137.0
F = 1 / 298.257223563

EVENTS = [
    ("2022-07-30T19:47:13.029698", -78.355322, 5.691025),
    ("2022-08-07T01:54:38.115447", -70.856822, 9.046319),
    ("2022-08-07T13:18:08.971936", 144.205448, -6.964714),
    ("2022-08-30T11:53:55.535126", -62.498890, 21.592573),
]


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def geodetic(x, y, z):
    """WGS84 直角坐标（米）→ 纬度、经度、高度。"""
    e2 = F * (2 - F)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        n = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + alt)))
    n = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def great_circle(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def pick(iso, kind):
    directory = f"{ROOT}/{iso[:10].replace('-', '/')}/{kind}"
    files = sorted(glob.glob(f"{directory}/*_{iso[11:13]}_v*.fits"))
    return files[-1] if files else None


for iso, blon, blat in EVENTS:
    t = met(iso)
    print("\n== 已发表 TGF %s  GECAM-B 星下点 (%.3f, %.3f)" % (iso, blon, blat))
    pa = pick(iso, "POSATT")
    if pa is None:
        print("   GECAM-C 该小时无 POSATT 文件")
        continue
    with fits.open(pa) as hdus:
        d = hdus["Orbit_Attitude"].data
        pt = np.asarray(d["TIME"], float)
        i = int(np.argmin(np.abs(pt - t)))
        if abs(pt[i] - t) > 5:
            print("   POSATT 最近点差 %.1f s，位置不可用" % abs(pt[i] - t))
            continue
        lat, lon, alt = geodetic(float(d["X_WGS84"][i]), float(d["Y_WGS84"][i]), float(d["Z_WGS84"][i]))
    dist = great_circle(blat, blon, lat, lon)
    print("   GECAM-C 星下点 (%.3f, %.3f)  高度 %.1f km  与 B 星下点相距 **%.0f km**"
          % (lon, lat, alt / 1e3, dist))
    grd = pick(iso, "GRD_EVT")
    if grd is None:
        print("   GECAM-C 该小时无 GRD 事例文件")
        continue
    with fits.open(grd, memmap=True) as hdus:
        gti = None
        for hdu in hdus:
            if hdu.name == "GTI":
                gti = (np.asarray(hdu.data["START"], float), np.asarray(hdu.data["STOP"], float))
        inside = bool(np.any((t >= gti[0]) & (t <= gti[1]))) if gti is not None else None
        # 该时刻附近 ±5 ms 的 GRD 计数（过 keep）
        n = 0
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            dd = hdu.data
            tt = np.asarray(dd["TIME"], float)
            m = (tt >= t - 5e-3) & (tt <= t + 5e-3)
            if not m.any():
                continue
            pi = np.asarray(dd["PI"])[m]
            n += int(((np.asarray(dd["EVT_TYPE"])[m] == 1) & (pi >= 54) & (pi < 448)).sum())
    print("   GECAM-C 有 GRD 数据；该时刻在 GTI 内: %s；±5 ms 内 keep 计数 %d（10 ms 本底期望约 15）"
          % (inside, n))
