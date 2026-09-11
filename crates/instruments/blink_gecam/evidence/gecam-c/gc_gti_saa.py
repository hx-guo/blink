"""GECAM-C：GTI 缺口落在哪（OPEN-QUESTIONS 第 9 条）。

SVOM 上验过 GTI 缺口全部落在南大西洋异常区，因而不必另立 SAA 掩模。GECAM 的缺口
更多更长，但没验过它们落在哪。这里取每个缺口的中点，用 POSATT 查星下点经纬度，
算偶极磁纬，看是否集中在 SAA。若不是，说明缺口另有来源，曝光核算的口径要重新想。

用法: python3 gc_gti_saa.py <日期 ...>
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
A, F = 6378137.0, 1 / 298.257223563
POLE_LAT, POLE_LON = np.radians(80.65), np.radians(-72.68)


def geodetic(x, y, z):
    e2 = F * (2 - F)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        n = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + alt)))
    n = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


def dipole_lat(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.degrees(np.arcsin(np.clip(np.sin(la) * np.sin(POLE_LAT)
                                        + np.cos(la) * np.cos(POLE_LAT) * np.cos(lo - POLE_LON), -1, 1)))


def newest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


gap_mid, gap_len, day_of = [], [], []
gti_total = file_span = 0.0
hours = 0
for day in sys.argv[1:]:
    d = day.replace("-", "/")
    for hour in range(24):
        grd = newest(f"{ROOT}/{d}/GRD_EVT/*_{hour:02d}_v*.fits")
        if grd is None:
            continue
        with fits.open(grd, memmap=True) as hdus:
            head = hdus[0].header
            start, stop = None, None
            s = e = np.array([])
            for hdu in hdus:
                if hdu.name == "GTI":
                    s = np.atleast_1d(np.asarray(hdu.data["START"], float))
                    e = np.atleast_1d(np.asarray(hdu.data["STOP"], float))
                if "TSTART" in hdu.header:
                    start, stop = float(hdu.header["TSTART"]), float(hdu.header["TSTOP"])
        if s.size == 0:
            continue
        order = np.argsort(s)
        s, e = s[order], e[order]
        # 合并重叠段（每探头一份 GTI 时会重复）
        merged = []
        for a, b in zip(s, e):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if not merged:
            continue
        merged = np.asarray(merged, float).reshape(-1, 2)
        hours += 1
        gti_total += (merged[:, 1] - merged[:, 0]).sum()
        file_span += (stop - start) if start is not None else 3600.0
        if start is not None:
            edges = np.concatenate([[start], merged[:, 1]])
            ends = np.concatenate([merged[:, 0], [stop]])
        else:
            edges, ends = merged[:-1, 1], merged[1:, 0]
        for a, b in zip(edges, ends):
            if b - a > 1.0:
                gap_mid.append((a + b) / 2)
                gap_len.append(b - a)
                day_of.append(day)

gap_mid = np.array(gap_mid)
gap_len = np.array(gap_len)
print("小时文件 %d 个；GTI 合计 %.0f s / 文件跨度 %.0f s → 占空比 %.1f%%"
      % (hours, gti_total, file_span, gti_total / file_span * 100))
print("缺口 %d 个，总时长 %.0f s（占跨度 %.1f%%）"
      % (gap_len.size, gap_len.sum(), gap_len.sum() / file_span * 100))
print("缺口时长分位 5/25/50/75/95: %s s"
      % np.round(np.percentile(gap_len, [5, 25, 50, 75, 95]), 1))

# 查每个缺口中点的星下点
lats, lons, kept_len = [], [], []
cache = {}
for day, mid, length in zip(day_of, gap_mid, gap_len):
    d = day.replace("-", "/")
    found = None
    for hour in range(24):
        key = (day, hour)
        if key not in cache:
            pa = newest(f"{ROOT}/{d}/POSATT/*_{hour:02d}_v*.fits")
            if pa is None:
                cache[key] = None
            else:
                with fits.open(pa, memmap=True) as hdus:
                    t = hdus["Orbit_Attitude"].data
                    cache[key] = (np.asarray(t["TIME"], float),
                                  np.asarray(t["X_WGS84"], float),
                                  np.asarray(t["Y_WGS84"], float),
                                  np.asarray(t["Z_WGS84"], float))
        got = cache[key]
        if got is None:
            continue
        if got[0][0] - 2 <= mid <= got[0][-1] + 2:
            i = int(np.argmin(np.abs(got[0] - mid)))
            if abs(got[0][i] - mid) <= 2:
                found = geodetic(got[1][i], got[2][i], got[3][i])
            break
    if found is not None:
        lats.append(found[0])
        lons.append(found[1])
        kept_len.append(length)

lats, lons, kept_len = np.array(lats), np.array(lons), np.array(kept_len)
print("\n定到位置的缺口 %d / %d" % (lats.size, gap_len.size))
if lats.size == 0:
    sys.exit()
mlat = np.abs(dipole_lat(lats, lons))
saa = (lons > -90) & (lons < 40) & (lats > -40) & (lats < 0)
print("  |偶极磁纬| 中位 %.1f°，分位 5/25/50/75/95 = %s"
      % (np.median(mlat), np.round(np.percentile(mlat, [5, 25, 50, 75, 95]), 1)))
print("  落在经典 SAA 框（lon −90..40, lat −40..0）内: %.1f%%" % (saa.mean() * 100))
print("  |磁纬| > 55° 的（极区）: %.1f%%" % ((mlat > 55).mean() * 100))
print("  |磁纬| < 30° 的（低纬）: %.1f%%" % ((mlat < 30).mean() * 100))
print("  纬度分位 5/25/50/75/95 = %s" % np.round(np.percentile(lats, [5, 25, 50, 75, 95]), 1))
print("  经度分位 5/25/50/75/95 = %s" % np.round(np.percentile(lons, [5, 25, 50, 75, 95]), 1))
# 长缺口和短缺口分开看
for label, m in (("长缺口 (>60 s)", kept_len > 60), ("短缺口 (<=60 s)", kept_len <= 60)):
    if m.sum() < 3:
        continue
    print("  %s n=%d: SAA 框内 %.0f%%，|磁纬| 中位 %.1f°，>55° 占 %.0f%%"
          % (label, m.sum(), saa[m].mean() * 100, np.median(mlat[m]), (mlat[m] > 55).mean() * 100))
