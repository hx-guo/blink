#!/usr/bin/env python3
"""Orbit 表的 X/Y/Z 是 ECI 还是 ECEF：拿 atan2(Y,X) 跟表里的 Lon 比，再跟扣掉 GMST 的比。"""
import sys
from datetime import datetime, timezone
import numpy as np
from astropy.io import fits
from astropy.time import Time

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)

with fits.open(sys.argv[1]) as h:
    o = h["Orbit"].data
    cols = h["Orbit"].columns
    print("列格式:", ", ".join("%s[%s]%s" % (c.name, c.format, (" TSCAL=%s TZERO=%s" % (c.bscale, c.bzero)) if (c.bscale not in (None, 1) or c.bzero not in (None, 0)) else "") for c in cols))
    t = np.array(o["Time"], float)
    X, Y, Z = (np.array(o[c], float) for c in ("X", "Y", "Z"))
    lon, lat, alt = (np.array(o[c], float) for c in ("Lon", "Lat", "Alt"))

print("样本数 %d  Time %.1f..%.1f" % (len(t), t[0], t[-1]))
print("X 范围 %.4g..%.4g   |r| 中位 %.6g" % (X.min(), X.max(), np.median(np.sqrt(X**2 + Y**2 + Z**2))))
print("Lon %.4g..%.4g  Lat %.4g..%.4g  Alt %.4g..%.4g" % (lon.min(), lon.max(), lat.min(), lat.max(), alt.min(), alt.max()))

utc = Time([MET_EPOCH.timestamp() + x for x in t], format="unix", scale="utc")
gmst = utc.sidereal_time("mean", "greenwich").deg

lon_xy = np.degrees(np.arctan2(Y, X)) % 360.0
lon_tab = lon % 360.0


def arc(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


print("\natan2(Y,X) 与表里 Lon 的差: 中位 %+.4f deg, 散布 %.4f" % (np.median(arc(lon_xy, lon_tab)), arc(lon_xy, lon_tab).std()))
print("atan2(Y,X)-GMST 与 Lon 的差: 中位 %+.4f deg, 散布 %.4f" % (np.median(arc(lon_xy - gmst, lon_tab)), arc(lon_xy - gmst, lon_tab).std()))
# 地心纬度 vs 表里的纬度（大地纬度），差最多 0.19 deg
latc = np.degrees(np.arcsin(Z / np.sqrt(X**2 + Y**2 + Z**2)))
print("地心纬度与表里 Lat 的差: 中位 %+.4f deg, 最大 |%.4f|" % (np.median(latc - lat), np.abs(latc - lat).max()))
