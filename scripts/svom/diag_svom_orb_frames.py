"""orb 表同时给了 J2000 与 WGS84 两套位置——用它直接检验方向链里的地固→惯性这一步。

`svom_wwlln_match.py` 把 WWLLN 落点（经纬度，地固）与卫星位置（经纬高，地固）
相减得到源方向，再用 astropy 的 ITRS→GCRS 转到惯性系。这一步若有系统误差，
TGF 的方位角就会错，而 GRB 250919A（直接用 RA/Dec）不受影响。

orb 表自带同一时刻的 X_J2000 与 X_WGS84，是任务方自己算的，可以当真值：
  1. 把 X_WGS84 用 astropy ITRS→GCRS 转过去，比 X_J2000，看差多少度；
  2. 把 LON/LAT/ALT 反算成地固直角坐标，比 X_WGS84，看经度约定对不对。
"""
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.io import fits
from astropy.time import Time

REF_MJD = 57754 + 0.000800740741       # MJDREFI + MJDREFF，TT
RE_A = 6378.137
RE_F = 1.0 / 298.257223563


def geodetic_xyz(lat, lon, h):
    p, l = np.radians(lat), np.radians(lon)
    e2 = RE_F * (2 - RE_F)
    n = RE_A / np.sqrt(1 - e2 * np.sin(p) ** 2)
    return np.array([(n + h) * np.cos(p) * np.cos(l),
                     (n + h) * np.cos(p) * np.sin(l),
                     (n * (1 - e2) + h) * np.sin(p)]).T


def spherical_xyz(lat, lon, h):
    p, l = np.radians(lat), np.radians(lon)
    r = RE_A + h
    return np.array([r * np.cos(p) * np.cos(l), r * np.cos(p) * np.sin(l),
                     r * np.sin(p)]).T


def main(path, n=400):
    with fits.open(path) as h:
        d = h["ORB"].data
        k = np.linspace(0, len(d) - 1, min(n, len(d))).astype(int)
        t = np.asarray(d["TIME"], float)[k]
        j = np.column_stack([d["X_J2000"], d["Y_J2000"], d["Z_J2000"]])[k]
        w = np.column_stack([d["X_WGS84"], d["Y_WGS84"], d["Z_WGS84"]])[k]
        lon = np.asarray(d["LON"], float)[k]
        lat = np.asarray(d["LAT"], float)[k]
        alt = np.asarray(d["ALT"], float)[k]

    times = Time(REF_MJD + t / 86400.0, format="mjd", scale="tt")
    c = ITRS(CartesianRepresentation((w / np.linalg.norm(w, axis=1)[:, None]).T
                                     * u.dimensionless_unscaled), obstime=times)
    g = c.transform_to(GCRS(obstime=times)).cartesian.xyz.value.T
    g /= np.linalg.norm(g, axis=1)[:, None]
    jn = j / np.linalg.norm(j, axis=1)[:, None]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", g, jn), -1, 1)))
    print("astropy ITRS→GCRS 的结果 vs orb 表自带的 J2000：")
    print("  夹角 中位 %.4f°，p90 %.4f°，最大 %.4f°" % (np.median(ang),
                                                  np.percentile(ang, 90), ang.max()))
    print("  |r| J2000 %.3f km，WGS84 %.3f km（应当一样）"
          % (np.linalg.norm(j, axis=1).mean(), np.linalg.norm(w, axis=1).mean()))

    for tag, f in (("WGS84 椭球", geodetic_xyz), ("球面近似", spherical_xyz)):
        v = f(lat, lon, alt)
        vn = v / np.linalg.norm(v, axis=1)[:, None]
        wn = w / np.linalg.norm(w, axis=1)[:, None]
        a2 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", vn, wn), -1, 1)))
        print("  LON/LAT/ALT 按%s反算 vs X_WGS84：夹角中位 %.4f°，最大 %.4f°；"
              "|r| 差中位 %.3f km"
              % (tag, np.median(a2), a2.max(),
                 np.median(np.linalg.norm(v, axis=1) - np.linalg.norm(w, axis=1))))
    # 经度符号翻转的对照
    v = spherical_xyz(lat, -lon, alt)
    vn = v / np.linalg.norm(v, axis=1)[:, None]
    wn = w / np.linalg.norm(w, axis=1)[:, None]
    a3 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", vn, wn), -1, 1)))
    print("  对照：经度取反后 夹角中位 %.1f°" % np.median(a3))
    # 时间尺度取错（TT 当 UTC）的对照：差 69.184 s ≈ 地球转 0.29°
    t2 = Time(REF_MJD + t / 86400.0, format="mjd", scale="utc")
    c2 = ITRS(CartesianRepresentation((w / np.linalg.norm(w, axis=1)[:, None]).T
                                      * u.dimensionless_unscaled), obstime=t2)
    g2 = c2.transform_to(GCRS(obstime=t2)).cartesian.xyz.value.T
    g2 /= np.linalg.norm(g2, axis=1)[:, None]
    a4 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", g2, jn), -1, 1)))
    print("  对照：obstime 用 UTC 而不是 TT，夹角中位 %.4f°" % np.median(a4))


if __name__ == "__main__":
    main(sys.argv[1])
