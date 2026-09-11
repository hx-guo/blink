#!/usr/bin/env python3
"""整链路验证：给几个整小时逐秒算天底方向在本体系里的位置。

两条物理判据：
  1. GMST 符号对了，望远镜才不会老盯着地球 —— 550 km 高度地球边缘张角 ~70 deg，
     指向轴与天底的夹角应当大多 > 70 deg。符号反了就会大量 < 70。
  2. 惯性指向的卫星，天底在本体系里必须一轨转一圈（~95 min）。若真"近乎固定"，
     theta/phi 的散布会很小。

用法: nadir_check.py <Att.FITS> <Orbit.FITS> [...成对]
"""
import sys
from datetime import datetime, timezone
import numpy as np
from astropy.io import fits
from astropy.time import Time

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)


def rot(q1, q2, q3):
    a = np.sqrt(max(0.0, 1.0 - q1 * q1 - q2 * q2 - q3 * q3))
    b, c, d = q1, q2, q3
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
    ])


def run(att_path, orb_path):
    with fits.open(att_path) as h:
        q = h["ATT_Quater"].data
        ta = np.array(q["Time"], float)
        qq = np.stack([np.array(q["Q%d" % i], float) for i in (1, 2, 3)], 1)
        p = h["ATT_Pointing"].data
        ra, dec = np.array(p["Ra"], float), np.array(p["Dec"], float)
    with fits.open(orb_path) as h:
        o = h["Orbit"].data
        to = np.array(o["Time"], float)
        xyz = np.stack([np.array(o[c], float) for c in ("X", "Y", "Z")], 1)
        alt = np.array(o["Alt"], float)

    t = to[(to >= ta[0]) & (to <= ta[-1])]
    gmst = Time(MET_EPOCH.timestamp() + t, format="unix", scale="utc") \
        .sidereal_time("mean", "greenwich").rad

    out = {}
    for sign, name in ((+1, "GMST +（ECEF->ECI，应当对）"), (-1, "GMST −（对照，应当错）")):
        th, ph = [], []
        for k, tt in enumerate(t):
            qv = [np.interp(tt, ta, qq[:, j]) for j in range(3)]
            pv = np.array([np.interp(tt, to, xyz[:, j]) for j in range(3)])
            g = sign * gmst[k]
            cg, sg = np.cos(g), np.sin(g)
            pe = np.array([cg * pv[0] - sg * pv[1], sg * pv[0] + cg * pv[1], pv[2]])
            nb = rot(*qv).T @ (-pe / np.linalg.norm(pe))
            th.append(np.degrees(np.arccos(np.clip(nb[0], -1, 1))))
            ph.append(np.degrees(np.arctan2(nb[2], nb[1])) % 360.0)
        out[name] = (np.array(th), np.array(ph))

    horizon = np.degrees(np.arcsin(6371.0 / (6371.0 + np.median(alt) / 1000.0)))
    print("  指向 Ra %.3f..%.3f  Dec %.3f..%.3f （这一小时移动 %.3f deg）"
          % (ra.min(), ra.max(), dec.min(), dec.max(),
             np.hypot((ra.max() - ra.min()) * np.cos(np.radians(dec.mean())),
                      dec.max() - dec.min())))
    print("  高度 %.1f km -> 地球半张角 %.1f deg，地平线在 theta = %.1f deg"
          % (np.median(alt) / 1000.0, horizon, 180.0 - horizon))
    for name, (th, ph) in out.items():
        below = (th < horizon).mean()
        print("  %s: theta %.1f..%.1f (中位 %.1f)  phi 跨度 %.0f deg  指向轴被地球遮挡的时间占比 %.1f%%"
              % (name, th.min(), th.max(), np.median(th), ph.max() - ph.min(), 100 * below))


for i in range(1, len(sys.argv), 2):
    print("=" * 78)
    print(sys.argv[i].split("/")[-1])
    run(sys.argv[i], sys.argv[i + 1])
