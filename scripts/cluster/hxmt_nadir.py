#!/usr/bin/env python3
"""算每个候选时刻天底方向在卫星本体系里的方位，与占优机箱做列联表。

两个约定都是实测定死的，不是照搬文档：

* 四元数（att_convention.py）：q = (Q1, Q2, Q3, q0)，q0 = sqrt(1 - Q1^2 - Q2^2 - Q3^2)，
  标量在后；R 的列就是本体轴在 ECI 里的方向，本体 +X 是 ATT_Pointing 那条望远镜
  指向轴（与表里 Ra/Dec 的中位夹角 0.0000 deg）。ECI -> body 于是是 R^T。
* 轨道（orbit_frame.py）：Orbit 表的 X/Y/Z 是地固系 —— atan2(Y,X) 与表里 Lon
  逐点相同（中位差 0.0000 deg）。所以要先绕 Z 转 +GMST 才进 ECI。

输出每个候选：
  theta = 天底方向与本体 +X（望远镜指向）的夹角
  phi   = 天底方向绕 +X 的方位角 = atan2(z_body, y_body)
"""
import csv, sys, os, collections
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.io import fits
from astropy.time import Time

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
K1 = os.environ.get("HXMT_1K_DIR", "/hxmt/work/HXMT-DATA/1K")
LAUNCH = datetime(2017, 6, 15, tzinfo=timezone.utc)


def met_of(iso):
    """9 位纳秒的 ISO 串 -> MET，绕开 datetime 的微秒截断。"""
    iso = iso.rstrip("Z")
    head, _, frac = iso.partition(".")
    base = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (base - MET_EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def folder(dt):
    num = (dt.replace(hour=0, minute=0, second=0, microsecond=0) - LAUNCH).days + 1
    return "%s/Y%04d%02d/%04d%02d%02d-%04d" % (K1, dt.year, dt.month,
                                               dt.year, dt.month, dt.day, num)


def pick(folder_path, prefix):
    if not os.path.isdir(folder_path):
        return None
    best, bestv = None, -1
    for name in os.listdir(folder_path):
        if name.startswith(prefix):
            try:
                v = int(name[len(prefix):len(prefix) + 1])
            except ValueError:
                v = 0
            if v > bestv:
                best, bestv = name, v
    return os.path.join(folder_path, best) if best else None


def load_hour(dt):
    f = folder(dt)
    stem = "HXMT_%04d%02d%02dT%02d_" % (dt.year, dt.month, dt.day, dt.hour)
    att, orb = pick(f, stem + "Att_FFFFFF_V"), pick(f, stem + "Orbit_FFFFFF_V")
    if not att or not orb:
        return None
    try:
        with fits.open(att) as h:
            q = h["ATT_Quater"].data
            ta = np.array(q["Time"], float)
            qq = np.stack([np.array(q["Q%d" % i], float) for i in (1, 2, 3)], 1)
        with fits.open(orb) as h:
            o = h["Orbit"].data
            to = np.array(o["Time"], float)
            xyz = np.stack([np.array(o[c], float) for c in ("X", "Y", "Z")], 1)
            lla = np.stack([np.array(o[c], float) for c in ("Lon", "Lat", "Alt")], 1)
    except Exception as exc:                       # 文件坏了就跳过，别整批停
        print("  bad hour %s: %s" % (dt, exc), flush=True)
        return None
    if len(ta) < 2 or len(to) < 2:
        return None
    return ta, qq, to, xyz, lla


def rot(q1, q2, q3):
    """本体 -> ECI 的旋转矩阵（列 = 本体轴在 ECI 的方向）。"""
    a = np.sqrt(max(0.0, 1.0 - q1 * q1 - q2 * q2 - q3 * q3))
    b, c, d = q1, q2, q3
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),         2*(b*d + a*c)],
        [2*(b*c + a*d),         a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),         2*(c*d + a*b),         a*a - b*b - c*c + d*d],
    ])


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(src)))
    mets = np.array([met_of(r["start"]) for r in rows])
    gmst = Time(MET_EPOCH.timestamp() + mets, format="unix", scale="utc") \
        .sidereal_time("mean", "greenwich").rad

    by_hour = collections.defaultdict(list)
    for i, m in enumerate(mets):
        dt = MET_EPOCH + timedelta(seconds=float(m))
        by_hour[dt.replace(minute=0, second=0, microsecond=0)].append(i)
    print("%d rows, %d distinct hours" % (len(rows), len(by_hour)), flush=True)

    res = [None] * len(rows)
    missing = 0
    for k, (hour, items) in enumerate(sorted(by_hour.items())):
        data = load_hour(hour)
        if data is None:
            missing += 1
            continue
        ta, qq, to, xyz, lla = data
        for i in items:
            m = mets[i]
            if not (ta[0] <= m <= ta[-1] and to[0] <= m <= to[-1]):
                continue
            q = [np.interp(m, ta, qq[:, j]) for j in range(3)]
            p = np.array([np.interp(m, to, xyz[:, j]) for j in range(3)])
            ll = [np.interp(m, to, lla[:, j]) for j in range(3)]
            g = gmst[i]
            cg, sg = np.cos(g), np.sin(g)
            p_eci = np.array([cg * p[0] - sg * p[1], sg * p[0] + cg * p[1], p[2]])
            nadir = -p_eci / np.linalg.norm(p_eci)
            nb = rot(*q).T @ nadir
            res[i] = (np.degrees(np.arccos(np.clip(nb[0], -1, 1))),
                      np.degrees(np.arctan2(nb[2], nb[1])) % 360.0,
                      nb[0], nb[1], nb[2], ll[0], ll[1], ll[2])
        if (k + 1) % 250 == 0:
            print("  %d/%d hours" % (k + 1, len(by_hour)), flush=True)

    hdr = list(rows[0].keys()) + ["nadir_theta", "nadir_phi", "nbx", "nby", "nbz",
                                  "olon", "olat", "oalt"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r, v in zip(rows, res):
            w.writerow(list(r.values()) +
                       ([""] * 8 if v is None else ["%.6f" % x for x in v]))
    print("missing hours %d, resolved %d/%d -> %s"
          % (missing, sum(v is not None for v in res), len(rows), out))


if __name__ == "__main__":
    main()
