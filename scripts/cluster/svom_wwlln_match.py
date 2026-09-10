"""把闪电证实的 TGF 对上 WWLLN 的实际落点，用真落点（而不是天底）算入射方向。

做流强与光子谱要把 CALDB 的方向相关响应正向折叠，而响应随入射角变化，所以
源方向不能用星下点凑合：实测真落点方向与天底方向差**中位 25.3°、最大 50°**。

关联判据与 `blink wwlln` 一致：±5 ms、离星下点 800 km 以内；同一窗里多个落点
时取距离最近的。TGF 产生高度取 15 km（对方向的影响 < 0.1°）。

姿态约定见 `crates/instruments/blink_svom_grm/src/io/att.rs`：
Q0 是标量、q 是 body→J2000、v_payload = M · R(q)⁻¹ · v_J2000。
JSON 里只存矢量部分，标量按 q0 = +sqrt(1−|v|²) 复原（SVOM 的 Q0 恒非负）。

WWLLN 原始 AE 文件（每天一个 `AEyyyymmdd.loc`，逗号分隔：
日期, 时刻, 纬度, 经度, resid, nstn, ...）。

用法: python3 svom_wwlln_match.py <tgfs.json> <AE 文件目录> <out.csv>
"""
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from scipy.spatial.transform import Rotation

M = Rotation.from_matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]])   # 卫星系 -> 载荷系
RE = 6378.137
DT_MAX = 5e-3
D_MAX = 800.0
H_TGF = 15.0


def parse_iso(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    t = dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return t + dt.timedelta(seconds=float("0." + f) if f else 0.0)


def great_circle(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return RE * np.arccos(np.clip(np.sin(p1) * np.sin(p2)
                                  + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def geodetic_xyz(lat, lon, h_km):
    p, l = np.radians(lat), np.radians(lon)
    r = RE + h_km
    return np.array([r * np.cos(p) * np.cos(l), r * np.cos(p) * np.sin(l), r * np.sin(p)])


def main(tgfs, wwlln_dir, out_csv):
    recs = json.load(open(tgfs))
    conf = [r for r in recs
            if r["signal"]["false_positive_per_year"] <= 1e-5
            and r["lightning"].get("in_coverage", True)
            and r["lightning"].get("associated")
            and r["signal"].get("attitude")]
    print("闪电证实候选 %d" % len(conf))

    by_day = {}
    for r in conf:
        t = parse_iso(r["signal"]["start"])
        by_day.setdefault(t.strftime("%Y%m%d"), []).append((r, t))

    out = []
    for day, items in sorted(by_day.items()):
        path = os.path.join(wwlln_dir, "AE%s.loc" % day)
        if not os.path.exists(path):
            print("缺 WWLLN 文件", day)
            continue
        # AE 文件一天上百万行，按「候选所在的整秒及前后各一秒」先用 grep 粗筛
        pats = set()
        for _, t in items:
            for off in (-1, 0, 1):
                pats.add((t + dt.timedelta(seconds=off)).strftime(",%H:%M:%S"))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(sorted(pats)) + "\n")
            pf = fh.name
        txt = subprocess.run(["grep", "-F", "-f", pf, path],
                             capture_output=True, text=True).stdout
        os.unlink(pf)
        rows = []
        for line in txt.splitlines():
            f = line.split(",")
            if len(f) < 4:
                continue
            try:
                hh, mm, ss = f[1].split(":")
                rows.append((int(hh) * 3600 + int(mm) * 60 + float(ss),
                             float(f[2]), float(f[3])))
            except ValueError:
                continue
        if not rows:
            print("  %s 没抓到落点" % day)
            continue
        arr = np.array(rows)
        for r, t in items:
            t_sec = (t - t.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            pos = r["signal"]["position"]
            near = np.abs(arr[:, 0] - t_sec) <= DT_MAX
            if not near.any():
                print("  %s 无 ±5 ms 落点" % r["signal"]["start"][:23])
                continue
            cand = arr[near]
            dist = great_circle(pos["latitude"], pos["longitude"], cand[:, 1], cand[:, 2])
            k = int(np.argmin(dist))
            if dist[k] > D_MAX:
                print("  %s 最近落点 %.0f km，超出 %.0f km"
                      % (r["signal"]["start"][:23], dist[k], D_MAX))
                continue
            out.append(dict(rec=r, t=t, lat=cand[k, 1], lon=cand[k, 2], dist=dist[k],
                            dt=cand[k, 0] - t_sec, sat_lat=pos["latitude"],
                            sat_lon=pos["longitude"], alt=pos["altitude"] / 1000.0))

    print("对上落点 %d / %d" % (len(out), len(conf)))
    d = np.array([o["dist"] for o in out])
    print("落点到星下点的距离：中位 %.0f km，p90 %.0f km，最大 %.0f km"
          % (np.median(d), np.percentile(d, 90), d.max()))

    times = Time([o["t"] for o in out])
    src = np.array([geodetic_xyz(o["lat"], o["lon"], H_TGF) for o in out])
    sat = np.array([geodetic_xyz(o["sat_lat"], o["sat_lon"], o["alt"]) for o in out])
    vec = src - sat
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    nad = -sat / np.linalg.norm(sat, axis=1)[:, None]

    quats = []
    for o in out:
        a = o["rec"]["signal"]["attitude"]
        v = np.array([a["q1"], a["q2"], a["q3"]])
        quats.append([v[0], v[1], v[2], np.sqrt(max(0.0, 1 - v @ v))])
    R = Rotation.from_quat(np.array(quats))

    def to_payload(v_itrs):
        c = ITRS(CartesianRepresentation(v_itrs.T * u.dimensionless_unscaled), obstime=times)
        g = c.transform_to(GCRS(obstime=times)).cartesian.xyz.value.T
        g /= np.linalg.norm(g, axis=1)[:, None]
        p = M.apply(R.inv().apply(g))
        return p / np.linalg.norm(p, axis=1)[:, None]

    vs, vn = to_payload(vec), to_payload(nad)
    th = np.degrees(np.arccos(np.clip(vs[:, 2], -1, 1)))
    ph = np.degrees(np.arctan2(vs[:, 1], vs[:, 0])) % 360
    sep = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", vs, vn), -1, 1)))
    print("真落点方向与天底方向的夹角：中位 %.1f°，p90 %.1f°，最大 %.1f°"
          % (np.median(sep), np.percentile(sep, 90), sep.max()))
    print("真落点的 θ：中位 %.1f°，θ<90° 的占 %.0f%%" % (np.median(th), 100 * (th < 90).mean()))

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "theta", "phi", "lat_stroke", "lon_stroke",
                    "dist_km", "dt_ms", "sep_nadir_deg"])
        for o, a, b, s in zip(out, th, ph, sep):
            w.writerow([o["rec"]["signal"]["start"], "%.3f" % a, "%.3f" % b,
                        "%.4f" % o["lat"], "%.4f" % o["lon"], "%.1f" % o["dist"],
                        "%.3f" % (o["dt"] * 1e3), "%.2f" % s])
    print("→", out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
