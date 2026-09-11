"""LST 判别量的第 0 步：三颗星各是什么轨道，升交点地方时漂不漂。

为什么这是硬前置。太阳同步轨道（SSO）的定义性质就是升交点地方时被锁死，
所以一颗 SSO 星在给定纬度上过顶的当地太阳时只有两个值（升段、降段）——
LST 覆盖必然有强结构，拿它做成分拟合，拟合出来的 `f_TGF` 是曝光结构不是雷暴。
非同步轨道的升交点地方时会以进动周期走完一整圈，覆盖才谈得上匀。

**判法不查文献，直接量数据**：从 posatt 的 `X/Y/Z_WGS84`（地固系、单位米）
找升交点（Z 由负转正），在那一刻算 LST = UTC + 经度/15，看它随任务时间怎么走。

用法: python3 orbit_probe.py <GECAM-A|GECAM-B|GECAM-C> <采样间隔天> <out.csv>
"""
import glob, os, sys, datetime as dt
import numpy as np
from astropy.io import fits

ROOT = {
    "GECAM-A": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_A", "posatt", "ga"),
    "GECAM-B": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_B", "posatt", "gb"),
    "GECAM-C": ("/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily", None, "POSATT", "gc"),
}
EPOCH = {"GECAM-A": (2019, 1, 1), "GECAM-B": (2019, 1, 1), "GECAM-C": (2021, 1, 1)}

sat, step_days, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
root, sub, pdir, pre = ROOT[sat]
ref = dt.datetime(*EPOCH[sat], tzinfo=dt.timezone.utc)

days = []
for y in sorted(glob.glob(root + "/*")):
    for m in sorted(glob.glob(y + "/*")):
        for d in sorted(glob.glob(m + "/*")):
            p = os.path.join(d, sub, pdir) if sub else os.path.join(d, pdir)
            if os.path.isdir(p):
                days.append((os.path.basename(y), os.path.basename(m),
                             os.path.basename(d), p))
days.sort()
print("%s 有 posatt 的天数 %d" % (sat, len(days)))
days = days[::step_days]
print("按每 %d 天抽一天 -> %d 天" % (step_days, len(days)))

rows = []
for y, m, d, p in days:
    fs = sorted(glob.glob(p + "/%s_posatt_%s%s%s_*.fits" % (pre, y[2:], m, d)))
    got = False
    for f in fs:
        try:
            with fits.open(f, memmap=False) as hd:
                t = hd[1].data
                cols = t.columns.names
                tm = np.asarray(t["TIME"], float)
                x = np.asarray(t["X_WGS84"], float)
                yy = np.asarray(t["Y_WGS84"], float)
                z = np.asarray(t["Z_WGS84"], float)
        except Exception as e:
            continue
        if len(tm) < 60:
            continue
        # 位置解为零或荒唐的行要先剔除：实测 GECAM-B 有整段 X=Y=Z=0 的
        # posatt 行，留着会在 Z=0 上来回穿，一个文件里造出上百个假升交点。
        r = np.sqrt(x * x + yy * yy + z * z)
        ok = (r > 6.5e6) & (r < 7.5e6)
        tm, x, yy, z = tm[ok], x[ok], yy[ok], z[ok]
        if len(tm) < 60:
            continue
        # 升交点：Z 由负转正。一天只留第一个，免得一个文件里的多次穿越
        # 把某一天在直方图里算成上百票。
        up = np.where((z[:-1] < 0) & (z[1:] >= 0))[0][:1]
        for i in up:
            # 线性插值到 Z=0
            w = -z[i] / (z[i + 1] - z[i])
            t0 = tm[i] + w * (tm[i + 1] - tm[i])
            xx = x[i] + w * (x[i + 1] - x[i])
            yz = yy[i] + w * (yy[i + 1] - yy[i])
            lon = np.degrees(np.arctan2(yz, xx))
            stamp = ref + dt.timedelta(seconds=float(t0))
            utc_h = stamp.hour + stamp.minute / 60 + stamp.second / 3600
            lst = (utc_h + lon / 15.0) % 24.0
            rows.append(dict(day="%s-%s-%s" % (y, m, d),
                             mjd=(stamp - dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc)).total_seconds() / 86400,
                             lon=round(float(lon), 3), lst=round(float(lst), 4),
                             alt_km=round(float(np.hypot(np.hypot(x[i], yy[i]), z[i])) / 1000 - 6371, 1)))
            got = True
        if got:
            break
    if not got:
        continue

import csv
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0].keys()))
w.writeheader()
w.writerows(rows)
mjd = np.array([r["mjd"] for r in rows])
lst = np.array([r["lst"] for r in rows])
print("升交点样本 %d 个, 跨 %.0f 天" % (len(rows), mjd.max() - mjd.min()))
print("  升交点 LST 范围 %.2f .. %.2f h" % (lst.min(), lst.max()))
ang = lst / 24 * 2 * np.pi
R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
print("  升交点 LST 圆集中度 R = %.3f  (SSO 应 ~1.0；进动走满一圈应 ~0)" % R)
# 覆盖：8 个 3h 格里有几个有样本
h, _ = np.histogram(lst, bins=np.arange(0, 25, 3))
print("  3h 一格的升交点样本数:", list(h))
print("落盘", out)
