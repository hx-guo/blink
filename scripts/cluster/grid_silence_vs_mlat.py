"""GRID-03B 单路沉默的发生率随 |偶极磁纬| 的变化。

若沉默由重离子打穿晶体、前端饱和后基线恢复造成，发生率应随截止刚度下降（高磁纬）而强烈上升。
沉默的定义与 grid_channel_silences.py 相同（该路 r_k·L > 20，四路合计前后各 1 s < 1500 c/s）。
位置取自 fits8 位姿文件（取有有限经纬度的最新版本），逐秒线性插值；没有位置的秒不计入。

输出每个 worker 一张表：|偶极磁纬| 每 5° 一档，该档安静且有位置的秒数、四路合计沉默段数、
该档四路合计平均计数率。

用法: python3 grid_silence_vs_mlat.py <worker> <nworkers> <抽几天> <输出 CSV>
"""
import csv, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID/GRID-03B"
QUIET, K = 1500.0, 20.0
POLE_LAT, POLE_LON = np.radians(80.7), np.radians(-72.7)
BINS = np.arange(0, 95, 5)


def mlat(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    s = np.sin(la) * np.sin(POLE_LAT) + np.cos(la) * np.cos(POLE_LAT) * np.cos(lo - POLE_LON)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def positions(day):
    T, LA, LO = [], [], []
    base = day.replace("/fits7/", "/fits8/")
    for vd in sorted(glob.glob(base + "/posatt*"))[::-1]:
        for f in glob.glob(vd + "/*.fits"):
            try:
                d = fits.getdata(f, 1)
            except Exception:
                continue
            la = np.asarray(d["Latitude"], float); ok = np.isfinite(la)
            if ok.any():
                T.append(np.asarray(d["TIME"], float)[ok]); LA.append(la[ok]); LO.append(np.asarray(d["Longitude"], float)[ok])
        if T:
            break
    if not T:
        return None
    t = np.concatenate(T); o = np.argsort(t)
    return t[o], np.concatenate(LA)[o], np.concatenate(LO)[o]


def at(pos, x):
    t, la, lo = pos
    ok = (x >= t[0]) & (x <= t[-1])
    # 经度跨 ±180 时按单位向量插值
    ex, ey = np.cos(np.radians(lo)), np.sin(np.radians(lo))
    lat = np.interp(x, t, la); lon = np.degrees(np.arctan2(np.interp(x, t, ey), np.interp(x, t, ex)))
    # 采样间隔大于 30 s 的地方不插
    j = np.clip(np.searchsorted(t, x), 1, len(t) - 1)
    ok &= (t[j] - t[j - 1]) <= 30
    return np.where(ok, np.abs(mlat(lat, lon)), np.nan)


w, nw, ndays, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
days = sorted(glob.glob(f"{G}/fits7/20*/*/*"))
pick = [days[int(round(k * (len(days) - 1) / max(ndays - 1, 1)))] for k in range(ndays)]
quiet_s = np.zeros(len(BINS) - 1); nsil = np.zeros(len(BINS) - 1); cnt = np.zeros(len(BINS) - 1)
for d in pick[w::nw]:
    pos = positions(d)
    if pos is None:
        continue
    vs = sorted(os.listdir(d))
    for f in sorted(glob.glob(f"{d}/{vs[-1]}/*.fits")):
        try:
            with fits.open(f, memmap=False) as h:
                gs, ge = float(h["GTI"].data["START"][0]), float(h["GTI"].data["STOP"][0])
                ev = [np.asarray(h[f"EVENTS{k}"].data["TIME"], float) for k in range(4)]
        except Exception:
            continue
        ev = [t[(t >= gs) & (t <= ge)] for t in ev]
        tall = np.sort(np.concatenate(ev))
        if tall.size < 1000:
            continue
        edges = np.arange(np.floor(gs) + 1, np.floor(ge) - 1)
        if edges.size < 3:
            continue
        c = np.histogram(tall, edges)[0]
        m = at(pos, edges[:-1] + 0.5)
        q = (c < QUIET) & np.isfinite(m)
        b = np.digitize(m[q], BINS) - 1
        np.add.at(quiet_s, b, 1); np.add.at(cnt, b, c[q])
        for t in ev:
            if t.size < 100:
                continue
            dt = np.diff(t)
            for i in np.flatnonzero(dt > 0.05):
                a, e = t[i], t[i + 1]
                if a - gs < 1.0 or ge - e < 1.0:
                    continue
                rk = (i + 1 - np.searchsorted(t, a - 1.0) + np.searchsorted(t, e + 1.0, side="right") - (i + 1)) / 2.0
                tb = np.searchsorted(tall, a) - np.searchsorted(tall, a - 1.0)
                ta = np.searchsorted(tall, e + 1.0) - np.searchsorted(tall, e)
                if rk * (e - a) <= K or max(tb, ta) >= QUIET:
                    continue
                mm = at(pos, np.array([a]))[0]
                if np.isfinite(mm):
                    nsil[min(np.digitize(mm, BINS) - 1, len(BINS) - 2)] += 1
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["mlat_lo", "quiet_s", "n_silence", "counts"])
    for i in range(len(BINS) - 1):
        wr.writerow([BINS[i], int(quiet_s[i]), int(nsil[i]), int(cnt[i])])
print("worker", w, "quiet_s", int(quiet_s.sum()), "silences", int(nsil.sum()))
