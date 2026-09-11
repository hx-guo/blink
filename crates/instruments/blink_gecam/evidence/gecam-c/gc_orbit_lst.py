"""GECAM-C（HEBS / 创新X SATech-01）：升交点地方时随任务时间怎么走，以及逐磁纬带的
当地太阳时曝光分不分得开。

这是「用当地太阳时（LST）的雷暴日变化去估候选里真 TGF 占比」那条判别量的**第 0 步
前置检验**，不是判别量本身。太阳同步轨道的升交点地方时被轨道设计锁死，一颗 SSO 星在
给定纬度上过顶的 LST 就是两个值（升段一个、降段一个）——**曝光在 LST 上必然有强结构，
那条判别量就用不了**。非同步轨道的轨道面会进动，LST 走遍一圈，曝光才可能平。

两个量都只用 POSATT，不碰事例流：

* **升交点地方时**：卫星纬度由负转正那一刻的平地方时 `LST = UT + 经度/15`。
  （平地方时与真太阳时差一个时差项，全年 ≤ 16 分钟，判进不进动绰绰有余。）
* **逐磁纬带的 LST 曝光直方**：按固定间隔抽采样点，算 LST 与偶极磁纬，看每条磁纬带
  里 LST 铺不铺得开。不平就是不能用，**不平本身就是结果，不要去硬拟合**。

用法: gc_orbit_lst.py <输出 CSV> [采样步长秒=60]
"""

import csv
import datetime as dt
import glob
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
A, F = 6378137.0, 1 / 298.257223563
POLE_LAT, POLE_LON = np.radians(80.65), np.radians(-72.68)

# POSATT 坏行的判据：**地心距落在物理高度带之外**（这里取 300–900 km，
# 覆盖任何近地轨道，不是按 C 星实测调出来的窗）。
#
# **`|r| > 0` 或 `|r| > 1e6` 这类判据不够。** GECAM-B 实测一天 75,971 行里 1,565
# 行是坏的而**精确为零的一行都没有**；大多是反常规格化浮点垃圾（`|r| > 1e6`
# 挡得住），但**有 36 行的 |r| 落在 1362 / 3150 / 5434 / 9060 km，外加一行
# 2.49e22 m，全都通过了 `|r| > 1e6`**。后果是那天算出 |磁纬| 最大 87.13°，
# 而 B 星倾角只有 29°。零行造的是**假升交点**，这批造的是**假高纬点**——
# 不改升交点，改的是纬度分档的归属。两种都要挡。
R_MIN, R_MAX = A + 300e3, A + 900e3
# 升交点插值只认时间上真正相邻的两点。坏行剔掉之后数组会留缝，跨缝插值会
# 造出既非升交也非降交的假交点。
MAX_NODE_GAP_SECONDS = 30.0


def geodetic(x, y, z):
    e2 = F * (2 - F)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        n = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + alt)))
    return np.degrees(lat), np.degrees(lon)


def dipole_lat(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.degrees(
        np.arcsin(
            np.clip(
                np.sin(la) * np.sin(POLE_LAT)
                + np.cos(la) * np.cos(POLE_LAT) * np.cos(lo - POLE_LON),
                -1,
                1,
            )
        )
    )


def local_solar_time(met, lon_deg):
    """平地方时（小时）。MET → UTC 当天的秒，再加经度带来的偏移。"""
    utc_seconds = (met + EPOCH.timestamp()) % 86400.0
    return (utc_seconds / 3600.0 + lon_deg / 15.0) % 24.0


def newest(pattern):
    files = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_v", 1)[-1][:2]))
    return files[-1] if files else None


def main():
    out = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    days = sorted({p.rsplit("/POSATT", 1)[0] for p in glob.glob(f"{ROOT}/*/*/*/POSATT")})
    print(f"有 POSATT 的天 {len(days)} 天：{days[0][-10:]} .. {days[-1][-10:]}", flush=True)

    writer = csv.writer(open(out, "w", newline=""), lineterminator="\n")
    writer.writerow(["kind", "day", "met", "lat", "lon", "lst_h", "mlat"])

    nodes = []          # (met, lst) 升交点
    samples = []        # (lst, mlat)
    dropped = kept = 0  # POSATT 坏行 / 好行
    for k, day_dir in enumerate(days):
        tag = day_dir[-10:].replace("/", "-")
        # 一天里取三个小时就够定升交点漂移（一圈约 95 分钟，一小时至少半圈）
        for hour in (0, 8, 16):
            path = newest(f"{day_dir}/POSATT/*_{hour:02d}_v*.fits")
            if path is None:
                continue
            try:
                with fits.open(path, memmap=True) as hdus:
                    table = hdus["Orbit_Attitude"].data
                    met = np.asarray(table["TIME"], float)
                    x = np.asarray(table["X_WGS84"], float)
                    y = np.asarray(table["Y_WGS84"], float)
                    z = np.asarray(table["Z_WGS84"], float)
            except Exception:
                continue
            if met.size < 100:
                continue
            # 坏行先剔掉，再算任何几何量
            radius = np.sqrt(x * x + y * y + z * z)
            good = np.isfinite(radius) & (radius >= R_MIN) & (radius <= R_MAX)
            dropped += int((~good).sum())
            kept += int(good.sum())
            if good.sum() < 100:
                continue
            met, x, y, z = met[good], x[good], y[good], z[good]
            lat, lon = geodetic(x, y, z)
            lst = local_solar_time(met, lon)
            mlat = dipole_lat(lat, lon)

            # 升交点：纬度由负转正的相邻两点，线性插到 lat = 0。
            # 剔掉坏行之后数组会留缝，跨缝的那一对不是真的相邻，不能插。
            up = np.flatnonzero(
                (lat[:-1] < 0)
                & (lat[1:] >= 0)
                & (met[1:] - met[:-1] <= MAX_NODE_GAP_SECONDS)
            )
            for i in up:
                w = -lat[i] / (lat[i + 1] - lat[i])
                m = met[i] + w * (met[i + 1] - met[i])
                # 经度在 ±180 处会跳，插值前先解缠
                a, b = lon[i], lon[i + 1]
                if b - a > 180:
                    b -= 360
                elif b - a < -180:
                    b += 360
                lo = (a + w * (b - a) + 180) % 360 - 180
                nodes.append((m, local_solar_time(m, lo)))
                writer.writerow(["node", tag, f"{m:.3f}", "0.000", f"{lo:.3f}",
                                 f"{local_solar_time(m, lo):.4f}", ""])

            take = slice(None, None, max(step, 1))
            for m, la, lo, ls, ml in zip(met[take], lat[take], lon[take], lst[take], mlat[take]):
                samples.append((ls, ml))
                writer.writerow(["sample", tag, f"{m:.1f}", f"{la:.3f}", f"{lo:.3f}",
                                 f"{ls:.4f}", f"{ml:.2f}"])
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(days)} 天，升交点 {len(nodes)}", flush=True)

    total = dropped + kept
    print(f"\nPOSATT 行：好 {kept}，坏 {dropped}"
          f"（{dropped / total * 100:.4f}%，判据是地心距落在 "
          f"{(R_MIN - A) / 1e3:.0f}–{(R_MAX - A) / 1e3:.0f} km 高度带之外）")

    nodes = np.array(nodes)
    print(f"\n升交点 {len(nodes)} 个")
    if len(nodes) == 0:
        return
    order = np.argsort(nodes[:, 0])
    met, lst = nodes[order, 0], nodes[order, 1]
    days_since = (met - met[0]) / 86400.0

    print("\n=== 升交点地方时随任务时间 ===")
    print("任务日区间      n      LTAN 中位(h)   四分位")
    for a in range(0, int(days_since.max()) + 60, 60):
        sel = (days_since >= a) & (days_since < a + 60)
        if sel.sum() < 5:
            continue
        q = np.percentile(lst[sel], [25, 50, 75])
        print(f"  {a:4d}–{a+60:4d}   {int(sel.sum()):5d}   {q[1]:10.3f}   {q[0]:.3f}–{q[2]:.3f}")
    print(f"\n全任务 LTAN：中位 {np.median(lst):.3f} h，5–95% "
          f"{np.percentile(lst,5):.3f}–{np.percentile(lst,95):.3f}，极差 {np.ptp(lst):.3f} h")
    early = lst[days_since < 60]
    late = lst[days_since > days_since.max() - 60]
    if early.size and late.size:
        drift = np.median(late) - np.median(early)
        print(f"头 60 天中位 {np.median(early):.3f} h → 末 60 天中位 {np.median(late):.3f} h，"
              f"漂移 {drift:+.3f} h / {days_since.max():.0f} 天")
        print("  太阳同步：近似不变（|漂移| ≲ 0.5 h）；非同步：走遍 24 h。")

    samples = np.array(samples)
    print(f"\n=== 逐磁纬带的 LST 曝光（采样点 {len(samples)}，步长 {step} s）===")
    print("磁纬带        n        LST 覆盖的 2h 格数/12   最大格/最小格   变异系数")
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 55), (55, 90)]
    for a, b in bands:
        sel = (np.abs(samples[:, 1]) >= a) & (np.abs(samples[:, 1]) < b)
        if sel.sum() < 100:
            continue
        hist = np.histogram(samples[sel, 0], bins=12, range=(0, 24))[0]
        nonzero = int((hist > 0).sum())
        ratio = hist.max() / max(hist.min(), 1)
        cv = hist.std() / hist.mean()
        print(f"  |mlat| {a:2d}–{b:2d}  {int(sel.sum()):8d}   {nonzero:5d}/12          "
              f"{ratio:10.1f}   {cv:8.3f}")
    hist = np.histogram(samples[:, 0], bins=24, range=(0, 24))[0]
    print("\n全样本 LST 逐小时曝光占比 (%):")
    print("  " + " ".join(f"{v/hist.sum()*100:.1f}" for v in hist))


if __name__ == "__main__":
    main()
