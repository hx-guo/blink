"""GECAM 候选的第一遍体检：候选为什么这么多，CPD 和逐路计数能不能把它们分开。

搜索一天出几千个候选，远超 `false_positive_per_year = 20` 该有的量，说明多出来的
不是统计涨落而是结构性的东西。这个脚本不下判据，只把几个能分辨来源的量摊开：

* CPD 符合：`n_acd/n` 相对基线 `n_acd_bg/n_bg` 的涨幅。带电粒子穿过整台仪器时
  CPD 会跟着响，光子不会。
* 逐路 GRD：亮了几路、最亮一路占多少。粒子和毛刺集中，真暴发铺得开。
* 地理：SAA 和高磁纬是粒子的老家。
* 本底率与候选计数：看 `min_number` 这条下限有没有在起作用。

用法: python3 gecam_triage.py <signals.json ...>
"""

import json
import sys

import numpy as np

# 地磁偶极北极（2020 历元），只用来粗分高低磁纬
POLE_LAT, POLE_LON = np.radians(80.65), np.radians(-72.68)


def dipole_latitude(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sine = np.sin(lat) * np.sin(POLE_LAT) + np.cos(lat) * np.cos(POLE_LAT) * np.cos(lon - POLE_LON)
    return np.degrees(np.arcsin(np.clip(sine, -1, 1)))


def load(paths):
    rows = []
    for path in paths:
        for signal in json.load(open(path)):
            position = signal["position"]
            acd = signal.get("acd") or {}
            detectors = signal.get("detectors") or {}
            window = np.asarray(detectors.get("window") or [], float)
            baseline = np.asarray(detectors.get("baseline") or [], float)
            rows.append(
                dict(
                    start=signal["start"],
                    count=signal["count"],
                    mean=signal["mean"],
                    fa=signal["false_positive_per_year"],
                    lon=position["longitude"],
                    lat=position["latitude"],
                    n=acd.get("n"),
                    n_acd=acd.get("n_acd"),
                    n_acd_multi=acd.get("n_acd_multi"),
                    n_bg=acd.get("n_bg"),
                    n_acd_bg=acd.get("n_acd_bg"),
                    baseline_seconds=detectors.get("baseline_seconds"),
                    window_counts=window,
                    baseline_counts=baseline,
                )
            )
    return rows


def quantiles(name, values, unit=""):
    values = np.asarray([v for v in values if v is not None], float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"  {name}: 无数据")
        return
    q = np.percentile(values, [5, 25, 50, 75, 95])
    print(
        f"  {name}: 中位 {q[2]:.3g}{unit}  四分位 {q[1]:.3g}–{q[3]:.3g}  "
        f"5–95% {q[0]:.3g}–{q[4]:.3g}  n={values.size}"
    )


def main():
    rows = load(sys.argv[1:])
    if not rows:
        print("没有候选")
        return
    print(f"候选总数 {len(rows)}")

    print("\n== 显著性与计数 ==")
    quantiles("窗内计数", [r["count"] for r in rows])
    quantiles("窗内本底期望", [r["mean"] for r in rows])
    quantiles("log10 每年假阳性", [np.log10(r["fa"]) for r in rows if r["fa"] > 0])
    counts = np.asarray([r["count"] for r in rows], float)
    print(f"  计数 == 8（min_number 下限）的占比: {(counts == 8).mean() * 100:.1f}%")
    print(f"  计数 <= 10 的占比: {(counts <= 10).mean() * 100:.1f}%")

    print("\n== CPD（带电粒子探测器）==")
    have_cpd = [r for r in rows if r["n"] and r["n_bg"]]
    print(f"  有 CPD 统计的候选 {len(have_cpd)}/{len(rows)}")
    if have_cpd:
        # 窗内 CPD 每 GRD 计数 ÷ 基线同一比值。粒子事件这个比会显著 > 1。
        ratio = []
        for r in have_cpd:
            window_rate = r["n_acd"] / max(r["n"], 1)
            baseline_rate = r["n_acd_bg"] / max(r["n_bg"], 1)
            ratio.append(window_rate / baseline_rate if baseline_rate > 0 else np.inf)
        ratio = np.asarray(ratio, float)
        finite = ratio[np.isfinite(ratio)]
        quantiles("窗内 CPD/GRD ÷ 基线 CPD/GRD", finite)
        for threshold in (1.5, 2.0, 3.0, 5.0, 10.0):
            print(f"    比值 > {threshold}: {(ratio > threshold).mean() * 100:5.1f}%")
        quantiles("窗内 CPD 计数", [r["n_acd"] for r in have_cpd])
        quantiles("窗内 CPD 多路符合", [r["n_acd_multi"] for r in have_cpd])
        print(f"    窗内 CPD 计数为 0 的占比: {np.mean([r['n_acd'] == 0 for r in have_cpd]) * 100:.1f}%")

    print("\n== 逐路 GRD ==")
    have_det = [r for r in rows if r["window_counts"].size]
    print(f"  有逐路计数的候选 {len(have_det)}/{len(rows)}")
    if have_det:
        lit = [int((r["window_counts"] > 0).sum()) for r in have_det]
        share = [r["window_counts"].max() / max(r["window_counts"].sum(), 1) for r in have_det]
        quantiles("点亮的探头路数", lit)
        quantiles("最亮一路的计数占比", share)
        print(f"    只亮 1 路的占比: {np.mean(np.asarray(lit) == 1) * 100:.1f}%")
        print(f"    亮 >= 半数路的占比: {np.mean(np.asarray(lit) >= have_det[0]['window_counts'].size / 2) * 100:.1f}%")

    print("\n== 地理 ==")
    lon = np.asarray([r["lon"] for r in rows], float)
    lat = np.asarray([r["lat"] for r in rows], float)
    mlat = np.abs(dipole_latitude(lat, lon))
    quantiles("|偶极磁纬|", mlat, "°")
    # 经典 SAA 框
    saa = (lon > -90) & (lon < 40) & (lat > -40) & (lat < 0)
    print(f"  落在经典 SAA 框内的占比: {saa.mean() * 100:.1f}%")
    print(f"  |磁纬| > 40° 的占比: {(mlat > 40).mean() * 100:.1f}%")
    print(f"  |磁纬| < 20° 的占比: {(mlat < 20).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
