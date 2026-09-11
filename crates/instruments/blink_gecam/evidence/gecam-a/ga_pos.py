"""A 星候选位置的体检：POSATT 里的 X=Y=Z=0 坏行会不会已经渗进候选表。

坏行在 B 星上造出过上百个假升交点（R 从真值 0.018 错成 0.645）。候选表里的
position 是从 POSATT 插出来的，所以**先查候选自己的高度**：坏行会把星下点
推到地心，高度约 −6371 km 或高度异常小。顺带给出 LST 的前置量：这几天的
曝光覆盖了进动周期的百分之几。
"""
import glob
import json

import numpy as np

PRECESSION_DAYS = 48.5   # A/B 实测进动周期

alt_all, lat_all, lon_all = [], [], []
days = []
for p in sorted(glob.glob("/scratchfs2/gecam/guohx/gecam_a/data/GECAM-A/*/*/*_signals.json")):
    sig = json.load(open(p))
    if not sig:
        continue
    alt = np.array([s["position"]["altitude"] for s in sig])
    lat = np.array([s["position"]["latitude"] for s in sig])
    lon = np.array([s["position"]["longitude"] for s in sig])
    day = p.split("/")[-1][:8]
    days.append(day)
    alt_all.append(alt)
    lat_all.append(lat)
    lon_all.append(lon)
    bad = (alt < 1e5) | (alt > 2e6) | ~np.isfinite(alt)
    zero = (np.abs(lat) < 1e-9) & (np.abs(lon) < 1e-9)
    print(f"{day}  n={len(sig):7d}  高度 m: p0 {alt.min():10.0f} p50 {np.median(alt):10.0f} "
          f"p100 {alt.max():10.0f}  可疑高度 {int(bad.sum())}  lat=lon=0 的 {int(zero.sum())}")

alt = np.concatenate(alt_all)
lat = np.concatenate(lat_all)
print(f"\n合计 {alt.size} 个候选：高度全部在 "
      f"[{alt.min()/1e3:.1f}, {alt.max()/1e3:.1f}] km，"
      f"|lat| 中位 {np.median(np.abs(lat)):.1f}°、p99 {np.percentile(np.abs(lat),99):.1f}°")
print(f"（A 星轨道倾角 29°，|lat| 该在 29° 以内；超出即位置有问题）")

span = len(days)
print(f"\nLST 前置：这批是 {span} 个互不相邻的单日，"
      f"合计覆盖进动周期（{PRECESSION_DAYS} 天）的 {span/PRECESSION_DAYS*100:.1f}%。")
print("单日之间相隔数月，相位是抽样不是连续扫描——**不能当作把 LST 扫匀了**。")
