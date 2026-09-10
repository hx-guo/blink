"""GECAM-C：把同戳量按"跨探头 / 同探头"拆开，并附 fa。

现在的 multiplet_frac 把两类东西混在一起：
  * 同一时戳落在**不同探头** —— 一个带电粒子穿过整台仪器的物理签名；
  * 同一时戳落在**同一路探头** —— 时戳量化（约 0.03 µs）下的偶然撞车或堆积，不是粒子证据。

对每个时戳簇，设簇内事例数 m、涉及的不同探头数 d：
  * 跨探头参与事例 = d（当 d >= 2），否则 0；
  * 同探头多余事例 = m - d。
xdet_multiplet_frac = Σ_{d>=2} d / n_core，same_det_extra = Σ (m - d)。

时戳按精确相等分簇之外，再做一版 100 ns 容差（同一粒子的两路读出可能差一个量化步）。

用法: python3 xdet_feat.py <signals.json> <输出 CSV> [卫星=GECAM-C]
"""

import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ARCHIVE = {
    "GECAM-A": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_A", "GRD_evt", "gag"),
    "GECAM-B": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_B", "GRD_evt", "gbg"),
    "GECAM-C": ("/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily", None, "GRD_EVT", "gcg"),
}
EPOCH = {"GECAM-A": (2019, 1, 1), "GECAM-B": (2019, 1, 1), "GECAM-C": (2021, 1, 1)}
MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE = 54, 448, 1
TOLERANCE = 1e-7  # 100 ns


def met(iso, satellite):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH[satellite], tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def hour_file(satellite, iso):
    root, sat_dir, subdir, prefix = ARCHIVE[satellite]
    day = iso[:10].replace("-", "/")
    directory = os.path.join(root, day, sat_dir, subdir) if sat_dir else os.path.join(root, day, subdir)
    stem = f"{prefix}_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    files = sorted(glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


def read_events(path):
    times, channels, detectors = [], [], []
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            pi = np.asarray(data["PI"])
            keep = (np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            times.append(np.asarray(data["TIME"], float)[keep])
            channels.append(pi[keep])
            detectors.append(np.full(int(keep.sum()), int(hdu.name[-2:])))
    if not times:
        return None
    time = np.concatenate(times)
    order = np.argsort(time, kind="stable")
    return time[order], np.concatenate(channels)[order], np.concatenate(detectors)[order]


def decompose(time, detector):
    """按精确时戳分簇，返回 (m 列表, d 列表)。time 已排序。"""
    if time.size == 0:
        return np.array([]), np.array([])
    boundary = np.flatnonzero(np.diff(time) != 0) + 1
    groups = np.split(np.arange(time.size), boundary)
    m = np.array([g.size for g in groups])
    d = np.array([np.unique(detector[g]).size for g in groups])
    return m, d


def decompose_tol(time, detector, tol):
    """把间隔 <= tol 的相邻事例并成一簇（单链接）。"""
    if time.size == 0:
        return np.array([]), np.array([])
    boundary = np.flatnonzero(np.diff(time) > tol) + 1
    groups = np.split(np.arange(time.size), boundary)
    m = np.array([g.size for g in groups])
    d = np.array([np.unique(detector[g]).size for g in groups])
    return m, d


def main():
    signals_path, out_path = sys.argv[1], sys.argv[2]
    satellite = sys.argv[3] if len(sys.argv) > 3 else "GECAM-C"
    signals = []
    for path in sorted(glob.glob(signals_path)):
        signals.extend(json.load(open(path)))
    print(f"候选 {len(signals)} 个", flush=True)

    writer = csv.writer(open(out_path, "w", newline=""))
    writer.writerow(["start", "fa", "count", "n_core", "bin_us", "delay_us",
                     "multiplet_frac", "xdet_frac", "same_det_extra", "n_groups", "max_d",
                     "xdet_frac_tol", "same_det_extra_tol", "max_d_tol",
                     "n_det_hit", "det_frac_max", "lon", "lat"])

    cache = {}
    written = mismatch = 0
    tick_report = 0
    for signal in signals:
        iso = signal["start"]
        key = iso[:13]
        if key not in cache:
            cache.clear()
            path = hour_file(satellite, iso)
            cache[key] = read_events(path) if path else None
            print(f"  载入 {key} -> {'ok' if cache[key] is not None else '缺文件'}", flush=True)
            if cache[key] is not None and tick_report < 1:
                gaps = np.diff(cache[key][0][:2000000])
                positive = gaps[gaps > 0]
                if positive.size:
                    print("  时戳量化步长实测最小正间隔 %.6g s，第 1 分位 %.6g s"
                          % (positive.min(), np.percentile(positive, 1)), flush=True)
                tick_report = 1
        grd = cache[key]
        if grd is None:
            continue
        time, _, detector = grd
        t0 = met(iso, satellite) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]
        lo = np.searchsorted(time, t0, "left")
        hi = np.searchsorted(time, t1, "right")
        n_core = hi - lo
        if n_core == 0:
            continue
        if n_core != signal["count"]:
            mismatch += 1
        wt, wd = time[lo:hi], detector[lo:hi]

        m, d = decompose(wt, wd)
        multiplet_frac = m[m >= 2].sum() / n_core
        xdet = d[d >= 2].sum() / n_core
        same_extra = int((m - d).sum())
        mt, dt_ = decompose_tol(wt, wd, TOLERANCE)
        xdet_tol = dt_[dt_ >= 2].sum() / n_core
        same_extra_tol = int((mt - dt_).sum())

        det_ids, det_counts = np.unique(wd, return_counts=True)
        writer.writerow(
            [iso[:23], f"{signal['false_positive_per_year']:.4e}", signal["count"], n_core,
             f"{signal['bin_size_best'] * 1e6:.3f}", f"{signal['delay'] * 1e6:.3f}",
             f"{multiplet_frac:.4f}", f"{xdet:.4f}", same_extra, len(m), int(d.max()),
             f"{xdet_tol:.4f}", same_extra_tol, int(dt_.max()),
             len(det_ids), f"{det_counts.max() / n_core:.4f}",
             f"{signal['position']['longitude']:.3f}", f"{signal['position']['latitude']:.3f}"])
        written += 1

    print(f"写出 {written}；n_core 与 count 对不上 {mismatch} 条")


if __name__ == "__main__":
    main()
