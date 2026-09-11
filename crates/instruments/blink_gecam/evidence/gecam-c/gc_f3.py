"""GECAM-C：f₃ —— 落在 ≥3 重同戳簇里的计数占比，在双增益合并之后算。

二重同戳的偶然期望是 ~C(n,2)·q/T，三重是 ~C(n,3)·(q/T)²——平方压下去，所以三重
基本不可能偶然发生，f₃ 天然避开"偶然符合淹没真信号"这个坑。

**必须在双增益合并之后算**：未合并时同一时戳上凭空多一条同探头重复记录，
一个真的跨探头二重加一条重复就冒充成三重，f₃ 会系统性虚高。合并后剩下的同戳
只有跨探头的，那时的 ≥3 重才真是"一个粒子同时点亮三路"。

`q`（时戳格）不写死：GECAM 的 TIME 是 float64 秒，格子就是该 MET 量级上的 ulp，
跨过 2ⁿ 秒会粗一倍（C 星 2023-02-19 前 7.45 ns、后 14.9 ns）。用 `np.spacing(t)`
逐候选取，永远对。

用法: python3 gc_f3.py <signals.json> <输出 CSV>
"""

import csv
import datetime as dt
import glob
import json
import sys
from math import comb

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = (2021, 1, 1)
MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE = 54, 448, 1
TOLERANCE = 1e-7  # 100 ns，按 GECAM-C 探头间相对定时精度 0.1 µs（arXiv 2308.11362）


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def hour_file(iso):
    directory = f"{ROOT}/{iso[:10].replace('-', '/')}/GRD_EVT"
    stem = f"gcg_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    files = sorted(glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


def read_events(path):
    times, detectors, gains = [], [], []
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            pi = np.asarray(data["PI"])
            keep = (np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            times.append(np.asarray(data["TIME"], float)[keep])
            detectors.append(np.full(int(keep.sum()), int(hdu.name[-2:])))
            gains.append(np.asarray(data["GAIN_TYPE"])[keep].astype(int))
    if not times:
        return None
    time = np.concatenate(times)
    order = np.argsort(time, kind="stable")
    return time[order], np.concatenate(detectors)[order], np.concatenate(gains)[order]


def merge_gain_pairs(time, detector, gain):
    """同探头同时戳的多条并成一条，保留低增益（GAIN_TYPE 高）那条，与 Rust 侧同规则。"""
    if time.size == 0:
        return time, detector
    key = np.stack([time, detector.astype(float)], 1)
    # 按 (时戳, 探头, −增益) 排序后取每组第一条
    order = np.lexsort((-gain, detector, time))
    t, d, g = time[order], detector[order], gain[order]
    first = np.ones(t.size, bool)
    first[1:] = (t[1:] != t[:-1]) | (d[1:] != d[:-1])
    t, d = t[first], d[first]
    o2 = np.argsort(t, kind="stable")
    return t[o2], d[o2]


def multiplet_stats(time, detector, tol=0.0):
    """返回 (簇大小数组 d, 每簇的探头数组)。tol=0 时按时戳精确相等分簇。"""
    if time.size == 0:
        return np.array([]), np.array([])
    if tol <= 0:
        boundary = np.flatnonzero(np.diff(time) != 0) + 1
    else:
        boundary = np.flatnonzero(np.diff(time) > tol) + 1
    groups = np.split(np.arange(time.size), boundary)
    sizes = np.array([g.size for g in groups])
    dets = np.array([np.unique(detector[g]).size for g in groups])
    return sizes, dets


def main():
    signals = json.load(open(sys.argv[1]))
    writer = csv.writer(open(sys.argv[2], "w", newline=""))
    writer.writerow(["start", "fa", "count", "n_core", "n_merged", "same_det_extra",
                     "bin_us", "q_ns", "f2", "f3", "f4", "n_trip", "max_d",
                     "exp_pair", "exp_trip", "f3_tol", "n_trip_tol", "max_d_tol"])
    cache = {}
    written = mismatch = 0
    for signal in signals:
        iso = signal["start"]
        key = iso[:13]
        if key not in cache:
            cache.clear()
            path = hour_file(iso)
            cache[key] = read_events(path) if path else None
            print("  载入", key, flush=True)
        got = cache[key]
        if got is None:
            continue
        time, detector, gain = got
        t0 = met(iso) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]
        lo = np.searchsorted(time, t0, "left")
        hi = np.searchsorted(time, t1, "right")
        n_core = hi - lo
        if n_core == 0:
            continue
        if n_core != signal["count"]:
            mismatch += 1
        wt, wd, wg = time[lo:hi], detector[lo:hi], gain[lo:hi]

        mt, md = merge_gain_pairs(wt, wd, wg)
        n_merged = mt.size
        sizes, dets = multiplet_stats(mt, md)
        # 合并之后同一时戳最多一条/探头，所以 sizes == dets
        f2 = dets[dets >= 2].sum() / n_merged
        f3 = dets[dets >= 3].sum() / n_merged
        f4 = dets[dets >= 4].sum() / n_merged
        n_trip = int((dets >= 3).sum())
        max_d = int(dets.max())

        q = float(np.spacing(t0))
        width = signal["bin_size_best"]
        # 偶然期望：n 个计数落进 T/q 个格子里，撞出二重/三重的期望簇数
        slots = width / q if width > 0 else 1.0
        exp_pair = comb(n_merged, 2) / slots if slots > 0 else np.inf
        exp_trip = comb(n_merged, 3) / slots ** 2 if slots > 0 else np.inf

        st, sd = multiplet_stats(mt, md, TOLERANCE)
        f3_tol = sd[sd >= 3].sum() / n_merged
        writer.writerow([iso[:23], "%.4e" % signal["false_positive_per_year"], signal["count"],
                         n_core, n_merged, n_core - n_merged, "%.3f" % (width * 1e6),
                         "%.4f" % (q * 1e9), "%.4f" % f2, "%.4f" % f3, "%.4f" % f4,
                         n_trip, max_d, "%.4g" % exp_pair, "%.4g" % exp_trip,
                         "%.4f" % f3_tol, int((sd >= 3).sum()), int(sd.max())])
        written += 1
    print("写出 %d；n_core 与 count 对不上 %d" % (written, mismatch))


if __name__ == "__main__":
    main()
