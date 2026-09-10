"""GECAM 候选的事例级特征：同戳簇集、探头铺开程度、能谱、以及多个宽度下的 CPD 符合。

搜索一天出几千个候选，而配置的假阳性率是每年 20 个，差了四个数量级——多出来的
不是统计涨落，是泊松独立性被破坏了。天格上同一个现象的根因是带电粒子穿过整星：
各路探测器在同一个时间戳上各留一个计数，八个"独立"计数其实是一次事件。这里量
的就是这件事，量完再决定要不要立判据。

CPD 用多个窗宽量，是因为候选窗常常只有十几微秒：GECAM-C 两路 CPD 合计约 300 c/s，
10 µs 里期望 0.003 个，看见 0 个什么也说明不了。窗要宽到期望计数有意义为止。

用法: python3 gecam_features.py <signals.json> <输出 CSV> [卫星=GECAM-C]
"""

import csv
import glob
import json
import os
import sys
import datetime as dt

import numpy as np
from astropy.io import fits

ARCHIVE = {
    "GECAM-A": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_A", "GRD_evt", "CPD_evt", "gag", "gac"),
    "GECAM-B": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_B", "GRD_evt", "CPD_evt", "gbg", "gbc"),
    "GECAM-C": ("/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily", None, "GRD_EVT", "CPD_EVT", "gcg", "gcc"),
}
EPOCH = {"GECAM-A": (2019, 1, 1), "GECAM-B": (2019, 1, 1), "GECAM-C": (2021, 1, 1)}
# 事例准入，与 Rust 侧 `Event::keep` 一致
MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE = 54, 448, 1
# CPD 符合的几个窗宽（秒），从候选窗两端各外扩这么多
CPD_HALF_WIDTHS = (1e-5, 1e-4, 1e-3, 1e-2)


def met(iso, satellite):
    body = iso.rstrip("Z")
    head, frac = body.split(".")
    stamp = dt.datetime.strptime(head + "." + (frac + "000000")[:6], "%Y-%m-%dT%H:%M:%S.%f")
    ref = dt.datetime(*EPOCH[satellite], tzinfo=dt.timezone.utc)
    return (stamp.replace(tzinfo=dt.timezone.utc) - ref).total_seconds()


def hour_file(satellite, iso, kind):
    root, sat_dir, grd, cpd, grd_prefix, cpd_prefix = ARCHIVE[satellite]
    day = iso[:10].replace("-", "/")
    subdir, prefix = (grd, grd_prefix) if kind == "grd" else (cpd, cpd_prefix)
    directory = os.path.join(root, day, sat_dir, subdir) if sat_dir else os.path.join(root, day, subdir)
    stem = f"{prefix}_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    files = sorted(f for f in glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


def read_events(path, want_pi):
    """把一个小时里各路事例合成按时间排好的 (时刻, 能道, 探头号)。"""
    times, channels, detectors = [], [], []
    with fits.open(path) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            keep = np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE
            pi = np.asarray(data["PI"]) if want_pi else None
            if want_pi:
                keep &= (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            times.append(np.asarray(data["TIME"], float)[keep])
            channels.append(pi[keep] if want_pi else np.zeros(int(keep.sum()), int))
            detectors.append(np.full(int(keep.sum()), int(hdu.name[-2:])))
    if not times:
        return None
    time = np.concatenate(times)
    order = np.argsort(time)
    return time[order], np.concatenate(channels)[order], np.concatenate(detectors)[order]


def main():
    signals_glob, out_path = sys.argv[1], sys.argv[2]
    satellite = sys.argv[3] if len(sys.argv) > 3 else "GECAM-C"

    signals = []
    for path in sorted(glob.glob(signals_glob)):
        signals.extend(json.load(open(path)))
    print(f"候选 {len(signals)} 个")

    writer = csv.writer(open(out_path, "w", newline=""))
    writer.writerow(
        ["start", "fa", "count", "mean", "bin_us", "lon", "lat", "n_core", "n_det_hit",
         "det_frac_max", "max_mult", "simul_frac", "min_dt_us", "pi_core_med", "pi_bkg_med",
         "pi_ratio", "rate_win"]
        + [f"cpd_{int(w * 1e6)}us" for w in CPD_HALF_WIDTHS]
    )

    grd_cache, cpd_cache = {}, {}
    written = 0
    for signal in signals:
        iso = signal["start"]
        t0 = met(iso, satellite) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]
        key = iso[:13]

        if key not in grd_cache:
            path = hour_file(satellite, iso, "grd")
            grd_cache[key] = read_events(path, want_pi=True) if path else None
            path = hour_file(satellite, iso, "cpd")
            cpd_cache[key] = read_events(path, want_pi=False) if path else None
        grd, cpd = grd_cache[key], cpd_cache[key]
        if grd is None:
            continue
        time, channel, detector = grd

        core = (time >= t0) & (time <= t1)
        n_core = int(core.sum())
        if n_core == 0:
            continue
        # 同一时间戳上最长的一串：带电粒子穿过整台仪器时各路同时响
        _, multiplicity = np.unique(time[core], return_counts=True)
        max_mult = int(multiplicity.max())
        _, det_counts = np.unique(detector[core], return_counts=True)
        gaps = np.diff(time[core])
        positive = gaps[gaps > 0]
        min_dt_us = positive.min() * 1e6 if positive.size else 0.0

        near = ((time >= t0 - 1.0) & (time <= t1 + 1.0)) & ~((time >= t0 - 0.01) & (time <= t1 + 0.01))
        pi_bkg = float(np.median(channel[near])) if near.any() else np.nan
        pi_core = float(np.median(channel[core]))
        window = (time >= t0 - 0.5) & (time <= t1 + 0.5)

        cpd_counts = []
        for half in CPD_HALF_WIDTHS:
            if cpd is None:
                cpd_counts.append("")
                continue
            cpd_time = cpd[0]
            cpd_counts.append(int(((cpd_time >= t0 - half) & (cpd_time <= t1 + half)).sum()))

        writer.writerow(
            [iso[:23], f"{signal['false_positive_per_year']:.3e}", signal["count"],
             f"{signal['mean']:.5f}", f"{signal['bin_size_best'] * 1e6:.2f}",
             f"{signal['position']['longitude']:.3f}", f"{signal['position']['latitude']:.3f}",
             n_core, len(det_counts), f"{det_counts.max() / n_core:.3f}", max_mult,
             f"{max_mult / n_core:.3f}", f"{min_dt_us:.3f}", f"{pi_core:.0f}",
             f"{pi_bkg:.0f}" if np.isfinite(pi_bkg) else "",
             f"{pi_core / pi_bkg:.3f}" if np.isfinite(pi_bkg) and pi_bkg > 0 else "",
             f"{window.sum() / (1.0 + signal['bin_size_best']):.0f}"]
            + cpd_counts
        )
        written += 1
        # 一小时的事例几千万，缓存只留当前小时
        if len(grd_cache) > 1:
            for stale in [k for k in grd_cache if k != key]:
                grd_cache.pop(stale)
                cpd_cache.pop(stale, None)

    print("写出", written)


if __name__ == "__main__":
    main()
