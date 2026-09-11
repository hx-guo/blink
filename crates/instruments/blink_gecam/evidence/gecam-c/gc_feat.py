"""GECAM-C 候选的事例级特征，口径与新二进制一致。

与 `scripts/cluster/gecam_features.py` 的差别只有一处，但它是致命的：**双增益去重按
死时间配对，不是按时戳相等**。C 星实测同一小时里死时间口径配到 1,735,222 对而时戳
相等只配到 1,324,865 对，**23.65% 的对时戳不等**（|dt| p90 = 104 ns、p99 = 209 ns）。
用时戳口径重算，候选窗内计数会系统性偏多，与搜索报的 `count` 对不上账。

去重用 `gc_dedupe.adjacent`（只认相邻的一对、全向量化），它与逐事例贪心的差别由
`gc_dedupe.py` 当场量出来报在日志里，不假设。

**顺序是先准入、后去重**，与 Rust 侧一致。

判别量按「与选样标准在构造上无关」挑：

* `f2/f3/f4` —— 落在 ≥2/≥3/≥4 重同戳簇里的计数占比，**去重之后算**，
  并逐候选给出偶然期望（`q` 用 `np.spacing(t)` 取，不写死）。
* `hard200` —— 200 keV 以上计数占比，用真实 EBOUNDS 折，与已发表目录同口径；
  同时给窗外 ±1 s 本底的同一个量。
* `cpd_*` —— **只当环境量记，不当真伪标签**。B 星拿 140 个已发表真 TGF 实测
  真 TGF 自己的 ±10 µs CPD obs/exp 就是 7.50，「CPD 紧符合 ⇒ 带电粒子」这个
  标签不成立，所以这里只记原始计数与**按活时间折的环境率**，解释留给后面。
* `t_bg` —— 本底窗与 GTI 的交集时长。分母必须是活时间，不是标称窗宽。

用法: gc_feat.py <signals.json glob> <输出 CSV>
"""

import csv
import datetime as dt
import glob
import json
import os
import sys
from math import comb

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gc_dedupe import adjacent  # noqa: E402

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = (2021, 1, 1)
MIN_CHANNEL, LADDER, NORMAL = 54, 448, 1
CPD_HALF_WIDTHS = (1e-5, 1e-4, 1e-3, 1e-2)
BASELINE_SECONDS = 1.0  # 候选两侧各这么宽
HOLLOW_SECONDS = 0.005  # 本底窗里挖掉候选两侧各这么宽


def met(iso):
    """ISO → MET。小数秒整取，不能截到微秒——窗最短只有 0.15 µs。"""
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def newest(pattern):
    files = []
    for path in glob.glob(pattern):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) >= 5 and parts[-1].startswith("v"):
            try:
                files.append((int(parts[-1][1:]), path))
            except ValueError:
                pass
    return max(files)[1] if files else None


def hour_path(iso, kind):
    directory = f"{ROOT}/{iso[:10].replace('-', '/')}/{'GRD_EVT' if kind == 'grd' else 'CPD_EVT'}"
    stem = f"{'gcg' if kind == 'grd' else 'gcc'}_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    return newest(f"{directory}/{stem}*.fits")


def load_hour(iso):
    """读一小时：GRD（准入 → 去重 → 按时间排）、CPD、GTI、EBOUNDS 的 200 keV 道。"""
    path = hour_path(iso, "grd")
    if path is None:
        return None
    times, channels, detectors = [], [], []
    with fits.open(path, memmap=True) as hdus:
        eb = hdus["EBOUNDS"].data
        e_max = np.asarray(eb["E_MAX"], float)
        ch200 = int(np.searchsorted(e_max[:LADDER], 200.0, "left"))
        gti = hdus["GTI"].data
        gti_start = np.asarray(gti["START"], float)
        gti_stop = np.asarray(gti["STOP"], float)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            pi = np.asarray(data["PI"]).astype(np.int16)
            evt = np.asarray(data["EVT_TYPE"]).astype(np.int8)
            keep = (evt == NORMAL) & (pi >= MIN_CHANNEL) & (pi < LADDER)
            if not keep.any():
                continue
            t = np.asarray(data["TIME"], float)[keep]
            order = np.argsort(t, kind="stable")
            t = t[order]
            pi = pi[keep][order]
            gain = np.asarray(data["GAIN_TYPE"])[keep][order].astype(np.int8)
            dead = np.asarray(data["DEAD_TIME"])[keep][order].astype(float) * 1e-6
            drop = adjacent(t, gain, dead)
            if drop.size:
                mask = np.ones(t.size, bool)
                mask[drop] = False
                t, pi = t[mask], pi[mask]
            times.append(t)
            channels.append(pi)
            detectors.append(np.full(t.size, int(hdu.name[-2:]), np.int8))
    if not times:
        return None
    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    grd = (t[order], np.concatenate(channels)[order], np.concatenate(detectors)[order])

    cpd_path = hour_path(iso, "cpd")
    cpd = None
    if cpd_path:
        ct = []
        with fits.open(cpd_path, memmap=True) as hdus:
            for hdu in hdus:
                if not hdu.name.startswith("EVENTS"):
                    continue
                data = hdu.data
                if data is None or len(data) == 0:
                    continue
                pi = np.asarray(data["PI"]).astype(np.int16)
                evt = np.asarray(data["EVT_TYPE"]).astype(np.int8)
                keep = evt == NORMAL
                ct.append(np.asarray(data["TIME"], float)[keep])
        if ct:
            cpd = np.sort(np.concatenate(ct))
        n_cpd_det = len(ct)
    else:
        n_cpd_det = 0
    return grd, cpd, (gti_start, gti_stop), ch200, n_cpd_det


def live_time(gti, low, high):
    """[low, high] 与 GTI 的交集时长。分母必须是活时间，不是标称窗宽。"""
    start, stop = gti
    return float(np.clip(np.minimum(stop, high) - np.maximum(start, low), 0, None).sum())


def cluster_stats(time, detector):
    """按时戳精确相等分簇，返回 (簇大小, 簇内探头数)。去重之后剩下的同戳只有跨探头的。"""
    if time.size == 0:
        return np.empty(0, int), np.empty(0, int)
    boundary = np.flatnonzero(np.diff(time) != 0) + 1
    groups = np.split(np.arange(time.size), boundary)
    return (
        np.array([g.size for g in groups]),
        np.array([np.unique(detector[g]).size for g in groups]),
    )


def main():
    signals = []
    for path in sorted(glob.glob(sys.argv[1])):
        signals.extend(json.load(open(path)))
    signals.sort(key=lambda s: s["start"])
    print(f"候选 {len(signals)} 个", flush=True)

    writer = csv.writer(open(sys.argv[2], "w", newline=""), lineterminator="\n")
    writer.writerow([
        "start", "fa", "count", "mean", "bin_us", "lon", "lat",
        "n_core", "match", "n_det_hit", "det_frac_max",
        "f2", "f3", "f4", "max_d", "n_trip", "q_ns", "exp_pair", "exp_trip",
        "min_dt_us", "hard200", "hard200_bkg", "n_bkg", "t_bg",
        "cpd_det", "cpd_rate_per_det",
    ] + [f"cpd_{int(w * 1e6)}us" for w in CPD_HALF_WIDTHS])

    cache_key, cache = None, None
    written = matched = 0
    for signal in signals:
        iso = signal["start"]
        key = iso[:13]
        if key != cache_key:
            cache_key, cache = key, load_hour(iso)
            print("  载入", key, flush=True)
        if cache is None:
            continue
        (time, channel, detector), cpd, gti, ch200, n_cpd_det = cache

        t0 = met(iso) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]
        lo = int(np.searchsorted(time, t0, "left"))
        hi = int(np.searchsorted(time, t1, "right"))
        n_core = hi - lo
        if n_core == 0:
            continue
        core = slice(lo, hi)

        sizes, dets = cluster_stats(time[core], detector[core])
        f2 = float(sizes[sizes >= 2].sum()) / n_core
        f3 = float(sizes[sizes >= 3].sum()) / n_core
        f4 = float(sizes[sizes >= 4].sum()) / n_core
        n_trip = int((sizes >= 3).sum())
        max_d = int(dets.max()) if dets.size else 0
        _, det_counts = np.unique(detector[core], return_counts=True)

        # 偶然期望：q 是该 MET 量级上 float64 的 ulp，跨过 2^n 秒粗一倍
        q = float(np.spacing(t0))
        width = max(t1 - t0, q)
        ratio = q / width
        exp_pair = comb(n_core, 2) * ratio if n_core >= 2 else 0.0
        exp_trip = comb(n_core, 3) * ratio ** 2 if n_core >= 3 else 0.0

        gaps = np.diff(time[core])
        positive = gaps[gaps > 0]
        min_dt_us = float(positive.min() * 1e6) if positive.size else 0.0

        far_lo = int(np.searchsorted(time, t0 - BASELINE_SECONDS, "left"))
        far_hi = int(np.searchsorted(time, t1 + BASELINE_SECONDS, "right"))
        hole_lo = int(np.searchsorted(time, t0 - HOLLOW_SECONDS, "left"))
        hole_hi = int(np.searchsorted(time, t1 + HOLLOW_SECONDS, "right"))
        bkg = np.concatenate((channel[far_lo:hole_lo], channel[hole_hi:far_hi]))
        t_bg = live_time(gti, t0 - BASELINE_SECONDS, t1 + BASELINE_SECONDS) - live_time(
            gti, t0 - HOLLOW_SECONDS, t1 + HOLLOW_SECONDS
        )

        hard = float((channel[core] >= ch200).mean())
        hard_bkg = float((bkg >= ch200).mean()) if bkg.size else float("nan")

        cpd_counts = []
        cpd_rate = ""
        if cpd is not None:
            for half in CPD_HALF_WIDTHS:
                a = int(np.searchsorted(cpd, t0 - half, "left"))
                b = int(np.searchsorted(cpd, t1 + half, "right"))
                cpd_counts.append(b - a)
            a = int(np.searchsorted(cpd, t0 - BASELINE_SECONDS, "left"))
            b = int(np.searchsorted(cpd, t1 + BASELINE_SECONDS, "right"))
            if t_bg > 0 and n_cpd_det:
                cpd_rate = f"{(b - a) / t_bg / n_cpd_det:.2f}"
        else:
            cpd_counts = [""] * len(CPD_HALF_WIDTHS)

        ok = int(n_core == signal["count"])
        matched += ok
        writer.writerow([
            iso[:23], f"{signal['false_positive_per_year']:.3e}", signal["count"],
            f"{signal['mean']:.5f}", f"{signal['bin_size_best'] * 1e6:.3f}",
            f"{signal['position']['longitude']:.3f}", f"{signal['position']['latitude']:.3f}",
            n_core, ok, len(det_counts), f"{det_counts.max() / n_core:.3f}",
            f"{f2:.4f}", f"{f3:.4f}", f"{f4:.4f}", max_d, n_trip,
            f"{q * 1e9:.3f}", f"{exp_pair:.3e}", f"{exp_trip:.3e}",
            f"{min_dt_us:.4f}", f"{hard:.4f}",
            f"{hard_bkg:.4f}" if np.isfinite(hard_bkg) else "",
            bkg.size, f"{t_bg:.4f}", n_cpd_det, cpd_rate,
        ] + cpd_counts)
        written += 1

    print(f"写出 {written}；与搜索 count 逐条相等 {matched}"
          f"（{matched / max(written, 1) * 100:.2f}%）")
    if matched != written:
        print("**对账没到 100%，窗口口径有问题，后面的数一个都别用。**")
        sys.exit(1)


if __name__ == "__main__":
    main()
