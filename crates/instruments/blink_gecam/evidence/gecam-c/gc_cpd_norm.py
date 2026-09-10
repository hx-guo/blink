"""GECAM-C：CPD 符合的窗长归一化 + 按窗长分箱，检验同戳判据（multiplet_frac）是否真在挑光子。

裸的"cpd_10us > 0 占比"不能比，因为幸存者的候选窗系统性更长（中位 169.8 µs vs 85.2 µs），
窗越长越容易撞上 CPD 计数。这里改报**观测/期望**：期望用该候选**当地**的 CPD 实际率
（从 CPD 事例流现数，不用 300 c/s 这个估计值）乘以符合窗的**GTI 有效时长**。

曝光一律走 GTI 交集，事例也一律先按 GTI 过滤——两边口径必须一致，否则落在 GTI 边缘的
候选会把率算低、把超出算高。

窗口时刻不从 feat3.csv 取（那里的 start 只截到毫秒、delay 也没存），直接回 signals.json 现算，
met() 用不截断的写法。

用法: python3 cpd_norm.py <signals.json> <feat3.csv> <输出 CSV>
"""

import csv
import datetime as dt
import glob
import json
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = (2021, 1, 1)
HALF_WIDTHS = (1e-5, 1e-4, 1e-3, 1e-2)
BASELINE = 1.0   # 本底窗半宽（秒）
GUARD = 0.01     # 本底窗里扣掉候选窗两侧这么多秒


def met(iso):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def cpd_file(iso):
    directory = f"{ROOT}/{iso[:10].replace('-', '/')}/CPD_EVT"
    stem = f"gcc_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    files = sorted(glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


def read_cpd(path):
    """返回 (全部事例时刻, GTI 内事例时刻, GTI 起止)。两份都留：前者对账 feat3，后者做物理。"""
    times, gti = [], None
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if hdu.name == "GTI":
                gti = (np.asarray(hdu.data["START"], float), np.asarray(hdu.data["STOP"], float))
            if hdu.name.startswith("EVENTS"):
                data = hdu.data
                keep = np.asarray(data["EVT_TYPE"]) == 1
                times.append(np.asarray(data["TIME"], float)[keep])
    if not times:
        return np.array([]), np.array([]), gti
    all_t = np.sort(np.concatenate(times))
    if gti is None:
        return all_t, all_t, gti
    inside = np.zeros(all_t.size, bool)
    for start, stop in zip(*gti):
        inside |= (all_t >= start) & (all_t <= stop)
    return all_t, all_t[inside], gti


def gti_overlap(a, b, gti):
    if gti is None:
        return max(b - a, 0.0)
    starts, stops = gti
    return float(np.clip(np.minimum(b, stops) - np.maximum(a, starts), 0, None).sum())


def count(sorted_times, a, b):
    return int(np.searchsorted(sorted_times, b, "right") - np.searchsorted(sorted_times, a, "left"))


def main():
    signals_path, feat_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    signals = json.load(open(signals_path))
    feat = list(csv.DictReader(open(feat_path)))
    print(f"候选 {len(signals)} 个, feat 行 {len(feat)} 行")
    if len(signals) != len(feat):
        print("！行数不等，按 start+count 对齐会失真，先停")
        return

    writer = csv.writer(open(out_path, "w", newline=""))
    writer.writerow(
        ["start", "bin_us", "multiplet_frac", "n_det_hit", "count", "pi_ratio", "lon", "lat",
         "cpd_rate", "n_bg", "t_bg"]
        + [f"obs_{int(w * 1e6)}us" for w in HALF_WIDTHS]
        + [f"exp_{int(w * 1e6)}us" for w in HALF_WIDTHS]
        + [f"raw_{int(w * 1e6)}us" for w in HALF_WIDTHS]
    )

    cache = {}
    mismatch_start = mismatch_cpd = 0
    written = 0
    for signal, row in zip(signals, feat):
        iso = signal["start"]
        if row["start"] != iso[:23]:
            mismatch_start += 1
            continue
        key = iso[:13]
        if key not in cache:
            cache.clear()
            path = cpd_file(iso)
            cache[key] = read_cpd(path) if path else (np.array([]), np.array([]), None)
        all_t, in_t, gti = cache[key]
        if all_t.size == 0:
            continue

        t0 = met(iso) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]

        # 本底：候选窗两侧各 BASELINE 秒，扣掉紧贴窗口的 GUARD
        left = (t0 - BASELINE, t0 - GUARD)
        right = (t1 + GUARD, t1 + BASELINE)
        n_bg = count(in_t, *left) + count(in_t, *right)
        t_bg = gti_overlap(*left, gti) + gti_overlap(*right, gti)
        rate = n_bg / t_bg if t_bg > 0 else np.nan

        obs, exp, raw = [], [], []
        for half in HALF_WIDTHS:
            a, b = t0 - half, t1 + half
            obs.append(count(in_t, a, b))
            exp.append(rate * gti_overlap(a, b, gti) if np.isfinite(rate) else np.nan)
            raw.append(count(all_t, a, b))
        # 对账：不带 GTI 过滤的重算必须与 feat3 的 cpd_* 逐条相等
        for value, name in zip(raw, ("cpd_10us", "cpd_100us", "cpd_1000us", "cpd_10000us")):
            if row[name] != "" and int(row[name]) != value:
                mismatch_cpd += 1
                break

        writer.writerow(
            [iso[:23], f"{signal['bin_size_best'] * 1e6:.3f}", row["multiplet_frac"],
             row["n_det_hit"], signal["count"], row["pi_ratio"],
             f"{signal['position']['longitude']:.3f}", f"{signal['position']['latitude']:.3f}",
             f"{rate:.2f}" if np.isfinite(rate) else "", n_bg, f"{t_bg:.4f}"]
            + obs
            + [f"{e:.6g}" if np.isfinite(e) else "" for e in exp]
            + raw
        )
        written += 1

    print(f"写出 {written}；start 对不上 {mismatch_start}；CPD 计数与 feat3 对不上 {mismatch_cpd}")


if __name__ == "__main__":
    main()
