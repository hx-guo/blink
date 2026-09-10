"""GECAM-C：CPD 符合统计的零假设对照。

"候选窗内 CPD 超出 = 带电粒子"这条推理有个前提：CPD 与 GRD 的时戳是同一时基、
没有共同的读出假象。如果随便挑一个时刻做同样的统计本身就超出，上面的推理全作废。

两个对照：
  * 平移对照：把每个候选窗整体平移 ±0.2 / ±0.5 秒，窗长不变，同样算 obs/exp。
    平移保留了候选的本地环境（同一段轨道、同样的本底率），只挪开了触发时刻。
  * 随机对照：在当天 GTI 内均匀抽时刻，窗长从候选的 bin_us 分布里抽，算 obs/exp。

两者都该回到 1.00。

用法: python3 cpd_null.py <signals.json> <输出 CSV>
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
BASELINE, GUARD = 1.0, 0.01
SHIFTS = (-0.5, -0.2, 0.2, 0.5)


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
        return np.array([]), gti
    all_t = np.sort(np.concatenate(times))
    if gti is None:
        return all_t, gti
    inside = np.zeros(all_t.size, bool)
    for start, stop in zip(*gti):
        inside |= (all_t >= start) & (all_t <= stop)
    return all_t[inside], gti


def gti_overlap(a, b, gti):
    if gti is None:
        return max(b - a, 0.0)
    starts, stops = gti
    return float(np.clip(np.minimum(b, stops) - np.maximum(a, starts), 0, None).sum())


def count(times, a, b):
    return int(np.searchsorted(times, b, "right") - np.searchsorted(times, a, "left"))


def measure(times, gti, t0, t1):
    left = (t0 - BASELINE, t0 - GUARD)
    right = (t1 + GUARD, t1 + BASELINE)
    n_bg = count(times, *left) + count(times, *right)
    t_bg = gti_overlap(*left, gti) + gti_overlap(*right, gti)
    if t_bg <= 0 or n_bg == 0:
        return None
    rate = n_bg / t_bg
    out = []
    for half in HALF_WIDTHS:
        a, b = t0 - half, t1 + half
        out.append((count(times, a, b), rate * gti_overlap(a, b, gti)))
    return out


def main():
    signals_path, out_path = sys.argv[1], sys.argv[2]
    signals = json.load(open(signals_path))
    rng = np.random.default_rng(20230615)

    writer = csv.writer(open(out_path, "w", newline=""))
    writer.writerow(["kind", "shift", "start", "bin_us"]
                    + [f"obs_{int(w*1e6)}us" for w in HALF_WIDTHS]
                    + [f"exp_{int(w*1e6)}us" for w in HALF_WIDTHS])

    by_hour = {}
    for signal in signals:
        by_hour.setdefault(signal["start"][:13], []).append(signal)

    totals = {}
    n_rand_total = 0
    for key in sorted(by_hour):
        group = by_hour[key]
        path = cpd_file(group[0]["start"])
        if path is None:
            continue
        times, gti = read_cpd(path)
        if times.size == 0:
            continue
        widths = np.array([s["bin_size_best"] for s in group])

        for signal in group:
            t0 = met(signal["start"]) + signal["delay"]
            width = signal["bin_size_best"]
            for shift in SHIFTS:
                got = measure(times, gti, t0 + shift, t0 + shift + width)
                if got is None:
                    continue
                writer.writerow(["shift", shift, signal["start"][:23], f"{width*1e6:.3f}"]
                                + [g[0] for g in got] + [f"{g[1]:.6g}" for g in got])
                for i, (o, e) in enumerate(got):
                    a, b = totals.setdefault(("shift", i), [0, 0.0])
                    totals[("shift", i)] = [a + o, b + e]

        # 随机对照：本小时 GTI 内抽与候选同样多的时刻
        if gti is not None:
            starts, stops = gti
            spans = stops - starts
            weight = spans / spans.sum()
            n_draw = len(group) * 4
            seg = rng.choice(len(spans), size=n_draw, p=weight)
            t0s = starts[seg] + rng.random(n_draw) * spans[seg]
            ws = rng.choice(widths, size=n_draw)
            for t0, width in zip(t0s, ws):
                got = measure(times, gti, t0, t0 + width)
                if got is None:
                    continue
                writer.writerow(["random", "", "", f"{width*1e6:.3f}"]
                                + [g[0] for g in got] + [f"{g[1]:.6g}" for g in got])
                n_rand_total += 1
                for i, (o, e) in enumerate(got):
                    a, b = totals.setdefault(("random", i), [0, 0.0])
                    totals[("random", i)] = [a + o, b + e]

    print("=== 零假设对照 obs/exp（该回到 1.00）===")
    for kind in ("shift", "random"):
        for i, half in enumerate(HALF_WIDTHS):
            if (kind, i) not in totals:
                continue
            o, e = totals[(kind, i)]
            ratio = o / e if e > 0 else np.nan
            sigma = np.sqrt(max(o, 1)) / e if e > 0 else np.nan
            print("  %-7s ±%-6d µs: obs=%7d exp=%9.1f  比值 %.4f ± %.4f  (偏离 1 的 z = %+.2f)"
                  % (kind, int(half * 1e6), o, e, ratio, sigma, (ratio - 1) / sigma))
    print("随机样本", n_rand_total)


if __name__ == "__main__":
    main()
