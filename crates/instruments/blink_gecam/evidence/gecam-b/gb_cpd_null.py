"""GECAM-B：CPD 符合统计的零假设自检。**这一步不过，后面所有 CPD 结论都不算数。**

"候选窗内 CPD 超出 = 带电粒子"有个前提：CPD 与 GRD 同一时基、没有共同读出假象，
而且期望值的算法（当地 CPD 率 × 符合窗与 GTI 的交集时长）无偏。随便挑一个时刻做
同样的统计要是本身就超出，上面的推理全作废。

两组对照，同一套机器：
  * 平移：候选窗整体平移 ±0.2 / ±0.5 s，窗长不变——保留本地环境、只挪开触发时刻；
  * 随机：GTI 内均匀抽时刻，窗长从候选的 bin 分布里抽。
两者的 obs/exp 都该回到 1.00。

**B 星必须自己量一遍，不能照抄 C 星的"基线窗被 GTI 切到只占 0.38%、可忽略"。**
C 星占空比 62.7%、缺口长而稀；B 星的 GTI 切得更碎（一小时常被切成一到四段），
同一个偏差在 B 星上未必可忽略——这里顺带把"本底窗被 GTI 切到"的比例也报出来。

用法: python3 gb_cpd_null.py <小时清单> <输出 CSV> <worker> <workers>
"""

import csv
import json
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_feat import (CPD_HALF_WIDTHS, BASELINE, GUARD, NORMAL_EVT_TYPE, SIGNAL_ROOT,
                     gti_overlap, hour_file, inside_gti, met, read_gti)

SHIFTS = (-0.5, -0.2, 0.2, 0.5)
MAX_PER_HOUR = 4000   # 每小时最多取这么多候选做对照，够统计量、不至于跑一天


def count(times, a, b):
    return int(np.searchsorted(times, b, "right") - np.searchsorted(times, a, "left"))


def measure(times, gti, t0, t1):
    left = (t0 - BASELINE, t0 - GUARD)
    right = (t1 + GUARD, t1 + BASELINE)
    n_bg = count(times, *left) + count(times, *right)
    t_bg = gti_overlap(*left, gti) + gti_overlap(*right, gti)
    if t_bg <= 0 or n_bg == 0:
        return None, 0.0
    rate = n_bg / t_bg
    out = []
    for half in CPD_HALF_WIDTHS:
        a, b = t0 - half, t1 + half
        out.append((count(times, a, b), rate * gti_overlap(a, b, gti)))
    nominal = 2 * BASELINE - 2 * GUARD - (t1 - t0)
    return out, t_bg / nominal if nominal > 0 else np.nan


def main():
    hours = [line.strip() for line in open(sys.argv[1]) if line.strip()]
    out_path, worker, workers = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    mine = [h for i, h in enumerate(hours) if i % workers == worker]
    rng = np.random.default_rng(20201210 + worker)

    writer = csv.writer(open(out_path, "w", newline=""))
    writer.writerow(["kind", "shift", "hour", "bin_us", "bg_live_frac"]
                    + [f"obs{i}" for i in range(len(CPD_HALF_WIDTHS))]
                    + [f"exp{i}" for i in range(len(CPD_HALF_WIDTHS))])

    totals, n_shift, n_rand, live_fracs = {}, 0, 0, []
    for iso_hour in mine:
        day = iso_hour[:10]
        sig_path = f"{SIGNAL_ROOT}/{day[:4]}/{day[5:7]}/{day.replace('-', '')}_signals.json"
        if not os.path.exists(sig_path):
            continue
        group = [s for s in json.load(open(sig_path)) if s["start"][:13] == iso_hour]
        if not group:
            continue
        if len(group) > MAX_PER_HOUR:
            idx = rng.choice(len(group), MAX_PER_HOUR, replace=False)
            group = [group[i] for i in sorted(idx)]

        path = hour_file(iso_hour + ":00:00", "cpd")
        if path is None:
            continue
        with fits.open(path, memmap=True) as hdus:
            gti = read_gti(hdus)
            parts = []
            for hdu in hdus:
                if hdu.name.startswith("EVENTS"):
                    data = hdu.data
                    k = np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE
                    parts.append(np.asarray(data["TIME"], float)[k])
        if not parts:
            continue
        allt = np.sort(np.concatenate(parts))
        times = allt[inside_gti(allt, gti)]
        if times.size == 0:
            continue
        widths = np.array([s["bin_size_best"] for s in group])

        for s in group:
            t0 = met(s["start"]) + s["delay"]
            w = s["bin_size_best"]
            for shift in SHIFTS:
                got, frac = measure(times, gti, t0 + shift, t0 + shift + w)
                if got is None:
                    continue
                writer.writerow(["shift", shift, iso_hour, f"{w * 1e6:.3f}", f"{frac:.4f}"]
                                + [g[0] for g in got] + [f"{g[1]:.6g}" for g in got])
                n_shift += 1
                for i, (o, e) in enumerate(got):
                    a, b = totals.setdefault(("shift", i), [0, 0.0])
                    totals[("shift", i)] = [a + o, b + e]

        if gti is not None:
            starts, stops = gti
            spans = stops - starts
            if spans.sum() > 0:
                weight = spans / spans.sum()
                n_draw = len(group)
                seg = rng.choice(len(spans), size=n_draw, p=weight)
                t0s = starts[seg] + rng.random(n_draw) * spans[seg]
                ws = rng.choice(widths, size=n_draw)
                for t0, w in zip(t0s, ws):
                    got, frac = measure(times, gti, t0, t0 + w)
                    if got is None:
                        continue
                    writer.writerow(["random", "", iso_hour, f"{w * 1e6:.3f}", f"{frac:.4f}"]
                                    + [g[0] for g in got] + [f"{g[1]:.6g}" for g in got])
                    n_rand += 1
                    live_fracs.append(frac)
                    for i, (o, e) in enumerate(got):
                        a, b = totals.setdefault(("random", i), [0, 0.0])
                        totals[("random", i)] = [a + o, b + e]
        print(f"  {iso_hour}: 候选 {len(group)}，CPD 事例 {times.size}", flush=True)

    print("=== 零假设对照 obs/exp（该回到 1.00）===")
    for kind in ("shift", "random"):
        for i, half in enumerate(CPD_HALF_WIDTHS):
            if (kind, i) not in totals:
                continue
            o, e = totals[(kind, i)]
            ratio = o / e if e > 0 else np.nan
            sigma = np.sqrt(max(o, 1)) / e if e > 0 else np.nan
            print("  %-7s ±%-6d µs: obs=%9d exp=%11.1f  比值 %.4f ± %.4f  (z = %+.2f)"
                  % (kind, int(half * 1e6), o, e, ratio, sigma, (ratio - 1) / sigma))
    print("样本：平移 %d，随机 %d" % (n_shift, n_rand))
    if live_fracs:
        lf = np.array(live_fracs)
        print("本底窗活时间 / 标称时长：中位 %.4f，被 GTI 切到（< 0.999）的占 %.2f%%，5 分位 %.4f"
              % (np.median(lf), (lf < 0.999).mean() * 100, np.percentile(lf, 5)))


if __name__ == "__main__":
    main()
