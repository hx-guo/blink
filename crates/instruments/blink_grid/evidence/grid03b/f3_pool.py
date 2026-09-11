"""对 v10 的全部 GRID-03B 候选（不只显著的）逐个算 f₃、k₃ 与偶然期望 λ₃。

用来给"乙案"定阈：判据不是固定的 f₃ 阈，而是逐候选算偶然期望 λ₃、用 Poisson 检验
`P(K₃ ≥ k₃ | λ₃)` 否决。固定阈有一个结构缺陷——GRID-03B 四路探头、同探头不重格
⇒ 同戳簇最多 4 重，`min_number = 8` ⇒ **只含一个簇的候选 f₃ ≤ 4/8 = 0.5**，所以
`f₃ > 0.5` 对"一次穿越"的粒子恒不触发，挪到 0.45 也只堵住 n = 8 那一档。

对账口径（方法论硬要求）：最佳格 = `[start + delay, + bin_size_best]`，窗内准入事例数
必须等于 JSON 的 `count`，不到 100% 命中先修窗口再谈别的。事例准入与 `Event::keep` 一致。

**时间轴不能用纳秒整数。** MET 量级 1.3×10⁸ s，`t × 1e9 ≈ 1.3×10¹⁷` 已越过 2⁵³，
f64 乘出来的"纳秒"本身被量化到 16 ns，实测因此漏掉窗末端的事例（对账 5/17）。
正确的整数轴是 **tick = t × 2²²**（≈5.5×10¹⁴ < 2⁵³，而且事例时刻严格落在这个栅格上，
取整无损，实测残差 0.0），时戳比较一律在 tick 上做。窗口两端用 f64 秒比较并留半格
（q/2 = 119 ns）余量：两端都是事例本身的时刻，最近的异格事例也在 q 之外，不会误收。

λ₃ 的解析式（逐探头口径，期望的线性性，精确）：
    a_d = n_d / m，m = T / q，q = 2⁻²² s
    λ₃ = m · Σ_{|S|≥3} Π_{d∈S} a_d · Π_{d∉S} (1 − a_d)

用法（农场 4 worker）：
    python3 f3_pool.py --data <data_v10_dir>/GRID-03B --worker i --nworkers 4 -o out_i.csv
"""

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
from itertools import combinations

import numpy as np
from astropy.io import fits

REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
ARCHIVE = "/gecamfs/Exchange/GSDC/missions/GRID/{sat}/fits7/{y}/{m}/{d}/"
ENERGY_THRESHOLD_KEV = 30.0
Q_S = 2.0**-22
TICK = 1.0 / Q_S
TRIPLE = 3
TOL_S = 0.5 * Q_S  # 窗口两端的半格余量


def iso_to_met(s):
    """ISO 时刻 → MET 秒。整秒交给 datetime，小数部分单独加，不走字符串截断。"""
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - REF).total_seconds() + (float("0." + frac) if frac else 0.0)


def day_files(sat, date):
    base = ARCHIVE.format(sat=sat, y="%04d" % date.year, m="%02d" % date.month, d="%02d" % date.day)
    vers = sorted(glob.glob(base + "evt_v*"))
    return sorted(glob.glob(vers[-1] + "/*.fits")) if vers else []


def load_pass(path):
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        T, D = [], []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            et = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            keep = (et == 1) & ok & (e >= ENERGY_THRESHOLD_KEV)
            T.append(t[keep])
            D.append(np.full(int(keep.sum()), i, dtype=np.int8))
    t = np.concatenate(T)
    o = np.argsort(t, kind="stable")
    t = t[o]
    return dict(gs=gs, ge=ge, t=t, tick=np.rint(t * TICK).astype(np.int64),
                det=np.concatenate(D)[o])


def lambda3(counts, m):
    """逐探头口径的偶然三重簇个数期望。"""
    if m <= 0:
        return float("nan")
    a = [min(c / m, 1.0) for c in counts]
    dets = range(len(a))
    lam = 0.0
    for size in range(TRIPLE, len(a) + 1):
        for s in combinations(dets, size):
            pr = 1.0
            for d in dets:
                pr *= a[d] if d in s else (1.0 - a[d])
            lam += pr
    return m * lam


def pois_ge(k, lam):
    """P(X ≥ k)，X ~ Poisson(lam)。

    **直接累尾，不用 `1 − 前 k 项`。** 后者在尾部小的时候是灾难性相消：λ ~ 1e-3 时
    k = 3 已有 2.6e-8 的相对误差，**k ≥ 5 直接返回 0**（真值 8.3e-18）；λ = 1.3e-5、
    k = 3 时错 21%。本判据要在 1e-3 附近比大小，而全池里 λ₃ 能小到 1e-6 量级，
    正落在会出错的那一档。直接累尾在同样参数下相对误差 ≤ 5e-15。
    """
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    total = 0.0
    for i in range(k, k + 1000):
        term = math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1))
        total += term
        if term < 1e-18 * max(total, 1e-300):
            break
    return min(1.0, total)


def cluster_stats(tick):
    """返回 (三重簇个数, 落在 ≥3 重簇里的事例数, 最长同戳串)。tick 已升序。"""
    if tick.size == 0:
        return 0, 0, 0
    edge = np.flatnonzero(np.diff(tick) != 0)
    starts = np.concatenate(([0], edge + 1))
    ends = np.concatenate((edge + 1, [tick.size]))
    size = ends - starts
    big = size >= TRIPLE
    return int(big.sum()), int(size[big].sum()), int(size.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="v10 的 <data>/GRID-03B 目录")
    ap.add_argument("--sat", default="GRID-03B")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    args = ap.parse_args()

    days = sorted(glob.glob(os.path.join(args.data, "*", "*", "*_signals.json")))
    days = days[args.worker::args.nworkers]
    out = []
    bad = 0
    for jp in days:
        cands = json.load(open(jp))
        if not cands:
            continue
        # 先只读 GTI 头把候选按过境分组，再逐个过境加载一次事例流：一天最多 21 次
        # 过境，全缓存住会吃掉几百 MB，没必要。
        date0 = (REF + dt.timedelta(seconds=iso_to_met(cands[0]["start"]))).date()
        spans = []
        for d in (date0 - dt.timedelta(days=1), date0, date0 + dt.timedelta(days=1)):
            for pth in day_files(args.sat, d):
                try:
                    g = fits.getdata(pth, extname="GTI")
                    spans.append((float(np.asarray(g["START"])[0]),
                                  float(np.asarray(g["STOP"])[0]), pth))
                except Exception:
                    continue
        groups = {}
        for c in cands:
            t0 = iso_to_met(c["start"]) + c["delay"]
            hit = [pth for gs, ge, pth in spans if gs <= t0 <= ge]
            if not hit:
                bad += 1
                continue
            groups.setdefault(hit[0], []).append((c, t0))

        for path, items in groups.items():
            p = load_pass(path)
            for c, t0 in items:
                t1 = t0 + c["bin_size_best"]
                lo = np.searchsorted(p["t"], t0 - TOL_S, side="left")
                hi = np.searchsorted(p["t"], t1 + TOL_S, side="right")
                tick = p["tick"][lo:hi]
                det = p["det"][lo:hi]
                n = tick.size
                if n != int(c["count"]):
                    bad += 1
                    continue
                counts = np.bincount(det, minlength=4)
                k3, ev3, longest = cluster_stats(tick)
                T = c["bin_size_best"]
                lam = lambda3(counts, T / Q_S)
                pos = c.get("position") or {}
                out.append(dict(
                    start=c["start"], n=n, T_us="%.3f" % (T * 1e6),
                    fa="%.6e" % c["false_positive_per_year"],
                    lat="%.3f" % pos["latitude"] if pos else "",
                    lon="%.3f" % pos["longitude"] if pos else "",
                    n_det=",".join(str(int(x)) for x in counts),
                    f2="%.4f" % (longest / n), f3="%.4f" % (ev3 / n), k3=k3,
                    lambda3="%.4e" % lam, p_pois="%.4e" % pois_ge(k3, lam),
                ))
            del p
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("worker %d: 写出 %d 个候选，对账失败/找不到过境 %d 个" % (args.worker, len(out), bad))


if __name__ == "__main__":
    main()
