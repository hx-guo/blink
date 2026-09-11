"""注入检验：τ 合并对一个真 TGF 要收多少代价，实测而不是解析。

解析式 `1 − exp(−n·τ/W)` 只算注入光子彼此撞车，漏掉两件真实的事：注入光子会跟
**本底计数**撞车，本底自己还带着同戳簇（一个粒子穿越是 2–8 个计数挤在 60 ns 里，
落进 TGF 窗口就会把窗内的某个注入光子一起并掉）。所以要把合成 TGF 注进真实的
事例流里量。

口径与搜索一致：先准入、再双增益去重（同探头、死时间内、一高一低），然后按 τ
做跨探头单链接合并。**注入的时戳必须先对齐到归档的 float64 格子上**（q = 29.8 ns），
否则注入光子永远不会跟任何东西同戳，代价会被系统性低估。
探头号按当小时实测的逐路计数比例抽，不用均匀分布——25 路的率差到 2 倍。

窗内时间分布给两种：均匀，和前段陡（半高斯，σ = W/4）。真 TGF 不是平的，
峰越陡局部密度越高、合并损失越大，所以陡的那一种是代价的上界。

用法: python3 ga_inject.py <YYYY-MM-DD> <HH> <输出 JSON>
"""

import glob
import json
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_CHANNEL, OVERFLOW, NORMAL = 54, 448, 1
DEADTIME_S = 4e-6
TAU = 1.5e-7
MIN_NUMBER = 8
N_TRIAL = 2000
GRID = [(8, 10), (8, 30), (10, 30), (10, 100), (15, 100), (20, 100),
        (20, 300), (30, 300), (30, 100), (50, 300), (50, 1000), (100, 1000)]


def hour_file(day, hh):
    d = f"{ROOT}/{day.replace('-', '/')}/GECAM_A/GRD_evt"
    stem = f"gag_evt_{day[2:4]}{day[5:7]}{day[8:10]}_{hh}_v"
    fs = sorted(glob.glob(f"{d}/{stem}*.fits"))
    return fs[-1] if fs else None


def load(path):
    times, dets = [], []
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            d = hdu.data
            if d is None or len(d) == 0:
                continue
            pi = np.asarray(d["PI"], np.int32)
            et = np.asarray(d["EVT_TYPE"], np.int32)
            gt = np.asarray(d["GAIN_TYPE"], np.int32)
            t = np.asarray(d["TIME"], float)
            keep = (et == NORMAL) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW)
            t, gt = t[keep], gt[keep]
            if t.size == 0:
                continue
            o = np.lexsort((-gt, t))
            t, gt = t[o], gt[o]
            drop = np.zeros(t.size, bool)
            drop[1:] = (t[1:] - t[:-1] < DEADTIME_S) & (gt[1:] != gt[:-1])
            t = t[~drop]
            times.append(t)
            dets.append(np.full(t.size, int(hdu.name[-2:]), np.int16))
    time = np.concatenate(times)
    o = np.argsort(time, kind="stable")
    return time[o], np.concatenate(dets)[o]


def n_clusters(t, d, tau):
    """跨探头单链接：同探头的相邻对不链接（死时间 3.99 µs 已保证同探头不会挤在一起）。"""
    o = np.argsort(t, kind="stable")
    t, d = t[o], d[o]
    link = (np.diff(t) <= tau) & (np.diff(d) != 0)
    return int((~link).sum()) + 1


def main():
    day, hh, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    time, det = load(hour_file(day, hh))
    q = float(np.spacing(float(time[0])))
    span = float(time[-1] - time[0])
    rate = time.size / span
    print(f"{day} {hh}h  去重后事例 {time.size:,}  跨度 {span:.0f}s  率 {rate:.0f} c/s  q={q*1e9:.2f}ns",
          flush=True)

    # 逐路计数比例，注入时按它抽探头号
    dets, counts = np.unique(det, return_counts=True)
    weight = counts / counts.sum()
    print(f"25 路计数占比 min/max = {weight.min()*100:.2f}% / {weight.max()*100:.2f}% "
          f"（极差 {weight.max()/weight.min():.2f} 倍）", flush=True)

    rng = np.random.default_rng(20240111)
    lo, hi = float(time[0]) + 1.0, float(time[-1]) - 1.0
    report = {"day": day, "hour": hh, "rate": rate, "q_ns": q * 1e9, "tau_ns": TAU * 1e9,
              "n_trial": N_TRIAL, "grid": []}

    print(f"\n{'n':>4} {'W(us)':>7} {'剖面':>5} | {'合并后计数中位':>13} {'损失中位':>9} "
          f"{'损失均值':>9} {'掉到<8 占比':>11} | {'解析式':>7}")
    for n, w_us in GRID:
        w = w_us * 1e-6
        for profile in ("uniform", "peaked"):
            t0 = rng.uniform(lo, hi, N_TRIAL)
            keep_n, loss = np.empty(N_TRIAL, int), np.empty(N_TRIAL)
            for i in range(N_TRIAL):
                if profile == "uniform":
                    off = rng.uniform(0, w, n)
                else:
                    off = np.abs(rng.normal(0, w / 4, n)) % w
                ts = t0[i] + off
                ts = np.round(ts / q) * q          # 对齐到归档格子
                ds = rng.choice(dets, n, p=weight)
                a = np.searchsorted(time, t0[i] - TAU, "left")
                b = np.searchsorted(time, t0[i] + w + TAU, "right")
                bg_t, bg_d = time[a:b], det[a:b]
                n_bg = n_clusters(bg_t, bg_d, TAU) if b > a else 0
                n_mrg = n_clusters(np.concatenate([bg_t, ts]),
                                   np.concatenate([bg_d, ds]), TAU)
                # 注入让簇数多了几个 —— 这就是搜索能看见的"多出来的事例数"
                keep_n[i] = n_mrg - n_bg
                loss[i] = 1 - keep_n[i] / n
            ana = 1 - np.exp(-n * TAU / w)
            below = float((keep_n < MIN_NUMBER).mean())
            print(f"{n:4d} {w_us:7.0f} {profile:>5} | {np.median(keep_n):13.1f} "
                  f"{np.median(loss)*100:8.2f}% {loss.mean()*100:8.2f}% {below*100:10.2f}% | "
                  f"{ana*100:6.2f}%", flush=True)
            report["grid"].append(dict(n=n, w_us=w_us, profile=profile,
                                       keep_median=float(np.median(keep_n)),
                                       loss_median=float(np.median(loss)),
                                       loss_mean=float(loss.mean()),
                                       frac_below_min=below, analytic=float(ana)))
    json.dump(report, open(out_path, "w"))


if __name__ == "__main__":
    main()
