"""f₃ 的数据驱动本底对照：真实本底里三重同戳有多少、按窗长折算到候选窗是多少。

`f3_chance.py` 算的是"量化偶然"——n 个事例在 m 个时戳格上撞在一起的概率，答案是
1e-5 量级。但真实本底里还有**真的穿星粒子**，它们本身就产生三重同戳。所以判据能不能
立，还要回答第二个问题：**一个真 TGF 的窗里偶然套进一个本底粒子三重的概率是多少。**

做法（逐候选，都在该候选自己那次过境的 keep 后事例流上）：

1. 本底段 = 过境内、距暴发中心 > 1 s 的部分（另给一个 ±30 s 的局部版，速率更贴近）。
2. 数本底段里重数 ≥ 3 的同戳簇个数 k₃，除以本底时长得**本底三重簇率** r₃ (个/s)。
   时戳按 tick = round(TIME × 2²²) 取整，`TIME` 严格落在该栅格上，取整无损。
3. 折算到候选窗：λ₃ᵇᵏᵍ = r₃ × T。一个簇整体落在窗内或窗外，所以
   P(窗里套进 ≥1 个本底三重) = 1 − exp(−λ₃ᵇᵏᵍ) ≈ λ₃ᵇᵏᵍ。
4. 对照 B（复刻旧口径）：随机取 n 个**连续**本底事例算 f₃，报 f₃ > 0 的占比与这 n 个
   事例的时间跨度中位。旧文里"本底对照 17500 次抽样 7% 非零"用的就是这一口径——
   它匹配的是**计数**不是**窗长**，跨度是毫秒量级而候选窗是几十到几百微秒，
   两者不可直接与候选的 f₃ 比。本脚本把跨度一起报出来，好看清差了多少倍。

用法（农场 4 worker）：
    python3 f3_background.py --index <burst_events>/index.csv --worker i --nworkers 4 -o out_i.csv
"""

import argparse
import csv
import datetime as dt
import glob
import os

import numpy as np
from astropy.io import fits

REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
ARCHIVE = "/gecamfs/Exchange/GSDC/missions/GRID/{sat}/fits7/{y}/{m}/{d}/"
ENERGY_THRESHOLD_KEV = 30.0
TICK = 2.0**22  # 1 / (2^-22 s)
TRIPLE = 3
EXCLUDE_S = 1.0
LOCAL_S = 30.0


def iso_to_met(s):
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - REF).total_seconds() + (float("0." + frac) if frac else 0.0)


def day_files(sat, date):
    base = ARCHIVE.format(sat=sat, y="%04d" % date.year, m="%02d" % date.month, d="%02d" % date.day)
    vers = sorted(glob.glob(base + "evt_v*"))
    return sorted(glob.glob(vers[-1] + "/*.fits")) if vers else []


def load_pass(path):
    """准入与 `Event::keep` 一致：EVT_TYPE==1、PI ∈ [1, n_channels)、E_MIN ≥ 30 keV。"""
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
    return dict(gs=gs, ge=ge, t=t[o], det=np.concatenate(D)[o])


def clusters_ge(ticks, k=TRIPLE):
    """已升序的整数 tick 序列里，重数 ≥ k 的同戳簇个数与其中的事例数。"""
    if ticks.size == 0:
        return 0, 0
    edge = np.flatnonzero(np.diff(ticks) != 0)
    starts = np.concatenate(([0], edge + 1))
    ends = np.concatenate((edge + 1, [ticks.size]))
    size = ends - starts
    big = size >= k
    return int(big.sum()), int(size[big].sum())


def f3_of_rows(x):
    """x: (trials, n) 每行排好序的 tick，返回每行落在 ≥3 重簇里的事例数。"""
    trials, n = x.shape
    idx = np.arange(n)
    starts = np.empty(x.shape, dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = x[:, 1:] != x[:, :-1]
    ends = np.empty(x.shape, dtype=bool)
    ends[:, -1] = True
    ends[:, :-1] = starts[:, 1:]
    s_i = np.maximum.accumulate(np.where(starts, idx, 0), axis=1)
    e_i = np.minimum.accumulate(np.where(ends, idx, n - 1)[:, ::-1], axis=1)[:, ::-1]
    return ((e_i - s_i + 1) >= TRIPLE).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--draws", type=int, default=200_000, help="对照 B 的抽样次数")
    ap.add_argument("--mc-windows", type=int, default=400_000, help="对照 A 的随机窗数（校验解析折算）")
    args = ap.parse_args()

    rng = np.random.default_rng(20260911)
    rows = [r for r in csv.DictReader(open(args.index))]
    rows = [r for i, r in enumerate(rows) if i % args.nworkers == args.worker]
    out = []
    for meta in rows:
        sat = meta["sat"]
        t0 = float(meta["met_best_start"])
        t_us = float(meta["bin_size_best_us"])
        n = int(meta["count"])
        date = REF + dt.timedelta(seconds=t0)
        path = None
        for p in day_files(sat, date.date()):
            if os.path.basename(p) == meta["pass_file"]:
                path = p
                break
        if path is None:
            for d in (date.date() - dt.timedelta(days=1), date.date()):
                for p in day_files(sat, d):
                    if os.path.basename(p) == meta["pass_file"]:
                        path = p
        if path is None:
            print("找不到过境文件", meta["start"][:23], meta["pass_file"])
            continue
        p = load_pass(path)
        ticks_all = np.rint(p["t"] * TICK).astype(np.int64)
        # 取整无损的自检：TIME 严格落在 2^-22 s 栅格上
        resid = np.abs(p["t"] * TICK - ticks_all).max() if p["t"].size else 0.0

        centre = t0 + 0.5 * t_us * 1e-6
        far = np.abs(p["t"] - centre) > EXCLUDE_S
        loc = far & (np.abs(p["t"] - centre) < LOCAL_S)
        dur_far = max((p["ge"] - p["gs"]) - 2 * EXCLUDE_S, 1e-9)
        dur_loc = max(2 * (LOCAL_S - EXCLUDE_S), 1e-9)

        rec = dict(start=meta["start"][:23], lightning=meta["lightning"], n=n,
                   T_us="%.2f" % t_us, tick_resid="%.1e" % resid)
        for tag, mask, dur in (("far", far, dur_far), ("loc", loc, dur_loc)):
            tk = ticks_all[mask]
            k3, ev3 = clusters_ge(tk)
            rate = tk.size / dur
            r3 = k3 / dur
            rec["%s_n" % tag] = int(tk.size)
            rec["%s_dur_s" % tag] = "%.1f" % dur
            rec["%s_rate_cps" % tag] = "%.1f" % rate
            rec["%s_k3" % tag] = k3
            rec["%s_frac_ev_in3" % tag] = "%.3e" % (ev3 / tk.size if tk.size else float("nan"))
            rec["%s_r3_per_s" % tag] = "%.4f" % r3
            rec["%s_lambda3_in_T" % tag] = "%.3e" % (r3 * t_us * 1e-6)
        # 对照 B：随机取 n 个连续本底事例（旧口径），报 f₃>0 占比与跨度
        tk = ticks_all[far]
        if tk.size > n + 10:
            i0 = rng.integers(0, tk.size - n, size=args.draws)
            seg = tk[i0[:, None] + np.arange(n)[None, :]]
            hit = f3_of_rows(seg)
            span_us = (seg[:, -1] - seg[:, 0]) / TICK * 1e6
            rec["ctrlB_draws"] = args.draws
            rec["ctrlB_frac_f3_pos"] = "%.4f" % float((hit > 0).mean())
            rec["ctrlB_span_us_med"] = "%.0f" % float(np.median(span_us))
            rec["ctrlB_span_over_T"] = "%.0f" % float(np.median(span_us) / t_us)
        # 对照 A：随机放同样窗长的窗，看里面有没有三重簇（校验 λ₃ 的解析折算）
        if tk.size:
            lo = p["gs"] + EXCLUDE_S
            hi = p["ge"] - EXCLUDE_S - t_us * 1e-6
            if hi > lo:
                w0 = rng.uniform(lo, hi, size=args.mc_windows)
                a = np.searchsorted(p["t"], w0)
                b = np.searchsorted(p["t"], w0 + t_us * 1e-6)
                sel = np.flatnonzero(b - a >= TRIPLE)
                pos = 0
                for i in sel:
                    k3w, _ = clusters_ge(ticks_all[a[i]:b[i]])
                    pos += 1 if k3w else 0
                rec["ctrlA_windows"] = args.mc_windows
                rec["ctrlA_frac_f3_pos"] = "%.3e" % (pos / args.mc_windows)
                rec["ctrlA_max_n"] = int((b - a).max())
                rec["ctrlA_frac_n_ge8"] = "%.3e" % float(((b - a) >= 8).mean())
        out.append(rec)
        print("done", rec["start"], "far r3=%s/s lambda3=%s ctrlB=%s"
              % (rec["far_r3_per_s"], rec["far_lambda3_in_T"], rec.get("ctrlB_frac_f3_pos")),
              flush=True)

    if out:
        keys = sorted({k for r in out for k in r}, key=lambda k: list(out[0]).index(k) if k in out[0] else 99)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
