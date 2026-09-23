"""跨探头相邻事例时间间隔的逐 tick 直方，给讲图 grid_talk_4_readout 用。

口径（与 diag/grid04/grid04_frame.py、diag/grid02/grid02_frame.py 相同的判据，只加了
同速率条件）：
  - 事例：EVT_TYPE == 1（不筛能量），四路合并按时间排序；
  - 时间轴：tick = round(TIME * 2^22)，四星时戳都严格落在这个栅格上；
  - 相邻对：合并流里相邻两个事例；"跨探头" = 两者探头号不同；
  - 同条件：把每个文件切成 10 s 片，只保留四路合计率在 [RATE_LO, RATE_HI) 的片，
    相邻对归到前一个事例所在的片；
  - 片上标定脉冲（02/04/07 为 500 Hz、03B 为 1 kHz 的四路同戳）会在 tick 0 堆出假尖峰，
    按"四路同戳簇率 > PULSER_MAX 个/s"整片剔除，剔除片数写进汇总。正常过境的四重簇率
    约 1 个/s（未决项 19 的本底三重簇率 1.06 个/s 同量级），与脉冲差两个多数量级。

用法: grid_readout_hist.py <sat> <YYYY/MM/DD> [<YYYY/MM/DD> ...] -o <前缀>
输出: <前缀>_hist.csv（tick, n_cross, n_same）与 <前缀>_summary.csv
"""
import argparse, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
TICK = 2.0 ** -22
NBIN = 1200          # 0..1199 tick = 0..286 µs
SLICE = 10.0
RATE_LO, RATE_HI = 500.0, 1000.0
PULSER_MAX = 50.0


def day_files(sat, day):
    dd = f"{G}/{sat}/fits7/{day}"
    v = sorted(glob.glob(dd + "/evt_v*"))
    return sorted(glob.glob(v[-1] + "/*.fits")) if v else []


def read(path):
    with fits.open(path, memmap=False) as h:
        ts, ds = [], []
        for d in range(4):
            e = h["EVENTS%d" % d].data
            t = np.asarray(e["TIME"], np.float64)
            m = np.asarray(e["EVT_TYPE"]) == 1
            ts.append(t[m]); ds.append(np.full(int(m.sum()), d, np.int8))
        gti = h["GTI"].data
        g = [(float(a), float(b)) for a, b in zip(gti["START"], gti["STOP"])]
    t = np.concatenate(ts); d = np.concatenate(ds)
    o = np.argsort(t, kind="stable")
    return t[o], d[o], g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sat"); ap.add_argument("days", nargs="+"); ap.add_argument("-o", required=True)
    a = ap.parse_args()
    hc = np.zeros(NBIN, np.int64); hs = np.zeros(NBIN, np.int64)
    S = dict(files=0, slices_all=0, slices_rate=0, slices_pulser=0, live_s=0.0, events=0,
             pairs_all=0, pairs_cross=0)
    rates = []
    for day in a.days:
        for f in day_files(a.sat, day):
            try:
                t, d, gti = read(f)
            except Exception as exc:
                print("skip", f, exc, file=sys.stderr); continue
            if t.size < 10:
                continue
            S["files"] += 1
            tk = np.round(t / TICK).astype(np.int64)
            # 10 s 片只在 GTI 内切，片长取与 GTI 的交
            for g0, g1 in gti:
                edges = np.arange(g0, g1, SLICE)
                for s0 in edges:
                    s1 = min(s0 + SLICE, g1)
                    if s1 - s0 < 0.5 * SLICE:
                        continue
                    i0, i1 = np.searchsorted(t, [s0, s1])
                    n = i1 - i0
                    S["slices_all"] += 1
                    rate = n / (s1 - s0)
                    if not (RATE_LO <= rate < RATE_HI):
                        continue
                    S["slices_rate"] += 1
                    k = tk[i0:i1]; dd = d[i0:i1]
                    # 四路同戳簇：同一 tick 上 4 个事例
                    u, c = np.unique(k, return_counts=True)
                    if (c >= 4).sum() / (s1 - s0) > PULSER_MAX:
                        S["slices_pulser"] += 1
                        continue
                    rates.append(rate)
                    S["live_s"] += s1 - s0; S["events"] += int(n)
                    dt = np.diff(k)
                    cross = dd[1:] != dd[:-1]
                    S["pairs_all"] += dt.size; S["pairs_cross"] += int(cross.sum())
                    m = dt < NBIN
                    hc += np.bincount(dt[m & cross], minlength=NBIN)[:NBIN]
                    hs += np.bincount(dt[m & ~cross], minlength=NBIN)[:NBIN]
    S["rate_median"] = float(np.median(rates)) if rates else float("nan")
    S["rate_p16"] = float(np.percentile(rates, 16)) if rates else float("nan")
    S["rate_p84"] = float(np.percentile(rates, 84)) if rates else float("nan")
    with open(a.o + "_hist.csv", "w") as fo:
        fo.write("tick,n_cross,n_same\n")
        for i in range(NBIN):
            fo.write("%d,%d,%d\n" % (i, hc[i], hs[i]))
    with open(a.o + "_summary.csv", "w") as fo:
        fo.write("key,value\n")
        fo.write("sat,%s\ndays,%s\nrate_lo,%g\nrate_hi,%g\nslice_s,%g\npulser_max_per_s,%g\n"
                 % (a.sat, " ".join(a.days), RATE_LO, RATE_HI, SLICE, PULSER_MAX))
        for k, v in S.items():
            fo.write("%s,%s\n" % (k, v))
    print(a.sat, S)


main()
