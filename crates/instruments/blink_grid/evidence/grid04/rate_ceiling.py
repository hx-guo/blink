"""GRID-04 的输出率上限，以及高速率处的同戳重数——用来分开两种共帧机制。

## 要分开的是什么

本底数据上，"触发后 w ≈ 3.8 µs 内 latch 其余各路"（窗模型）与"帧内其余各路以
约 15% 的概率被并进来"（接受率模型）给出同一个二重占比——两者按构造在 q ≪ 1 时
简并。**只有高入射率能分开它们**，因为饱和值不同：

| | 每帧其余路的并入概率 q | 极高率下 E[m] | **输出率上限 E[m]/τ** |
|---|---|---|---|
| 窗模型 | 1 − e^{−λ_d w} → 1 | → 4 | **140 kc/s** |
| 接受率模型 | 0.15·(1 − e^{−λ_d τ}) → 0.15 | → 1.45 | **51 kc/s** |

这个区分不是学术细节：TGF 落在共帧星上能留下几个计数，直接由它定，而那正是折扣
因子、也正是"只有 03B 探到 TGF"这条结论的支点。

## 量什么

逐 10 ms 格的四路合计计数 → 速率分布的高端；对速率最高的那些片再算同戳重数分布，
与两个模型并排。10 ms 是有意取的：它比帧长长三个量级（统计够），又比轨道上速率
变化的尺度短得多（不会把辐射带峰值平均掉）。

用法: python3 rate_ceiling.py <SAT> <out.csv> <YYYY/MM/DD> [更多日期...]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
TICK = 2.0**22
BIN = 0.010


def load_pass(path):
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        T = []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            T.append(t[(ty == 1) & ok & (e >= ETH)])
    return gs, ge, T


def main():
    sat, out = sys.argv[1], sys.argv[2]
    fh = open(out, "w")
    fh.write("day,pass_file,bin_rate_cps,n_bin,em,f1,f2,f3,f4\n")
    for daypath in sys.argv[3:]:
        vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, daypath)))
        if not vers:
            continue
        day = daypath.replace("/", "-")
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            try:
                gs, ge, T = load_pass(path)
            except Exception:
                continue
            allt = np.sort(np.concatenate(T))
            if allt.size < 1000:
                continue
            nb = int((ge - gs) / BIN) + 1
            idx = np.clip(((allt - gs) / BIN).astype(np.int64), 0, nb - 1)
            cnt = np.bincount(idx, minlength=nb)
            # 每次过境只留速率最高的 30 个 10 ms 格，够画高端分布又不写爆文件
            top = np.argsort(cnt)[::-1][:30]
            bounds = np.searchsorted(allt, gs + BIN * np.arange(nb + 1))
            for b in top:
                n = int(cnt[b])
                if n < 50:
                    continue
                seg = allt[bounds[b]:bounds[b + 1]]
                tk = np.rint(seg * TICK).astype(np.int64)
                ed = np.flatnonzero(np.diff(tk) != 0)
                sz = np.concatenate((ed + 1, [tk.size])) - np.concatenate(([0], ed + 1))
                f = [float(sz[sz == k].sum()) / n for k in (1, 2, 3, 4)]
                fh.write("%s,%s,%.1f,%d,%.4f,%.5f,%.5f,%.5f,%.5f\n"
                         % (day, path.split("/")[-1], n / BIN, n, float(sz.mean()),
                            f[0], f[1], f[2], f[3]))
            fh.flush()
    fh.close()
    print("done ->", out)


if __name__ == "__main__":
    main()
