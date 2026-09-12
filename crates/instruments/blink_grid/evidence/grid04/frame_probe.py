"""从 GRID-04 自己的数据里量读出参数：帧长、同帧共戳、实际死时间。

不继承任何已有数字。三个量：

1. **逐探头相邻事例 dt 的逐 tick 直方**（tick = 2⁻²² s）。硬边沿 = "从零跳到峰"的
   那一档，取法是首个计数超过峰值 1% 的 tick，避免被偶发毛刺带偏（`min(dt)` 会）。
2. **跨探头相邻事例 dt** 的形态：落在开区间 (0, τ) 的占比、恰好为 0 的占比。
   共帧读出要求前者近乎为空（帧内每路最多 1 个、全帧共用触发时戳），
   四路独立读出则连续铺满。
3. **同戳簇的探头构成**：簇里有几个不同探头。共帧下簇大小 = 帧内亮路数 ≤ 4。

用法: python3 frame_probe.py <SAT> <YYYY/MM/DD> [最多几次过境]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
TICK = 2.0**22


def load_pass(path):
    """按 `Event::keep` 的三条准入读一次过境：EVT_TYPE==1、1 ≤ PI < n_ch、E_MIN ≥ 30 keV。"""
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        out, n_all, n_t2 = [], 0, 0
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            n_all += t.size
            n_t2 += int((ty == 2).sum())
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            out.append(np.sort(t[(ty == 1) & ok & (e >= ETH)]))
    return gs, ge, out, n_all, n_t2


def edge_tick(dt_ticks, nmax=400):
    """硬边沿：首个计数 ≥ 峰值 1% 的 tick。返回 (边沿, 众数, 低于边沿的占比)。"""
    h = np.bincount(dt_ticks[dt_ticks <= nmax], minlength=nmax + 1)
    if h.sum() == 0:
        return -1, -1, float("nan")
    peak = h.max()
    idx = np.flatnonzero(h >= 0.01 * peak)
    e = int(idx[0]) if idx.size else -1
    return e, int(h.argmax()), float((dt_ticks < e).mean())


def main():
    sat, day = sys.argv[1], sys.argv[2]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, day)))
    files = sorted(glob.glob(vers[-1] + "/*.fits"))[:lim]
    for path in files:
        gs, ge, T, n_all, n_t2 = load_pass(path)
        tot = sum(x.size for x in T)
        if tot < 1000:
            continue
        print("\n=== %s  GTI %.1f s  keep %d (%.0f c/s)  EVT_TYPE=2 %.2f%%"
              % (path.split("/")[-1], ge - gs, tot, tot / (ge - gs), 100.0 * n_t2 / max(n_all, 1)))
        for i, t in enumerate(T):
            if t.size < 100:
                print("  det%d  n=%d (太少)" % (i, t.size))
                continue
            dtk = np.rint(np.diff(t) * TICK).astype(np.int64)
            e, mode, below = edge_tick(dtk)
            print("  det%d  n=%-7d %5.0f c/s  边沿 %3d tick = %8.4f µs  众数 %3d  低于边沿 %.2e"
                  % (i, t.size, t.size / (ge - gs), e, e / TICK * 1e6, mode, below))
        # 跨探头：把四路合起来排序，看相邻对
        allt = np.sort(np.concatenate(T))
        dtk = np.rint(np.diff(allt) * TICK).astype(np.int64)
        tau = 120
        print("  跨探头 dt==0 %.4f   0<dt<%d tick %.3e   dt in[%d,%d] %.4f"
              % ((dtk == 0).mean(), tau, ((dtk > 0) & (dtk < tau)).mean(),
                 tau, tau + 2, ((dtk >= tau) & (dtk <= tau + 2)).mean()))
        # 跨探头 dt 在 1..200 tick 的逐档计数，用来看帧边界的位置
        sub = dtk[(dtk >= 1) & (dtk <= 200)]
        h = np.bincount(sub, minlength=201)
        nz = [(k, int(h[k])) for k in range(1, 201) if h[k] > 0]
        print("  跨探头 1..200 tick 非零档（前 12 个）: %s" % nz[:12])
        # 同戳簇构成
        tk = np.rint(allt * TICK).astype(np.int64)
        edge = np.flatnonzero(np.diff(tk) != 0)
        st = np.concatenate(([0], edge + 1))
        sz = np.concatenate((edge + 1, [tk.size])) - st
        print("  簇大小分布 1..5+: %s   max %d   frac_ev_in3 %.5f"
              % ([int((sz == k).sum()) for k in (1, 2, 3, 4)] + [int((sz >= 5).sum())],
                 int(sz.max()), sz[sz >= 3].sum() / tk.size))


if __name__ == "__main__":
    main()
