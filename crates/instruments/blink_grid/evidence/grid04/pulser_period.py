"""脉冲的**基频**与精确周期：共帧星与 GRID-03B 是不是同一个设置。

`grid03b` 在 03B 的 2022-03-19 过境上把四重簇间隔按梳齿拆开，发现基频在
**4194/4195 tick（1 kHz）**，而 8388/8389（2 kHz 的一半，即 500 Hz）只是"漏掉一发"
的二次谐波。**共帧星上是不是也这样，决定清单里所有频率数字，也决定安全余量**——
1 kHz 时一个 1 ms 搜索窗按相位能装下 2 发脉冲 = 8 个四路同戳计数 = `min_number`，
而 v11 在共帧星上已取消同戳门。

本脚本量两件事：

1. **梳齿逐齿占比**：k = 1…N 的 `k × P₀` 各有多少对，**1× 那一格是不是空的**；
2. **精确周期**：把落在梳齿上的四重簇按脉冲序号做线性回归（残差 rms 应当在 tick
   量级），区分"恰好 2.000000 ms"与"2 × 1.000110 ms"——前者说明是独立设定，
   后者说明同一个振荡器分频。

用法: python3 pulser_period.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from pulser import load                                          # noqa: E402

TICK = 2.0**22
CASES = (("GRID-04", "2022-05-26", "2205260508"),
         ("GRID-04", "2022-12-06", "2212060715"),
         ("GRID-04", "2022-07-02", "2207020724"),
         ("GRID-02", "2021-01-15", "2101151710"),
         ("GRID-02", "2020-12-14", "2012141710"),
         ("GRID-03B", "2022-03-19", "2203190605"))


def quad_times(sat, day, key):
    got = load(sat, day, key)
    if got is None:
        return None
    gs, ge, t, pi, dd, egeo = got
    tk = np.rint(t * TICK).astype(np.int64)
    ed = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], ed + 1))
    sz = np.concatenate((ed + 1, [tk.size])) - st
    big = np.flatnonzero(sz >= 4)
    return tk[st[big]].astype(np.float64)


def main():
    print("%-9s %-11s %7s %8s %8s %8s %8s %9s"
          % ("星", "日期", "四重簇", "1×占比", "2×占比", "3×占比", "梳齿合计", "基频档计数"))
    keep = {}
    for sat, day, key in CASES:
        ft = quad_times(sat, day, key)
        if ft is None or ft.size < 200:
            continue
        d = np.diff(ft)
        tot = d.size
        base = 4194.5                     # 1 kHz 的名义 tick 数（1 ms × 2²² = 4194.304）
        occ = []
        for k in (1, 2, 3):
            lo, hi = base * k - 4, base * k + 4
            occ.append(int(((d >= lo) & (d <= hi)).sum()))
        comb = 0
        for k in range(1, 40):
            lo, hi = base * k - 4, base * k + 4
            comb += int(((d >= lo) & (d <= hi)).sum())
        print("%-9s %-11s %7d %8.4f %8.4f %8.4f %8.4f %9d"
              % (sat, day, ft.size, occ[0] / tot, occ[1] / tot, occ[2] / tot,
                 comb / tot, occ[0]))
        keep[(sat, day)] = (ft, d)
    print()
    print("精确周期（落在梳齿上的四重簇按脉冲序号线性回归）")
    print("%-9s %-11s %9s %14s %13s %9s"
          % ("星", "日期", "用到的点", "周期 tick", "周期 ms", "残差 ns"))
    for (sat, day), (ft, d) in keep.items():
        step = 4194.5 if sat == "GRID-03B" else 8388.6
        ok = np.zeros(ft.size, bool)
        for k in range(1, 40):
            m = np.abs(d - step * k) < 4
            ok[:-1] |= m
            ok[1:] |= m
        f = ft[ok]
        if f.size < 300:
            continue
        n = np.rint((f - f[0]) / step)
        # 只取序号严格递增的一段，避免粒子四重簇混进来把回归拉偏
        good = np.concatenate(([True], np.diff(n) > 0))
        f, n = f[good], n[good]
        A = np.vstack([n, np.ones(n.size)]).T
        sol, *_ = np.linalg.lstsq(A, f, rcond=None)
        rms = float(np.sqrt(np.mean((f - A @ sol) ** 2)))
        print("%-9s %-11s %9d %14.4f %13.6f %9.0f"
              % (sat, day, f.size, sol[0], sol[0] / TICK * 1e3, rms / TICK * 1e9))


if __name__ == "__main__":
    main()
