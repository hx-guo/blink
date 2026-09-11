"""三种共帧读出机制的模拟，用来标定估计量对目标错误的灵敏度（正/负对照）。

要分开的三件事（都能产生"跨探头 dt 要么恰好 0、要么 ≥ τ"的形态，只看那个分不出来）：

- **丢弃**：触发后 w 内其余各路被 latch 进来共用时戳，落在 (w, τ) 的击中**丢掉**。
- **排队**：落在 (w, τ) 的击中**进缓冲**，在随后的帧里各自以新时戳发出（帧背靠背）。
- **混合**：一半排队一半丢弃。

三者对**计数**的后果完全不同——丢弃真的少计数，排队只把时间结构梳成 τ 的等间距——
而折扣因子、进而"只有 GRID-03B 探到 TGF"这条结论直接取决于是哪一种。

## 估计量（`readout_window.py` 用的就是这两个）

    Λ_mean = 1/(平均帧间隔 − τ)      Λ_tail = 由帧间隔分布尾部（Δ ≥ τ + 1/Λ_mean）定
    β = 1 − Λ_tail/Λ_mean            守恒比 = R_obs / Λ_true

纯丢弃下帧间隔严格是 τ + Exp(Λ)，所以 β ≡ 0；排队会插入一批间隔恰为 τ 的背靠背帧，
把均值拉短而不动尾部斜率，所以 β > 0。**本脚本在已知真值上量这两个估计量**，
这样实测的 β 与守恒比才读得懂——正对照是"排队的合成数据能不能被认成排队"，
负对照是"丢弃的合成数据会不会被误认成排队"。

用法: python3 sim_readout.py
"""

import numpy as np

TICK = 2.0**22
TAU = 120 / TICK
W = 4.0e-6
RNG = np.random.default_rng(20260911)


def simulate(lam_tot, mode, n_target=400000):
    """按机制生成输出流。返回 (帧触发时刻, 每帧事例数, 输出事例数, 入射事例数, 时长)。"""
    dur = n_target / lam_tot
    n = RNG.poisson(lam_tot * dur)
    t = np.sort(RNG.random(n) * dur)
    det = RNG.integers(0, 4, n)
    ftime, fsize = [], []
    queue = 0
    i = 0
    cur_end = -1.0
    while True:
        if queue > 0:
            start = cur_end          # 队列非空：上一帧一结束立刻背靠背开下一帧
            queue -= 1
        else:
            while i < n and t[i] < cur_end:
                i += 1               # 丢弃模式下残留的帧内击中（正常路径已跳过）
            if i >= n:
                break
            start = t[i]
            i += 1
        size = 1
        end_latch, end_frame = start + W, start + TAU
        latched = set()
        while i < n and t[i] < end_latch:
            if det[i] not in latched and len(latched) < 3:
                latched.add(det[i])
                size += 1
            i += 1
        spill = 0
        while i < n and t[i] < end_frame:
            spill += 1
            i += 1
        if mode == "queue":
            queue += spill
        elif mode == "mix":
            queue += spill // 2
        ftime.append(start)
        fsize.append(size)
        cur_end = end_frame
    ft = np.asarray(ftime)
    fs = np.asarray(fsize)
    return ft, fs, int(fs.sum()), n, dur


def estimate(ftime, nev, dur):
    d = np.diff(ftime)
    d = d[d > 0]
    lam_mean = 1.0 / (dur / ftime.size - TAU)
    tail0 = TAU + 1.0 / lam_mean
    sel = d >= tail0
    lam_tail = 1.0 / (d[sel].mean() - tail0) if sel.sum() > 50 else np.nan
    bb = float(((d >= TAU - 1.5 / TICK) & (d <= TAU + 1.5 / TICK)).mean())
    return lam_mean, lam_tail, 1 - lam_tail / lam_mean, bb, nev / dur


print("τ = %.4f µs, w = %.2f µs" % (TAU * 1e6, W * 1e6))
print("%-8s %-8s %9s %9s %9s %9s %9s %9s %9s" % (
    "机制", "Λ真值", "R_obs", "Λ_mean", "Λ_tail", "beta", "背靠背占比", "R/Λ真", "R/Λ_tail"))
for lam in (500, 1000, 2000, 5000, 10000, 20000, 40000):
    for mode in ("discard", "queue", "mix"):
        ft, fs, nev, nin, dur = simulate(lam, mode)
        lm, lt, beta, bb, robs = estimate(ft, nev, dur)
        print("%-8s %-8d %9.0f %9.0f %9.0f %9.4f %9.4f %9.4f %9.4f" % (
            mode, lam, robs, lm, lt, beta, bb, robs / (nin / dur), robs / lt))
    print()
