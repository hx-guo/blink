"""B 角的 3 ms 是本来就有的，还是读出把一个短亮暴摊开的？——正对照式检验。

## 检验对象

假说 H2："B 角其实是个 ~100 µs 的短亮暴，共帧读出的队列把它摊成了 3 ms。"
对照 H1："B 角本来就是 ~3 ms 的暴。"

两者对**背靠背帧的游程长度**给出差一个量级的预言：读出每个事例至少要 120 tick
（`dead(s) = 120 + 37(s−1)`，`frame_cost.py` 实测），所以 N 个计数挤在 100 µs 里
必然排成**一条长度约 N 的连续背靠背游程**；而本来就摊在 3 ms 上的同样 N 个计数
只会零星排队，游程长度是个位数。

**这就是新规则要的正对照**：把已知真值（H1 / H2 各自的合成暴）喂进同一条测量流程，
看它能不能把真值认回来，再拿实测值去读。

## 读出模型（全部来自 GRID-04 实测，不是假设）

- 帧成本 `dead(s) = 120 + 37(s−1)` tick，s = 本帧事例数（`frame_cost.py`，硬边沿）
- latch 窗 w：帧开头这段时间里其余各路的击中并进本帧、共用触发时戳（每路至多 1 个）
- 落在 (w, dead) 的击中：以概率 `p_queue` 进队列（各自占一帧、各自新时戳），其余丢弃

用法: python3 burst_sim.py
"""

import numpy as np

TICK = 2.0**22
BASE_COST, PER_EVENT = 120, 37
RNG = np.random.default_rng(4)


def readout(t, det, w_s, p_queue):
    """把入射流过一遍读出，返回 (帧时刻 tick, 每帧事例数)。"""
    n = t.size
    ftime, fsize = [], []
    queue = 0
    i, cur_end = 0, -1e18
    while True:
        if queue > 0:
            start = cur_end
            queue -= 1
        else:
            while i < n and t[i] < cur_end:
                i += 1
            if i >= n:
                break
            start = t[i]
            i += 1
        size = 1
        joined = set()
        while i < n and t[i] < start + w_s:
            if det[i] not in joined and len(joined) < 3:
                joined.add(det[i])
                size += 1
            i += 1
        cost = (BASE_COST + PER_EVENT * (size - 1)) / TICK
        spill = 0
        while i < n and t[i] < start + cost:
            spill += 1
            i += 1
        queue += int(RNG.binomial(spill, p_queue))
        ftime.append(start)
        fsize.append(size)
        cur_end = start + cost
    return np.asarray(ftime), np.asarray(fsize)


def run_stats(ftime, fsize):
    """背靠背帧占比与游程长度。帧间隔等于本帧成本（±2 tick）即算背靠背。"""
    if ftime.size < 3:
        return 0.0, 0, 0.0
    d = np.rint(np.diff(ftime) * TICK).astype(np.int64)
    floor = BASE_COST + PER_EVENT * (fsize[:-1] - 1)
    bb = d <= floor + 2
    rl, cur = [], 0
    for v in bb:
        if v:
            cur += 1
        elif cur:
            rl.append(cur)
            cur = 0
    if cur:
        rl.append(cur)
    return float(bb.mean()), (max(rl) if rl else 0), (float(np.mean(rl)) if rl else 0.0)


def trial(n_inc, dur_burst, bkg_cps, w_s, p_queue, span=0.006):
    """一次试验：span 秒的窗口，中间放一个 n_inc 个光子、时长 dur_burst 的暴。"""
    nb = RNG.poisson(bkg_cps * span)
    tb = RNG.random(nb) * span
    t0 = span / 2 - dur_burst / 2
    ts = t0 + RNG.random(n_inc) * dur_burst
    t = np.concatenate([tb, ts])
    det = RNG.integers(0, 4, t.size)
    o = np.argsort(t)
    ft, fs = readout(t[o], det[o], w_s, p_queue)
    return run_stats(ft, fs), int(fs.sum())


print("读出：dead(s) = %d + %d(s−1) tick；实测 GRID-04" % (BASE_COST, PER_EVENT))
print("实测 B 角（GRID-04 的 21 个显著候选，±3 ms 窗）："
      "背靠背占比中位 0.18（0.023–0.293），**最长游程 1–5**，本底窗 0.008")
for w_us, pq in ((3.8, 0.19), (3.8, 1.0), (20.0, 0.19), (20.0, 1.0)):
    print("\n=== w = %.1f µs, 排队份额 = %.2f ===" % (w_us, pq))
    print("%-28s %10s %10s %10s %10s" % ("假说", "输出计数", "背靠背占比", "最长游程", "游程均值"))
    for tag, n_inc, dur in (("H1 本来就 3 ms（47 个）", 60, 3.0e-3),
                            ("H1 本来就 3 ms（120 个）", 150, 3.0e-3),
                            ("H2 其实 100 µs（要出 47 个）", 60, 100e-6),
                            ("H2 其实 100 µs（入射 200 个）", 200, 100e-6),
                            ("H2 其实 300 µs（入射 100 个）", 100, 300e-6)):
        res = [trial(n_inc, dur, 1270.0, w_us * 1e-6, pq) for _ in range(200)]
        nout = np.median([r[1] for r in res])
        bb = np.median([r[0][0] for r in res])
        mx = np.median([r[0][1] for r in res])
        av = np.median([r[0][2] for r in res])
        print("%-28s %10.0f %10.4f %10.0f %10.2f" % (tag, nout, bb, mx, av))
