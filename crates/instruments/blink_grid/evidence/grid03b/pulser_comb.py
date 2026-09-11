"""判一次过境里的四重同戳簇是不是**片上标定脉冲**，并定出脉冲的精确周期与开关时刻。

背景：2022-03-19 那次"坏过境"先前被定案成"粒子主导的过境段"，依据是四路同戳、四路沉积
能量一致、边界很陡。**这三条对"贯穿粒子"和"标定脉冲"给出的预期完全相同**——它们把异常
与本底分开了，却分不开两种异常。**真正分得开的是周期性。**

三件事，缺一不可：

1. **梳齿**：四重簇的相邻间隔是否堆在 `k × P₀` 上。基频不能只看某一个齿——漏掉一个脉冲
   就落到 2 倍齿上，只数 2 倍齿会把占比低估一个数量级（先前把 8388/8389 当基频，那是
   二次谐波，只占 8.1%，基频 4194/4195 占 89.9%）。
2. **偶然期望要用实测的局部分布**，不能用"均匀间隔"。本例里齿两侧的邻格实测**是零**，
   局部本底给 841 倍，均匀假设给 208 倍，**差 4 倍**。
3. **相位**：用脉冲序号做线性回归定精确周期再算相位。**周期必须回归出来**——284 s 里有
   28 万个周期，周期差 1e-5 相位就转完一整圈，拿个约数去算相位会得到"无结构"的假阴性。

⚠️ **过境内的相对时刻一律从 GTI 取，不要解析文件名。** 实测文件名写 `2203190605_2203190618`
而 GTI 是 `06:14:28–06:19:12`，**差 9.5 分钟**；拿文件名当起点会把相对时刻错约 570 s。

用法：
    python3 pulser_comb.py <evt .fits> [--p0 4194.5] [--at <ISO 时刻>]
`--at` 给一个候选时刻，就额外报它附近的四重簇在不在梳齿上、相位多少——用来判"这个候选
是不是脉冲造的"。
"""

import argparse
import datetime as dt

import numpy as np
from astropy.io import fits

TICK = 2.0**22
ETH = 30.0
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
N_DET = 4


def iso_to_met(s):
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - REF).total_seconds() + (float("0." + frac) if frac else 0.0)


def load(path):
    """keep 后的 (时戳格, 探头, 道号, 能量, GTI)。与 `Event::keep` 同口径。"""
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        if np.any(np.diff(emin) <= 0):
            raise ValueError("E_MIN 非单调")
        ch_lo = int(np.flatnonzero(emin >= ETH)[0]) + 1
        T, D, P = [], [], []
        for i in range(N_DET):
            d = hd["EVENTS%d" % i].data
            pi = np.asarray(d["PI"])
            ty = np.asarray(d["EVT_TYPE"])
            k = (ty == 1) & (pi >= ch_lo) & (pi < nch)
            T.append(np.asarray(d["TIME"], dtype=np.float64)[k])
            P.append(pi[k].astype(np.int32))
            D.append(np.full(int(k.sum()), i, np.int8))
    t = np.concatenate(T)
    o = np.argsort(t, kind="stable")
    return (np.rint(t[o] * TICK).astype(np.int64), np.concatenate(D)[o],
            np.concatenate(P)[o], emin, gs, ge)


def quads(tk, pi, emin):
    """四重同戳簇的 (时戳, 簇平均沉积能量)。"""
    edge = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], edge + 1))
    en = np.concatenate((edge + 1, [tk.size]))
    sel = np.flatnonzero((en - st) == N_DET)
    e = np.array([emin[pi[st[i]:en[i]] - 1].mean() for i in sel])
    return tk[st[sel]], e, int(((en - st) >= 3).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits")
    ap.add_argument("--p0", type=float, default=4194.5, help="基频周期的初猜（tick）")
    ap.add_argument("--at", help="候选 ISO 时刻，判它是不是脉冲造的")
    ap.add_argument("--tol", type=float, default=4.0, help="判定落在梳齿上的容差（tick）")
    args = ap.parse_args()

    tk, det, pi, emin, gs, ge = load(args.fits)
    t4, e4, k3 = quads(tk, pi, emin)
    print("GTI %.3f–%.3f（%.1f s）= %s – %s UTC"
          % (gs, ge, ge - gs,
             (REF + dt.timedelta(seconds=gs)).strftime("%H:%M:%S"),
             (REF + dt.timedelta(seconds=ge)).strftime("%H:%M:%S")))
    print("keep 后 %d 个事例，≥3 重簇 %d 个，四重簇 %d 个，簇能量中位 %.1f keV"
          % (tk.size, k3, t4.size, np.median(e4) if e4.size else float("nan")))
    if t4.size < 10:
        print("四重簇太少，不判")
        return

    d4 = np.diff(t4)
    print()
    print("=== 1. 梳齿（基频不能只看一个齿）===")
    print("%4s %14s %10s %8s" % ("齿", "tick 区间", "对数", "占比"))
    for k in range(1, 5):
        c = args.p0 * k
        lo, hi = int(np.floor(c - 3)), int(np.ceil(c + 3))
        n = int(((d4 >= lo) & (d4 <= hi)).sum())
        print("%3dx %6d–%-7d %10d %7.2f%%" % (k, lo, hi, n, 100.0 * n / d4.size))
    ontooth = np.zeros(d4.size, bool)
    for k in range(1, 30):
        c = args.p0 * k
        ontooth |= (d4 >= c - 3) & (d4 <= c + 3)
    print("落在 1–29 倍齿上：%d / %d = **%.1f%%**" % (ontooth.sum(), d4.size, 100.0 * ontooth.mean()))

    print()
    print("=== 2. 偶然期望用实测的局部分布，不用均匀假设 ===")
    first = d4[(d4 >= args.p0 - 5) & (d4 <= args.p0 + 5)]
    if first.size:
        lo, hi = int(np.floor(args.p0 - 1)), int(np.ceil(args.p0 + 1))
        obs = int(((d4 >= lo) & (d4 <= hi)).sum())
        w = (d4 >= args.p0 - 100) & (d4 <= args.p0 + 100)
        nb = 201 - (hi - lo + 1)
        bkg = (int(w.sum()) - obs) / max(nb, 1)
        exp = bkg * (hi - lo + 1)
        print("   基频齿实测 %d 对；局部本底 %.2f 对/tick ⇒ 期望 %.1f" % (obs, bkg, exp))
        print("   **实测 / 期望 = %.0f 倍**" % (obs / exp if exp > 0 else float("inf")))
        print("   （均匀假设会给 %.1f，两者可差几倍——本底必须用实测分布）"
              % ((hi - lo + 1) / d4.mean() * d4.size))

    print()
    print("=== 3. 回归定精确周期，再算相位 ===")
    idx = np.concatenate(([0], np.cumsum(np.rint(d4 / args.p0)))).astype(np.int64)
    on = np.abs(d4 - np.rint(d4 / args.p0) * args.p0) < args.tol
    core = np.concatenate(([on[0]], on[:-1] | on[1:], [on[-1]]))[:t4.size]
    if core.sum() < 10:
        print("   梳齿上的簇太少，不回归")
        return
    A = np.vstack([idx[core], np.ones(int(core.sum()))]).T
    P, c0 = np.linalg.lstsq(A, t4[core].astype(float), rcond=None)[0]
    resid = t4[core] - (c0 + P * idx[core])
    print("   梳齿上的四重簇 %d / %d = **%.1f%%**" % (core.sum(), t4.size, 100.0 * core.mean()))
    print("   周期 P = %.4f tick = %.6f ms = **%.3f Hz**，残差 rms %.2f tick = %.0f ns"
          % (P, P / TICK * 1e3, TICK / P, resid.std(), resid.std() / TICK * 1e9))
    ph = np.mod(t4 - c0, P) / P
    ph = np.minimum(ph, 1 - ph)
    print("   |相位| < 0.001 的占 **%.1f%%**（随机应为 0.2%%）" % (100.0 * (ph < 0.001).mean()))
    print("   注：周期必须回归。%.0f s 里有 %.0f 个周期，周期差 1e-5 相位就转完一圈，"
          % (ge - gs, (ge - gs) * TICK / P))
    print("   拿约数去算相位会得到「无结构」的假阴性。")

    print()
    print("=== 4. 开关廓形（梳齿上的簇，按时间分箱）===")
    t_rel = t4 / TICK - gs
    h, edges = np.histogram(t_rel[core], bins=20, range=(0, ge - gs))
    for i in range(len(h)):
        print("   %6.1f–%-6.1f s : %6d %s"
              % (edges[i], edges[i + 1], h[i], "#" * min(int(h[i] / max(h.max(), 1) * 40), 40)))
    if core.any():
        print("   梳齿上最后一个簇在 %.3f s" % t_rel[core].max())

    if args.at:
        q = iso_to_met(args.at)
        print()
        print("=== 5. 候选 %s 是不是脉冲造的 ===" % args.at)
        print("   MET %.3f，**GTI 起点后 %.3f s**（相对时刻取自 GTI，不是文件名）" % (q, q - gs))
        m = np.abs(t_rel - (q - gs)) < 0.002
        print("   ±2 ms 内的四重簇 %d 个：" % m.sum())
        for i in np.flatnonzero(m):
            print("      %+8.3f ms  %6.1f keV  在梳齿上: %s  |相位| %.4f"
                  % ((t_rel[i] - (q - gs)) * 1e3, e4[i], "是" if core[i] else "否", ph[i]))
        if m.any() and core[m].all():
            print("   ⇒ **全在梳齿上：这个候选是脉冲造的。**")


if __name__ == "__main__":
    main()
