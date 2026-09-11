"""GECAM-C：两档是不是同一把尺子 —— 用同时戳对做，判别量换成「脱离对角线的位置」。

`gc_gainscale2.py` 想用「同一低增益道段内、逐路偏差 vs 该路 edge_hi 的相关」来分开
「两把尺子」和「接近满量程压缩」，**那个检验没有功效**：r = +0.161（9 路），而且各路在
那个道段里的样本数从 45 到 25423 差三个量级，中位数根本不可比。换一个判别量。

判别量：**逐路的「脱离点」**——把 `PI_hi` 当 `PI_lo` 的函数，两档若是同一把尺子，
低处必然贴着对角线 `PI_hi = PI_lo`，到某个道号才向下弯（高增益支路见顶）。

* 同一把尺子 + 接近满量程压缩 → 脱离点**随该路自己的 `edge_hi` 走**，
  12 路的 `edge_hi` 实测 161–208，脱离点该跟着差 40 道。
* 两把尺子 → 低处就不贴对角线，脱离点与 `edge_hi` 无关。

只用 **dt == 0 的对**：偶然配对靠的是两个独立事例挤进死时间窗，时戳恰好逐位相等的
概率是 q/dead ≈ 15 ns / 4 µs 量级，可忽略；平移对照会把这一条量出来而不是假定。

用法: gc_gainscale3.py <YYYY-MM-DD> <hour>
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_COUNTS = 10
NULL_SHIFT_SECONDS = 0.37
DEPARTURE_CHANNELS = 5.0  # 中位落后对角线这么多道就算脱离


def same_stamp_pairs(time, gain, evt, pi):
    """同时戳且增益一高一低的对。同时戳天然只在同一路探头内（一个 EVENTS 表一路）。"""
    boundary = np.flatnonzero(np.diff(time) != 0) + 1
    groups = np.split(np.arange(time.size), boundary)
    hi, lo = [], []
    for g in groups:
        if g.size != 2:
            continue
        a, b = g
        if gain[a] == gain[b]:
            continue
        if gain[a] == 0:
            hi.append(a)
            lo.append(b)
        else:
            hi.append(b)
            lo.append(a)
    return np.array(hi, int), np.array(lo, int)


def edge_of(pi, mask):
    if mask.sum() == 0:
        return -1
    hist = np.bincount(np.clip(pi[mask], 0, 511), minlength=512)
    nz = np.flatnonzero(hist >= MIN_COUNTS)
    return int(nz[-1]) if nz.size else -1


def departure_point(lo_pi, hi_pi):
    """`PI_hi` 的中位第一次落后对角线 DEPARTURE_CHANNELS 道的那个 `PI_lo`。"""
    for c in range(100, 260):
        sel = (lo_pi >= c - 4) & (lo_pi <= c + 4)
        if sel.sum() < 100:
            continue
        if np.median(hi_pi[sel]) < c - DEPARTURE_CHANNELS:
            return c
    return -1


def main():
    day, hour = sys.argv[1], int(sys.argv[2])
    path = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits"))[-1]
    print("文件", path)

    rows = []
    n_null_same = 0
    with fits.open(path, memmap=True) as hdus:
        eb = hdus["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], float)
        emax = np.asarray(eb["E_MAX"], float)
        centre = np.sqrt(emin[:448] * emax[:448])

        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            t = np.asarray(data["TIME"], float)
            order = np.argsort(t, kind="stable")
            t = t[order]
            pi = np.asarray(data["PI"])[order].astype(int)
            gain = np.asarray(data["GAIN_TYPE"])[order].astype(int)
            evt = np.asarray(data["EVT_TYPE"])[order].astype(int)

            edge_hi = edge_of(pi, (evt == 1) & (gain == 0))
            hi, lo = same_stamp_pairs(t, gain, evt, pi)
            if hi.size == 0:
                continue
            # 偶然对照：同时戳口径下平移之后还能配到几对
            shifted = t.copy()
            shifted[gain == 1] += NULL_SHIFT_SECONDS
            o2 = np.argsort(shifted, kind="stable")
            nhi, _ = same_stamp_pairs(shifted[o2], gain[o2], evt[o2], pi[o2])
            n_null_same += nhi.size

            both = (evt[hi] == 1) & (evt[lo] == 1)
            rows.append(
                dict(
                    name=hdu.name,
                    edge_hi=edge_hi,
                    n_pair=hi.size,
                    n_both=int(both.sum()),
                    hi_pi=pi[hi][both],
                    lo_pi=pi[lo][both],
                )
            )

    n_pair = sum(r["n_pair"] for r in rows)
    n_both = sum(r["n_both"] for r in rows)
    print(f"\n同时戳对 {n_pair}，其中两条都是 EVT_TYPE == 1 的 {n_both}"
          f"（{n_both/n_pair*100:.2f}%）")
    print(f"平移对照在同时戳口径下配到 {n_null_same} 对"
          f"（{n_null_same/n_pair*100:.4f}%）——同时戳对里偶然的可忽略")

    print("\n逐路：脱离点 vs 满量程道")
    print("探头      edge_hi  未溢出对   脱离点  脱离点/edge_hi")
    pts = []
    for r in rows:
        if r["n_both"] < 500 or r["edge_hi"] <= 0:
            print(f"{r['name']}   {r['edge_hi']:5d}  {r['n_both']:8d}   样本不足")
            continue
        dep = departure_point(r["lo_pi"], r["hi_pi"])
        pts.append((r["edge_hi"], dep))
        print(
            f"{r['name']}   {r['edge_hi']:5d}  {r['n_both']:8d}   {dep:5d}   "
            f"{dep / r['edge_hi'] if dep > 0 else float('nan'):.3f}"
        )
    if len(pts) >= 4:
        e = np.array([p[0] for p in pts], float)
        d = np.array([p[1] for p in pts], float)
        ok = d > 0
        if ok.sum() >= 4:
            r = np.corrcoef(e[ok], d[ok])[0, 1]
            slope = np.polyfit(e[ok], d[ok], 1)[0]
            print(f"\n  edge_hi 与脱离点的相关 r = {r:+.3f}，斜率 {slope:+.2f} 道/道"
                  f"（n = {int(ok.sum())} 路）")
            print("  同一把尺子 + 见顶压缩预言 r ≈ +1、斜率 ≈ +1；两把尺子预言 r ≈ 0。")
            print(f"  脱离点/edge_hi 的极差 {np.ptp(d[ok]/e[ok]):.3f}"
                  f"（若真由 edge_hi 定，这个比该是常数）")

    print("\n=== 全体未溢出同时戳对：PI_hi 的中位随 PI_lo 走 ===")
    lo = np.concatenate([r["lo_pi"] for r in rows if r["n_both"] > 0])
    hi = np.concatenate([r["hi_pi"] for r in rows if r["n_both"] > 0])
    print("PI_lo   n        PI_hi 中位   差     能量比 E(hi)/E(lo)")
    for c in range(110, 250, 10):
        sel = (lo >= c) & (lo < c + 10)
        if sel.sum() < 50:
            continue
        m = float(np.median(hi[sel]))
        print(
            f"{c:5d}  {int(sel.sum()):8d}   {m:8.1f}  {m - (c + 5):6.1f}   "
            f"{centre[int(round(m))] / centre[c + 5]:.4f}"
        )


if __name__ == "__main__":
    main()
