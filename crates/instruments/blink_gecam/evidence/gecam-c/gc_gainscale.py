"""GECAM-C：两个增益档的 PI 是不是同一把尺子。

OPEN-QUESTIONS 第 17 条的未决项。先前记的是「两档共用同一套 PI 刻度，所以道号可以
直接比」，但实测双增益对里 PI 相同的占 0%。那个 0% 有两种可能：全是高增益饱和造成的
（饱和条一律顶到满量程道），或者两档本来就是两把尺子。

**这件事的后果不是计数而是能量**：去重保留低增益那条，如果两档不是同一把尺子，
`MIN_CHANNEL = 54` 在两档里就是两个能量，能阈随事例走哪条支路而变。

配对规则与 Rust 侧 `dedupe_gain_pairs` 一致：**同探头 + 死时间窗内 + 增益一高一低**，
不是「时戳完全相等」（实测有一批差 89/119 ns）。贪心从早到晚扫，每条最多配一次。

用法：gc_gainscale.py <YYYY-MM-DD> <hour>
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"


def hour_file(day, hour):
    files = sorted(
        glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits")
    )
    return files[-1] if files else None


def pair_by_dead_time(time, gain, dead_time_s):
    """返回 (高增益下标, 低增益下标, dt) 三个数组。贪心，与 Rust 侧同语义。"""
    n = time.size
    state = np.zeros(n, dtype=np.int8)  # 0 未配 1 已配
    hi, lo, dts = [], [], []
    for a in range(n):
        if state[a]:
            continue
        deadline = time[a] + dead_time_s[a]
        b = a + 1
        while b < n and time[b] <= deadline:
            if not state[b] and gain[b] != gain[a]:
                state[a] = state[b] = 1
                if gain[a] == 0:
                    hi.append(a)
                    lo.append(b)
                else:
                    hi.append(b)
                    lo.append(a)
                dts.append(time[b] - time[a])
                break
            b += 1
    return np.array(hi, int), np.array(lo, int), np.array(dts, float)


def main():
    day, hour = sys.argv[1], int(sys.argv[2])
    path = hour_file(day, hour)
    print("文件", path)

    with fits.open(path, memmap=True) as hdus:
        eb = hdus["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], float)
        emax = np.asarray(eb["E_MAX"], float)
        centre = np.sqrt(emin[:448] * emax[:448])

        all_hi_pi, all_lo_pi, all_dt = [], [], []
        gain_of_kept = np.zeros(2, dtype=np.int64)
        n_raw = n_pair = 0
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
            dead = np.asarray(data["DEAD_TIME"])[order].astype(float) * 1e-6

            n_raw += t.size
            # keep 之后的增益构成——绝大多数事例根本不成对，走哪条支路决定它的能量刻度
            keep = (evt == 1) & (pi >= 54) & (pi < 448)
            gain_of_kept += np.bincount(gain[keep], minlength=2)

            hi, lo, dt = pair_by_dead_time(t, gain, dead)
            n_pair += hi.size
            all_hi_pi.append(pi[hi])
            all_lo_pi.append(pi[lo])
            all_dt.append(dt)

    hi_pi = np.concatenate(all_hi_pi)
    lo_pi = np.concatenate(all_lo_pi)
    dt = np.concatenate(all_dt)

    print(f"\n原始事例 {n_raw}，双增益对 {n_pair}（占 {2*n_pair/n_raw*100:.3f}%）")
    print(
        "对内 dt 分位 (ns) 0/50/90/99/100 = "
        + " / ".join(f"{np.percentile(dt, p)*1e9:.1f}" for p in (0, 50, 90, 99, 100))
    )
    print(f"dt == 0 的占 {(dt == 0).mean()*100:.2f}%")

    kept_total = gain_of_kept.sum()
    print(
        f"\nkeep 之后的增益构成：高 {gain_of_kept[0]} ({gain_of_kept[0]/kept_total*100:.2f}%)，"
        f"低 {gain_of_kept[1]} ({gain_of_kept[1]/kept_total*100:.2f}%)"
    )

    # 高增益饱和的先剔掉——那批 PI 被顶到满量程道，比的是饱和不是刻度
    sat = hi_pi >= 448
    print(f"\n对里高增益条 PI >= 448（溢出段）占 {sat.mean()*100:.2f}%")
    for tag, mask in (("全部对", np.ones_like(sat)), ("高增益未溢出", ~sat)):
        if mask.sum() == 0:
            continue
        h, l = hi_pi[mask], lo_pi[mask]
        same = (h == l).mean() * 100
        diff = h.astype(int) - l.astype(int)
        print(
            f"\n--- {tag}（n={mask.sum()}）：PI 相同占 {same:.2f}%，"
            f"高−低 分位 5/25/50/75/95 = "
            + " / ".join(f"{np.percentile(diff, p):.0f}" for p in (5, 25, 50, 75, 95))
        )
        # 按低增益道分段看差值走向——同一把尺子的话差值该处处为零
        print("   低增益道段    n      高−低 中位   高增益能量/低增益能量 中位")
        edges = [54, 100, 150, 200, 250, 300, 350, 379, 448]
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (l >= a) & (l < b)
            if sel.sum() < 20:
                continue
            hh, ll = h[sel], l[sel]
            ok = (hh < 448) & (ll < 448)
            ratio = (
                np.median(centre[hh[ok]] / centre[ll[ok]]) if ok.sum() > 20 else float("nan")
            )
            print(
                f"   [{a:3d},{b:3d})  {sel.sum():7d}   {np.median(hh.astype(int)-ll.astype(int)):7.0f}"
                f"        {ratio:.4f}"
            )

    # 能阈的后果：ch54 在两档里各是什么能量。若两档是同一把尺子，两个数必然相等。
    unsat = ~sat
    near = unsat & (lo_pi >= 50) & (lo_pi < 70)
    if near.sum() > 50:
        print(
            f"\n低增益 ch50–70 段（n={near.sum()}）对应的高增益道中位 "
            f"{np.median(hi_pi[near]):.0f}，即 {centre[int(np.median(hi_pi[near]))]:.1f} keV；"
            f"低增益 ch54 名义 {centre[54]:.1f} keV"
        )


if __name__ == "__main__":
    main()
