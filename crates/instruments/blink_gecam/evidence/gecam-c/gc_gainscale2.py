"""GECAM-C：两个增益档是不是同一把尺子 —— 带偶然配对对照的版本。

`gc_gainscale.py` 第一轮拿到的是「未饱和对里高增益读数系统性偏低，低增益道号越高
偏得越多」。但那个数里混着**偶然配对**：同一路探头在死时间窗内本来就会偶尔挤进两个
互不相干的事例，一高一低就被当成一对。任何配对率都必须并排给出偶然期望，否则读不出
意义——这里的对照是**把低增益支路的时戳整体平移一个远大于死时间的常数再配一次**，
配到的全是偶然的。

第二件事是把「两把尺子」和「高增益接近满量程时压缩」分开。两者都能造出「高增益读数
偏低」，但预言不同：

* 两把尺子 → 偏差在**道号**上处处存在，与该路探头自己的满量程道无关；
* 接近满量程压缩 → 偏差只在 `PI_hi / edge_hi` 接近 1 时出现，**逐路的拐点随各自的
  `edge_hi` 走**。C 星 12 路的 `edge_hi` 实测 161–201（1.25 倍），杠杆不大但够用。

`edge_hi` 逐路从数据自己的直方图求（最后一个 ≥10 计数的道），不查表。

用法: gc_gainscale2.py <YYYY-MM-DD> <hour>
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_COUNTS = 10
NULL_SHIFT_SECONDS = 0.37  # 远大于死时间，又远小于轨道尺度


def pair_by_dead_time(time, gain, dead_s):
    """与 Rust 侧 `dedupe_gain_pairs` 同语义：同探头 + 死时间窗内 + 增益一高一低。"""
    n = time.size
    taken = np.zeros(n, dtype=bool)
    hi, lo, dts = [], [], []
    for a in range(n):
        if taken[a]:
            continue
        deadline = time[a] + dead_s[a]
        b = a + 1
        while b < n and time[b] <= deadline:
            if not taken[b] and gain[b] != gain[a]:
                taken[a] = taken[b] = True
                if gain[a] == 0:
                    hi.append(a)
                    lo.append(b)
                else:
                    hi.append(b)
                    lo.append(a)
                dts.append(abs(time[b] - time[a]))
                break
            b += 1
    return np.array(hi, int), np.array(lo, int), np.array(dts, float)


def edge_of(pi, mask):
    if mask.sum() == 0:
        return -1
    hist = np.bincount(np.clip(pi[mask], 0, 511), minlength=512)
    nz = np.flatnonzero(hist >= MIN_COUNTS)
    return int(nz[-1]) if nz.size else -1


def main():
    day, hour = sys.argv[1], int(sys.argv[2])
    path = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits"))[-1]
    print("文件", path)

    rows = []
    dead_values = set()
    n_real = n_null = 0
    all_dt = []
    with fits.open(path, memmap=True) as hdus:
        eb = hdus["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], float)
        emax = np.asarray(eb["E_MAX"], float)
        centre = np.sqrt(emin[:448] * emax[:448])
        print(f"EBOUNDS 行数 {emin.size}；溢出段各行 E_MIN（= 该支路满量程能量）:")
        print("  " + " ".join(f"{v:.0f}" for v in emin[448:]))

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
            dead = np.asarray(data["DEAD_TIME"])[order].astype(float)
            dead_values.update(np.unique(dead).tolist())
            dead_s = dead * 1e-6

            edge_hi = edge_of(pi, (evt == 1) & (gain == 0))
            edge_lo = edge_of(pi, (evt == 1) & (gain == 1))

            hi, lo, dt = pair_by_dead_time(t, gain, dead_s)
            n_real += hi.size
            all_dt.append(dt)

            # 偶然对照：低增益支路整体平移，重排后再配一次
            shifted = t.copy()
            shifted[gain == 1] += NULL_SHIFT_SECONDS
            o2 = np.argsort(shifted, kind="stable")
            nhi, nlo, _ = pair_by_dead_time(shifted[o2], gain[o2], dead_s[o2])
            n_null += nhi.size

            rows.append(
                dict(
                    name=hdu.name,
                    edge_hi=edge_hi,
                    edge_lo=edge_lo,
                    n_pair=hi.size,
                    n_null=nhi.size,
                    hi_pi=pi[hi],
                    lo_pi=lo_pi_of(pi, lo),
                    hi_evt=evt[hi],
                )
            )

    dt = np.concatenate(all_dt)
    print(f"\nDEAD_TIME 取值 {sorted(dead_values)}（列单位未标；按 µs 解释）")
    print(f"配到的对 {n_real}，偶然对照 {n_null}（占 {n_null/n_real*100:.2f}%）")
    print(
        "对内 |dt| 分位 (ns) 50/90/99/100 = "
        + " / ".join(f"{np.percentile(dt, p)*1e9:.1f}" for p in (50, 90, 99, 100))
        + f"；dt == 0 占 {(dt == 0).mean()*100:.2f}%"
    )

    print("\n逐路：满量程道与配对数")
    print("探头      edge_hi  能量(keV)   edge_lo  能量(keV)    对数    偶然")
    for r in rows:
        eh, el = r["edge_hi"], r["edge_lo"]
        print(
            f"{r['name']}   {eh:5d}  {centre[eh] if 0<=eh<448 else float('nan'):8.1f}   "
            f"{el:5d}  {centre[el] if 0<=el<448 else float('nan'):8.1f}  "
            f"{r['n_pair']:8d}  {r['n_null']:6d}"
        )

    # 只看高增益未溢出的对：偏差 vs 低增益道；再 vs 归一化的 PI_hi/edge_hi
    print("\n=== 未溢出对（高增益 EVT_TYPE == 1）：偏差随能量怎么走 ===")
    print("低增益道段    n       高−低 中位   |  同段按 PI_hi/edge_hi 分：")
    bands = [(100, 130), (130, 160), (160, 190), (190, 220), (220, 260), (260, 320)]
    for a, b in bands:
        n = 0
        diffs = []
        for r in rows:
            sel = (r["hi_evt"] == 1) & (r["lo_pi"] >= a) & (r["lo_pi"] < b)
            if sel.sum() == 0:
                continue
            n += int(sel.sum())
            diffs.append((r["hi_pi"][sel] - r["lo_pi"][sel]).astype(float))
        if n < 50:
            continue
        d = np.concatenate(diffs)
        print(f"  [{a:3d},{b:3d})  {n:8d}   {np.median(d):8.1f}")

    print("\n=== 压缩假说的判别：偏差 vs PI_hi / edge_hi（逐路归一化）===")
    print("  x = PI_hi/edge_hi 分箱      n        高−低 中位   低增益道中位")
    xs, ds, ls = [], [], []
    for r in rows:
        if r["edge_hi"] <= 0:
            continue
        sel = r["hi_evt"] == 1
        xs.append(r["hi_pi"][sel] / r["edge_hi"])
        ds.append((r["hi_pi"][sel] - r["lo_pi"][sel]).astype(float))
        ls.append(r["lo_pi"][sel].astype(float))
    x = np.concatenate(xs)
    d = np.concatenate(ds)
    l = np.concatenate(ls)
    for a, b in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
        sel = (x >= a) & (x < b)
        if sel.sum() < 50:
            continue
        print(
            f"  [{a:.2f},{b:.2f})  {int(sel.sum()):9d}   {np.median(d[sel]):8.1f}   "
            f"{np.median(l[sel]):8.0f}"
        )

    # 两把尺子的判别量：把同一个低增益道段内的偏差按探头拆开，看它是否随 edge_hi 走
    print("\n=== 同一低增益道段 [160,190) 内，逐路偏差 vs 该路 edge_hi ===")
    print("探头      edge_hi     n     高−低 中位")
    pts = []
    for r in rows:
        sel = (r["hi_evt"] == 1) & (r["lo_pi"] >= 160) & (r["lo_pi"] < 190)
        if sel.sum() < 30 or r["edge_hi"] <= 0:
            continue
        med = float(np.median(r["hi_pi"][sel] - r["lo_pi"][sel]))
        pts.append((r["edge_hi"], med))
        print(f"{r['name']}   {r['edge_hi']:5d}  {int(sel.sum()):7d}   {med:8.1f}")
    if len(pts) >= 4:
        e = np.array([p[0] for p in pts], float)
        m = np.array([p[1] for p in pts], float)
        rho = np.corrcoef(e, m)[0, 1]
        print(f"  edge_hi 与偏差的相关系数 r = {rho:+.3f}（n={len(pts)} 路）")
        print("  压缩假说预言 r 显著为正（edge 越高、同一道处离满量程越远、偏差越小）；")
        print("  两把尺子预言 r ≈ 0。")


def lo_pi_of(pi, lo):
    return pi[lo]


if __name__ == "__main__":
    main()
