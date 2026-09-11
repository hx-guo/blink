"""GECAM-B：实测尾概率 ÷ 泊松尾概率——「标称误报率作废」的那个可量化的 N。

**分母必须逐段算。** 搜索用的 λ 是**局部的**（候选两侧的本底窗）；若把所有安静段
汇总成一个 λ 再比，**段与段之间的速率差就被算成过离散**，尾比被系统性高估，而且
**正是在大 `W`（候选窗那一档）上高估最多**。这里两种分母并排给，**以逐段那一版
为准**。

（同一个根也污染 Fano：速率非平稳同时伤二阶矩与尾巴。区别在于 Fano 只是诊断量，
而这个尾比是要写进论文的那个数。）

**段长 10 s 的依据**：搜索的本底窗是 1 s，所以 **1 s 那一档最接近搜索的真实口径、
10 s 已经是上界**；但段越短 λ 估得越噪，而 `sf` 对 λ 是凸的 ⇒ 期望被抬高、超出被
压低（Jensen）。**真值夹在两者之间，所以报曲线不报单点。**

用法: python3 gb_tailratio.py [目录=poisson25]
"""

import glob
import json
import sys

import numpy as np
from scipy import stats as st

BINS = (1.0, 10.0, 100.0, 1000.0)
KS = (4, 6, 8, 9, 10, 12)
SEG = 10.0


def load(directory):
    rows = []
    for p in sorted(glob.glob(f"{directory}/*.json")):
        rows += json.load(open(p))
    return rows


def ratios(rows, b, ks):
    nb_seg = int(SEG / (b * 1e-6))
    occ = np.sum([r[f"occ_pre_{b:g}"] for r in rows], 0).astype(float)
    occ_post = np.sum([r[f"occ_post_{b:g}"] for r in rows], 0).astype(float)
    lam_seg = np.array([r[f"lam_{b:g}"] for r in rows])
    lam_pool = float(lam_seg.mean())
    out = []
    for k in ks:
        obs = float(occ[k - 1:].sum())
        obs_post = float(occ_post[k - 1:].sum())
        p_pool = len(rows) * nb_seg * st.poisson.sf(k - 1, lam_pool)
        p_loc = float(nb_seg * st.poisson.sf(k - 1, lam_seg).sum())
        out.append((k, obs, obs_post, p_pool, p_loc))
    return out, lam_pool, lam_seg


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "poisson25"
    rows = load(directory)
    hours = len({r["hour"] for r in rows})
    print(f"段 {len(rows)} × {SEG:g} s = {SEG * len(rows):.0f} s（{hours} 小时）")
    rate = np.array([r["rate"] for r in rows])
    print(f"段间速率：中位 {np.median(rate):.0f} c/s，"
          f"5–95% {np.percentile(rate, 5):.0f}..{np.percentile(rate, 95):.0f}，"
          f"（max−min）/中位 = {(rate.max() - rate.min()) / np.median(rate):.2f}"
          f"  ← 这个散布就是汇总 λ 会被算成过离散的那一部分")
    print()
    head = (f"{'W':>8}{'k':>4}{'实测':>12}{'合并后':>10}{'泊松(汇总λ)':>14}"
            f"{'泊松(逐段λ)':>14}{'比(汇总)':>12}{'比(逐段)':>12}{'汇总/逐段':>10}")
    print(head)
    for b in BINS:
        out, lp, ls = ratios(rows, b, KS)
        for k, obs, obs_post, pp, pl in out:
            r1 = obs / pp if pp > 0 else float("inf")
            r2 = obs / pl if pl > 0 else float("inf")
            print(f"{b:>8g}{k:>4d}{obs:>12.0f}{obs_post:>10.0f}{pp:>14.4g}"
                  f"{pl:>14.4g}{r1:>12.4g}{r2:>12.4g}{r1 / max(r2, 1e-300):>10.1f}")
        print(f"   (λ 汇总 {lp:.4f}；逐段 5–95% {np.percentile(ls, 5):.4f}..{np.percentile(ls, 95):.4f})")

    print()
    print("=== 按 |磁纬| 三档（逐段 λ；B 星只到 38.5°，所以高磁纬那一档是下界）===")
    am = np.abs(np.array([r["mlat"] for r in rows]))
    q = np.percentile(am, [33.3, 66.7])
    for lab, msk in ((f"低 <{q[0]:.1f}°", am < q[0]),
                     (f"中 {q[0]:.1f}–{q[1]:.1f}°", (am >= q[0]) & (am < q[1])),
                     (f"高 >{q[1]:.1f}°", am >= q[1])):
        sub = [r for r, ok in zip(rows, msk) if ok]
        line = f"  {lab:<14} n={len(sub):>4}  "
        for b in (10.0, 100.0):
            out, _, _ = ratios(sub, b, (8,))
            k, obs, _, _, pl = out[0]
            line += f"W={b:g}µs k=8 比 {obs / pl if pl > 0 else float('inf'):>10.4g}   "
        print(line)


if __name__ == "__main__":
    main()
