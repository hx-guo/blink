"""把 `ga_jitter.py` 出的逐候选表折成"合并后还剩几个候选"。

合并改的不只是窗内计数，本底率也同步降，所以 `fa` 必须重算，不能只看计数。
重算用反标定的试验数：`T = fa / sf`，`sf = P(Poisson(mean) >= count)`。
**泊松上尾是正则化下不完全 Γ**：`P(X >= k) = gammainc(k, λ)`；`gammaincc` 是
`P(X <= k-1)`，方向反了会让 `T` 散到十几个数量级（第一版就栽在这里）。
T 的散布本身是这套近似可信度的自检，四分位差应在 ±20% 量级。

用法: python3 ga_merge_analyse.py <jit_*_cand.csv> <jit_*_report.json> [<recut_*.csv>]
"""

import csv
import json
import sys

import numpy as np
from scipy.special import gammainc

TAUS = [0, 29, 59, 100, 150, 200, 300, 500, 1000]
MIN_NUMBER = 8


def main():
    cand_path, rep_path = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(cand_path)))
    rep = json.load(open(rep_path))
    recut = {}
    if len(sys.argv) > 3:
        recut = {r["start"]: r for r in csv.DictReader(open(sys.argv[3]))}

    hour = np.array([r["hour"] for r in rows])
    fa = np.array([float(r["fa"]) for r in rows])
    cnt = np.array([int(r["count"]) for r in rows], float)
    mean = np.array([float(r["mean"]) for r in rows])
    bin_us = np.array([float(r["bin_us"]) for r in rows])
    ncore = np.array([int(r["n_core"]) for r in rows], float)
    nt = {t: np.array([int(r[f"n_t{t}"]) for r in rows], float) for t in TAUS}

    hit = float((ncore == cnt).mean())
    print(f"对账 n_core == count: {hit*100:.2f}%（不到 100% 就别信下面任何一个数）")

    T = fa / gammainc(cnt, mean)
    Tm = float(np.median(T))
    print(f"反标定试验数 T 中位 {Tm:.4g}，四分位 {np.percentile(T,25):.4g} .. {np.percentile(T,75):.4g}")

    exposure = sum(h.get("searched_seconds", 0) for h in rep.get("exposure", [])) or None
    print(f"\n原始候选 {len(rows)}")
    print(f"{'τ(ns)':>7} {'全局并掉':>9} {'n>=8':>7} {'fa<=20':>8} {'fa<=1':>7} {'fa<=1e-2':>9}")
    for t in TAUS:
        k = TAUS.index(t)
        sh = np.array([1 - rep["hours"][h]["clusters"][k]["merged_frac"] for h in hour])
        fa2 = gammainc(np.maximum(nt[t], 1), mean * sh) * Tm
        ok = nt[t] >= MIN_NUMBER
        print(f"{t:7d} {(1-sh.mean())*100:8.2f}% {int(ok.sum()):7d} "
              f"{int((ok&(fa2<=20)).sum()):8d} {int((ok&(fa2<=1)).sum()):7d} "
              f"{int((ok&(fa2<=1e-2)).sum()):9d}")

    k = TAUS.index(150)
    sh = np.array([1 - rep["hours"][h]["clusters"][k]["merged_frac"] for h in hour])
    fa2 = gammainc(np.maximum(nt[150], 1), mean * sh) * Tm
    surv = (nt[150] >= MIN_NUMBER) & (fa2 <= 20)
    print(f"\nτ=150 ns 的幸存者（{int(surv.sum())} 个）")
    print(f"{'start':26} {'bin_us':>9} {'cnt':>4} {'n150':>5} {'fa原':>10} {'fa合':>10} "
          f"{'mf':>6} {'超量程':>7}")
    for i in np.where(surv)[0]:
        r = rows[i]
        rc = recut.get(r["start"], {})
        orf = (1 - int(rc["n_cut_adapt"]) / int(rc["n_core"])) * 100 if rc else float("nan")
        print(f"{r['start']:26} {bin_us[i]:9.2f} {int(cnt[i]):4d} {int(nt[150][i]):5d} "
              f"{fa[i]:10.3g} {fa2[i]:10.3g} {float(rc.get('multiplet_frac','nan')):6.3f} {orf:6.1f}%")

    print("\n同戳簇相对偶然撞车的富集（逐小时）")
    for h in sorted(rep["hours"]):
        hh = rep["hours"][h]
        n = hh["n_after_dedupe"]
        span = 3600.0          # 粗口径：用文件跨度，只为给富集倍数一个量级
        r = n / span
        for t in (0, 150):
            kk = TAUS.index(t)
            tau = t * 1e-9 if t else hh["quantum_ns"] * 1e-9
            chance = 1 - np.exp(-r * tau)
            obs = hh["clusters"][kk]["merged_frac"]
            print(f"  [{h}] τ={t:4d}ns 偶然 {chance*100:6.3f}% 实测 {obs*100:6.2f}% "
                  f"富集 {obs/chance:6.1f}x", end="")
        print()


if __name__ == "__main__":
    main()
