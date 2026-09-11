"""汇总 `gb_caliber2.py` 的四套并排：{新去重, 旧去重} × {同戳 tol=0, 容差 tol=150 ns}。

**两个簇口径不是同一个量，比较前必须对齐**：约 100 ns 的支路延迟落在 `tol = 0`
之外、`tol = 150 ns` 之内，所以换去重口径的影响在两种簇上完全不同。A 星的
`f₃` 是严格同戳（τ = 0），要跟它对的是 `{新, tol=0}` 那一列。

用法: python3 gb_cal_agg.py [目录=caliber2]
"""

import glob
import sys

import numpy as np


def load(directory):
    acc, ev = {}, [0, 0, 0]
    files = sorted(glob.glob(f"{directory}/*.npz"))
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith("n_event"):
                continue
            acc.setdefault(k, []).append(z[k])
        ev[0] += int(z["n_event_new"][0])
        ev[1] += int(z["n_event_old"][0])
        ev[2] += int(z["n_event_raw"][0])
    return {k: np.concatenate(v) for k, v in acc.items()}, ev, len(files)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "caliber2"
    D, ev, nfile = load(directory)
    n = D["t0"].size
    print(f"小时 {nfile}，候选 {n}")
    print(f"整小时准入后：新 {ev[0]} / 旧 {ev[1]}  →  旧口径多留 "
          f"{(ev[1] - ev[0]) / ev[0] * 100:.3f}%")
    m = D["n_new"] > 0
    print(f"逐候选窗内：均值多留 "
          f"{np.mean((D['n_old'][m] - D['n_new'][m]) / np.maximum(D['n_new'][m], 1)) * 100:.3f}%，"
          f"至少多留一条的候选占 {(D['n_old'] > D['n_new']).mean() * 100:.2f}%")
    print()
    print("=== 四套并排（均值） ===")
    print(f"{'量':<8}{'新·同戳':>12}{'旧·同戳':>12}{'旧/新':>9}"
          f"{'新·τ150':>12}{'旧·τ150':>12}{'旧/新':>9}")
    for k in ("mf", "f2", "f3"):
        a = np.nanmean(D[f"{k}_new"][m])
        b = np.nanmean(D[f"{k}_old"][m])
        c = np.nanmean(D[f"{k}t_new"][m])
        d = np.nanmean(D[f"{k}t_old"][m])
        print(f"{k:<8}{a:>12.4f}{b:>12.4f}{b / max(a, 1e-12):>9.4f}"
              f"{c:>12.4f}{d:>12.4f}{d / max(c, 1e-12):>9.4f}")
    for k, lab in (("f3pos", "f3>0 占比"), ):
        a, b = D[f"{k}_new"][m].mean(), D[f"{k}_old"][m].mean()
        c, d = D[f"{k}t_new"][m].mean(), D[f"{k}t_old"][m].mean()
        print(f"{lab:<8}{a * 100:>11.2f}%{b * 100:>11.2f}%{b / max(a, 1e-12):>9.4f}"
              f"{c * 100:>11.2f}%{d * 100:>11.2f}%{d / max(c, 1e-12):>9.4f}")
    print()
    print("=== `f₃ == 0` 留下多少（A 星报的是这一面：老口径 2.4% / 新口径 39%，16 倍）===")
    for suf, lab in (("", "同戳 tol=0"), ("t", "容差 tol=150 ns")):
        a = 1 - D[f"f3pos{suf}_new"][m].mean()
        b = 1 - D[f"f3pos{suf}_old"][m].mean()
        print(f"  {lab:<16}：新口径留 {a * 100:6.2f}%   旧口径留 {b * 100:6.2f}%   "
              f"新/旧 = {a / max(b, 1e-12):.3f} 倍")
    print()
    print("=== 同戳 vs 容差：换簇口径本身的影响（同一套去重下）===")
    for cal, lab in (("new", "新去重"), ("old", "旧去重")):
        a = np.nanmean(D[f"f3_{cal}"][m])
        c = np.nanmean(D[f"f3t_{cal}"][m])
        pa = D[f"f3pos_{cal}"][m].mean()
        pc = D[f"f3pos{'t'}_{cal}"][m].mean()
        print(f"  {lab}：f₃ 同戳 {a:.4f} → τ150 {c:.4f}（×{c / max(a, 1e-12):.3f}）；"
              f"f₃>0 {pa * 100:.2f}% → {pc * 100:.2f}%")
    print()
    print("=== 按窗宽分档（判别「偶然撞同戳」：(n−1)q/W 跨三个数量级而倍数恒定 ⇒ 不是偶然）===")
    bw = D["bin_s"] * 1e6
    edges = [0, 0.3, 1, 3, 10, 30, 100, 1e9]
    labs = ["<0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", ">=100"]
    print(f"{'bin µs':<8}{'n':>9}{'多留%':>8}"
          f"{'mf 旧/新':>10}{'f3 旧/新':>10}{'f3τ 旧/新':>11}{'(n-1)q/W':>11}")
    for i, lab in enumerate(labs):
        s = m & (bw >= edges[i]) & (bw < edges[i + 1])
        if s.sum() < 50:
            continue
        ex = np.mean((D["n_old"][s] - D["n_new"][s]) / np.maximum(D["n_new"][s], 1)) * 100
        r_mf = np.nanmean(D["mf_old"][s]) / max(np.nanmean(D["mf_new"][s]), 1e-12)
        r_f3 = np.nanmean(D["f3_old"][s]) / max(np.nanmean(D["f3_new"][s]), 1e-12)
        r_f3t = np.nanmean(D["f3t_old"][s]) / max(np.nanmean(D["f3t_new"][s]), 1e-12)
        acc = (np.median(D["n_new"][s]) - 1) * 14.9e-9 / (np.median(bw[s]) * 1e-6)
        print(f"{lab:<8}{s.sum():>9d}{ex:>8.2f}{r_mf:>10.4f}{r_f3:>10.4f}"
              f"{r_f3t:>11.4f}{acc:>11.4f}")


if __name__ == "__main__":
    main()
