"""把三重同戳数在**候选窗之外**：用邻域里的粒子穿越个数当判别量。

**为什么要换窗。** 窗内 f₃ 有两个毛病，都是窗长造成的：

  1. **结构上限**：四路探头、同探头不重格 ⇒ 簇最多 4 重，`min_number = 8` ⇒ 单簇候选
     f₃ ≤ 0.5，固定阈对「一次穿越」恒不触发（见 `f3_pool.py`）。
  2. **短窗里偶然期望爆掉**：λ₃ ∝ n³/T²。实测最极端的一例（2022-03-19，8 个事例挤在
     8.3 µs = 36 格，四路各 2 个），窗内偶然出现一个三重簇的概率**高达 5.5%**
     （含 20 tick 死时间约束；不含约束是 2.4%——**死时间约束把偶然抬高而不是压低**，
     因为它逼着每一路"一早一晚"，反而更容易对齐）。所以任何基于窗内 Poisson 检验的
     判据在那里都判不动。

**邻域窗没有这两个毛病。** 穿星粒子不挑时刻，本底里本来就一直在发生，率是实测的；
一个候选的邻域里出现多少次三重同戳，期望就是 r₃ᵇᵏᵍ × 窗宽，**与候选自己的亮度和窗长
无关**。真 TGF 的邻域里不该多出粒子穿越，除非它恰好赶上。

判别量：k₃(±X) = 候选前后 ±X 内（含窗内）的 ≥3 重同戳簇个数，
对照期望 λ = r₃ᵇᵏᵍ × 2X（r₃ᵇᵏᵍ 由该候选自己的过境实测，见 `f3_background.py`）。

用法：
    python3 f3_wide.py <burst_events 目录> --bkg f3_background.csv [-o out.csv]
"""

import argparse
import csv
import math
import os

import numpy as np

Q_US = (2.0**-22) * 1e6
TRIPLE = 3
WIDTHS_US = (500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def pois_ge(k, lam):
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    s = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, 1.0 - s)


def clusters(tk):
    if tk.size == 0:
        return 0
    edge = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], edge + 1))
    en = np.concatenate((edge + 1, [tk.size]))
    return int(((en - st) >= TRIPLE).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("burst_dir")
    ap.add_argument("--bkg", required=True, help="f3_background.py 的产物，取逐候选的本底三重簇率")
    ap.add_argument("-o", "--out", default="f3_wide.csv")
    args = ap.parse_args()

    r3 = {r["start"]: float(r["far_r3_per_s"]) for r in csv.DictReader(open(args.bkg))}
    index = {r["csv"]: r for r in csv.DictReader(open(os.path.join(args.burst_dir, "index.csv")))}
    rows = []
    for name in sorted(index):
        meta = index[name]
        path = os.path.join(args.burst_dir, name)
        if not os.path.exists(path):
            continue
        recs = list(csv.DictReader(open(path)))
        t = np.array([float(r["t_us_rel_best_start"]) for r in recs])
        tk = np.rint(t / Q_US).astype(np.int64)
        o = np.argsort(tk)
        tk = tk[o]
        t = t[o]
        w = float(meta["bin_size_best_us"])
        rate = r3.get(meta["start"][:23], float("nan"))
        rec = dict(start=meta["start"][:23], lightning=meta["lightning"],
                   n=meta["count"], W_us="%.2f" % w, r3_per_s="%.3f" % rate)
        for x in WIDTHS_US:
            sel = (t >= -x) & (t <= w + x)
            k = clusters(tk[sel])
            lam = rate * (2 * x + w) * 1e-6
            rec["k3_pm%g" % x] = k
            rec["p_pm%g" % x] = "%.3e" % pois_ge(k, lam)
            rec["lam_pm%g" % x] = "%.3e" % lam
        rows.append(rec)

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    lit = np.array([r["lightning"] == "1" for r in rows])
    print("38 个显著候选，本底三重簇率中位 %.2f 个/s" % np.median([float(r["r3_per_s"]) for r in rows]))
    print()
    print("%-10s %8s %10s %10s %10s %10s" % ("邻域 ±X", "λ 期望", "7 认证 k₃", "其余 31 k₃", "认证 p 最小", "其余 p<1e-3"))
    for x in WIDTHS_US:
        k = np.array([r["k3_pm%g" % x] for r in rows])
        p = np.array([float(r["p_pm%g" % x]) for r in rows])
        lam = np.median([float(r["lam_pm%g" % x]) for r in rows])
        print("%-10.0fµs %8.4f %10s %10s %10.2e %10d"
              % (x, lam,
                 "中位 %d 最大 %d" % (np.median(k[lit]), k[lit].max()),
                 "中位 %d 最大 %d" % (np.median(k[~lit]), k[~lit].max()),
                 p[lit].min(), int((p[~lit] < 1e-3).sum())))
    print()
    print("逐候选（邻域 ±1 ms）：")
    print("%-22s %3s %9s %5s %10s %6s" % ("start", "n", "W µs", "k₃", "p", "闪电"))
    for r in sorted(rows, key=lambda z: float(z["p_pm1000"])):
        print("%-22s %3s %9s %5d %10s %6s"
              % (r["start"][:19], r["n"], r["W_us"], r["k3_pm1000"], r["p_pm1000"],
                 "★" if r["lightning"] == "1" else ""))


if __name__ == "__main__":
    main()
