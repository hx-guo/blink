"""把 occ_*.json 汇成两张表：随 k 的曲线（在 A 星自己的窗宽上）与随 W 的依赖。"""
import json
import sys

import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/gecam_a/occ_20240111_occ.json"
d = json.load(open(p))
KS = d["k_list"]
WS = [str(w) for w in d["widths_us"]]
H = d["hours"]
hours = sorted(H)
print(f"{d['day']}  小时 {hours}\n")

print("表一：在 A 星自己的窗宽 W = 0.149 µs（候选 bin 中位）上，实测/泊松随 k")
print(f"{'k':>4} " + " ".join(f"{'h'+h:>12}" for h in hours) + f" {'几何平均':>12}")
for i, k in enumerate(KS):
    vals = []
    for h in hours:
        v = H[h]["0.149"]["log10_ratio"][str(k)]
        vals.append(v)
    good = [v for v in vals if v is not None]
    gm = 10 ** np.mean(good) if good else float("nan")
    cells = " ".join(f"{(10**v if v is not None else float('nan')):12.3e}" for v in vals)
    mark = "  <- 候选 count 中位" if k == 9 else ""
    print(f"{k:4d} {cells} {gm:12.3e}{mark}")

print("\n表二：k = 9（候选 count 中位）处，实测/泊松随窗宽 W")
print(f"{'W (µs)':>10} {'λ (h01)':>10} " + " ".join(f"{'h'+h:>12}" for h in hours))
for w in WS:
    lam = H[hours[0]][w]["lambda"]
    cells = []
    for h in hours:
        v = H[h][w]["log10_ratio"]["9"]
        cells.append(f"{(10**v if v is not None else float('nan')):12.3e}")
    print(f"{float(w):10g} {lam:10.4g} " + " ".join(cells))

print("\n表三：同一件事换个看法——'窗里挤进 >= 8 个计数'的格数几乎不随 W 变，"
      "\n      而泊松预言的概率跨 16 个数量级。这就是成簇的直接证据。")
print(f"{'W (µs)':>10} {'λ':>10} {'实测 >=8 的格数':>16} {'泊松尾概率':>12}")
h = hours[0]
for w in WS:
    rec = H[h][w]
    lam = rec["lambda"]
    lr = rec["log10_ratio"]["8"]
    if lr is None:
        continue
    from scipy.stats import poisson
    logpoi = poisson.logsf(7, lam) / np.log(10)
    n_cells = 10 ** (lr + logpoi) * rec["ncells"]
    print(f"{float(w):10g} {lam:10.4g} {n_cells:16.0f} {10**logpoi:12.3e}")
