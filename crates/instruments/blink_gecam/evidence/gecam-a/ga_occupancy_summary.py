"""把 occ_*.json 汇成三张表：随 k、随 W，以及"漂移污染的量级"（两套期望之比）。

λ 有两套：整小时一个（会把一小时之内的速率漂移算成过离散）和逐 10 s 段一个
（对搜索实际生效的那个，搜索用的本底窗是候选两侧各半秒）。**逐段那一版才是
结论**；整小时那一版留着，两者之比就是"漂移被算成过离散"的量级，是方法论量
不是物理量。
"""
import json
import sys

import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/gecam_a/occ_20240111_occ.json"
d = json.load(open(p))
KS = d["k_list"]
WS = [str(w) for w in d["widths_us"]]
H = d["hours"]
hours = sorted(H)
W0 = "0.149"      # A 星候选 bin 中位
K0 = "9"          # A 星候选 count 中位
print(f"{d['day']}  小时 {hours}  段长 {d['seg_seconds']:.0f} s\n")


def g(h, w, k, key):
    v = H[h][w][key][str(k)]
    return None if v is None else float(v)


def gm(vals):
    good = [v for v in vals if v is not None]
    return 10 ** np.mean(good) if good else float("nan")


print("表〇：一小时之内的速率漂移有多大（这是输入，不是结果）")
print(f"{'小时':>6} {'段数':>5} {'p5':>7} {'p50':>7} {'p95':>7} {'p95/p5':>7} {'max/min':>8}")
for h in hours:
    r = H[h]["rate_seg"]
    print(f"{h:>6} {r['n_seg']:5d} {r['p5']:7.0f} {r['p50']:7.0f} {r['p95']:7.0f} "
          f"{r['p95'] / r['p5']:7.2f} {r['max_over_min']:8.2f}")

print(f"\n表一：W = {W0} µs（A 星候选 bin 中位），实测/泊松随 k —— **逐段 λ**")
print(f"{'k':>4} " + " ".join(f"{'h' + h:>12}" for h in hours)
      + f" {'几何平均':>12} {'整小时口径':>12} {'漂移污染':>9}")
for k in KS:
    seg = [g(h, W0, k, "log10_ratio_seg") for h in hours]
    glo = [g(h, W0, k, "log10_ratio_global") for h in hours]
    cells = " ".join(f"{(10 ** v if v is not None else float('nan')):12.3e}" for v in seg)
    mark = "  <- 候选 count 中位" if k == 9 else ""
    print(f"{k:4d} {cells} {gm(seg):12.3e} {gm(glo):12.3e} {gm(glo) / gm(seg):9.2f}{mark}")

print(f"\n表二：k = {K0}（候选 count 中位），实测/泊松随窗宽 W —— **逐段 λ**")
print(f"{'W (µs)':>10} " + " ".join(f"{'h' + h:>12}" for h in hours)
      + f" {'几何平均':>12} {'整小时口径':>12} {'漂移污染':>9}")
for w in WS:
    seg = [g(h, w, int(K0), "log10_ratio_seg") for h in hours]
    glo = [g(h, w, int(K0), "log10_ratio_global") for h in hours]
    cells = " ".join(f"{(10 ** v if v is not None else float('nan')):12.3e}" for v in seg)
    print(f"{float(w):10g} {cells} {gm(seg):12.3e} {gm(glo):12.3e} {gm(glo) / gm(seg):9.2f}")

print("\n表三：不含 λ、不含泊松的那一条——实测 '>=8 的格数' 随 W 怎么变")
print(f"{'W (µs)':>10} " + " ".join(f"{'h' + h:>10}" for h in hours))
for w in WS:
    print(f"{float(w):10g} " + " ".join(f"{H[h][w]['obs_tail']['8']:10d}" for h in hours))
print("（这张表整条计算里没有出现过 λ，不受逐段/整小时之争影响）")

print("\n表四：漂移污染的机制是不是 <λ^k>/<λ>^k —— 预言 vs 实测，不是断言")
print("  小 λ 极限下 P(X>=k) ~ λ^k/k!，所以两套期望之比 = <λ_s^k>/<λ_s>^k（按格数 m_s 加权），")
print("  **与 W 无关**（W 在分子分母里同次幂约掉）。这解释了表二里那一列为什么几乎是常数。")
print(f"{'小时':>6} {'k':>4} {'预言 <λ^k>/<λ>^k':>18} {'实测(W=0.149µs)':>16} {'实测(W=1µs)':>14}")
for h in hours:
    sl = np.array(H[h]["seg_live"])
    sn = np.array(H[h]["seg_n"])
    r = sn / sl                      # 逐段率，λ_s ∝ r_s
    wgt = sl / sl.sum()              # 格数正比于活时间
    rbar = float((wgt * r).sum())
    for k in (4, 9, 12):
        pred = float((wgt * (r / rbar) ** k).sum())
        a = g(h, "0.149", k, "log10_ratio_global") - g(h, "0.149", k, "log10_ratio_seg")
        b = g(h, "1.0", k, "log10_ratio_global") - g(h, "1.0", k, "log10_ratio_seg")
        print(f"{h:>6} {k:4d} {pred:18.2f} {10 ** a:16.2f} {10 ** b:14.2f}")
