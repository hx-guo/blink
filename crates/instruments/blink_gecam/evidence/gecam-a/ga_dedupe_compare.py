"""f2/f3 对双增益去重口径有多敏感——同一天、同一批候选，只换去重规则。

统筹手上那条链（2024-01-11：62,830 -> 22,420 -> 1,821 -> 44 -> 9）里的
`f3 == 0` 只留 2.4%，而按 Rust 现行去重口径重算是留 39%。gecam-c 的注记早就
指出过机制：不合并时"一个真的跨探头二重 + 一条同探头重复"会冒充成三重，f3
系统性虚高。这里把两种口径并排跑一遍，把这个机制量成数。
"""
import csv

import numpy as np

D = "/scratchfs2/gecam/guohx/gecam_a/"
EXP = 7050.0     # 2024-01-11 曝光
TARGET = 6.20e-6


def load(p):
    rows = list(csv.DictReader(open(p)))
    out = {}
    for k, t in [("count", int), ("n_obs", int), ("n_keep", int), ("bin_us", float),
                 ("f2", float), ("f3", float), ("n_trip", int), ("e_n_trip", float),
                 ("fa", float)]:
        out[k] = np.array([t(r[k]) for r in rows])
    out["start"] = np.array([r["start"] for r in rows])
    return out


new = load(D + "chain_20240111.csv")
old = load(D + "chain_20240111_olddedupe.csv")
assert (new["start"] == old["start"]).all(), "两次跑的候选顺序不一致"
N = new["count"].size
print(f"2024-01-11，同一批 {N} 个候选，只换双增益去重口径\n")

print(f"{'量':28} {'新口径(死时间/一高一低)':>24} {'老口径(时戳相等)':>20}")
for name, key in [("对账 n_obs == count", None),
                  ("窗内事例数 n_obs 中位", "n_obs"),
                  ("f2 中位", "f2"),
                  ("f3 中位", "f3"),
                  ("f3 > 0 的占比 %", None),
                  ("f2 <= 0.3 的占比 %", None)]:
    if key:
        a, b = np.median(new[key]), np.median(old[key])
        print(f"{name:28} {a:24.3f} {b:20.3f}")
    elif name.startswith("对账"):
        a = (new["n_obs"] == new["count"]).mean() * 100
        b = (old["n_obs"] == old["count"]).mean() * 100
        print(f"{name:28} {a:23.2f}% {b:19.2f}%")
    elif name.startswith("f3 >"):
        print(f"{name:28} {(new['f3']>0).mean()*100:23.2f}% {(old['f3']>0).mean()*100:19.2f}%")
    else:
        print(f"{name:28} {(new['f2']<=0.3).mean()*100:23.2f}% {(old['f2']<=0.3).mean()*100:19.2f}%")

print(f"\n统筹那条链的次序，两种口径各走一遍")
print(f"{'步骤':30} {'新口径':>9} {'老口径':>9}")
for name, mk in [("原始池", lambda d: np.ones(N, bool)),
                 ("+ 准入上界后仍 >= 8", lambda d: d["n_keep"] >= 8),
                 ("+ 窗 >= 10 us", lambda d: (d["n_keep"] >= 8) & (d["bin_us"] >= 10)),
                 ("+ f3 == 0", lambda d: (d["n_keep"] >= 8) & (d["bin_us"] >= 10) & (d["f3"] == 0)),
                 ("+ f2 <= 0.3", lambda d: (d["n_keep"] >= 8) & (d["bin_us"] >= 10)
                  & (d["f3"] == 0) & (d["f2"] <= 0.3))]:
    print(f"{name:30} {int(mk(new).sum()):9d} {int(mk(old).sum()):9d}")

print(f"\n同一条链把 窗>=10us 换成 窗>=50us（新口径）")
for thr in (10, 50):
    s = (new["n_keep"] >= 8) & (new["bin_us"] >= thr) & (new["f3"] == 0) & (new["f2"] <= 0.3)
    n = int(s.sum())
    print(f"  窗 >= {thr:3d} us -> {n:5d} 个 = {n/EXP:.3e} /s，离 6.20e-6 差 {n/EXP/TARGET:.0f} 倍")

print(f"\nf3 的偶然期望（新口径，逐候选按自己的 (n, W, q) 算）")
print(f"{'窗长档':16} {'n':>7} {'偶然三重期望和':>13} {'实测三重数':>10} {'实测/偶然':>10} {'f3>0 占比':>10}")
b = new["bin_us"]
for lo, hi, name in [(0, 1, "< 1 us"), (1, 10, "1–10 us"), (10, 50, "10–50 us"),
                     (50, 200, "50–200 us"), (200, 1e9, ">= 200 us")]:
    s = (b >= lo) & (b < hi)
    if not s.any():
        continue
    e, o = new["e_n_trip"][s].sum(), new["n_trip"][s].sum()
    print(f"{name:16} {int(s.sum()):7d} {e:13.3f} {int(o):10d} "
          f"{o/max(e,1e-9):10.1f} {(new['f3'][s]>0).mean()*100:9.2f}%")
