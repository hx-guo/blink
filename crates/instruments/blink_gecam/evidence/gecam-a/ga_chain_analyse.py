"""GECAM-A 判据链：准入上界 → bin_size_best → f3 → f2，全是纯计数，不含泊松模型。

池里每个候选按构造就已经满足 fa <= 20（那是搜索自己的门槛），所以链上不再
重复这一刀；每一步只数"还剩几个"。bin_size_best 这一刀的外部真值定标在 B 星
147 个已发表 TGF 上做（bin >= 50 us 留 82.1%），这里只量它在 A 星池上的效果。
"""
import csv
import glob
import json
import sys

import numpy as np
from scipy.stats import poisson

YEAR = 3600 * 24 * 365.25

D = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/gecam_a"
BIN_THR = 50.0
MIN_N = 8
TGF_KEEP = 0.821          # bin >= 50 us 对真 TGF 的保留率（B 星 147 个）
TGF_KEEP_LO, TGF_KEEP_HI = 0.750, 0.876
TGF_COUNT_MED = 64        # B 星 147 个已发表 TGF 被本搜索搜出来时的 count 中位
TARGET = 6.20e-6          # B 星已发表率 /s


# 2026-06-01 的 signals.json 是双增益去重修好之前的老二进制（e34f560，"同探头
# 同时戳"口径）出的，它的 count 系统性偏高，与本脚本用的死时间口径不可比
# （实测对账只有 49.27%，差值全为负）。**整天剔除**，不做部分采信。
SKIP = {"2026-06-01": "signals.json 出自去重修好前的老二进制，count 口径不可比"}


def load():
    days = {}
    for p in sorted(glob.glob(f"{D}/chain_2???????.csv")):
        tag = p.rsplit("_", 1)[-1].split(".")[0]
        day = f"{tag[:4]}-{tag[4:6]}-{tag[6:]}"
        hp = f"{D}/data/GECAM-A/{tag[:4]}/{tag[4:6]}/{tag}_hours.json"
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        if day in SKIP:
            print(f"剔除 {day}：{SKIP[day]}")
            continue
        days[day] = (float(json.load(open(hp))["searched_seconds"]), rows)
    return days


days = load()
if not days:
    sys.exit("没有 chain_*.csv")

cols = {}
for k, typ in [("count", int), ("n_obs", int), ("n_keep", int), ("n_det", int),
               ("bin_us", float), ("f2", float), ("f3", float), ("n_trip", int),
               ("e_f3", float), ("e_n_trip", float), ("fa", float), ("lat", float),
               ("mean", float)]:
    cols[k] = np.concatenate([np.array([typ(r[k]) for r in rows]) for _, rows in days.values()])
# 搜索报的 sf 是 P(X > count)（statrs 的 DiscreteCDF::sf），观测到 count 个
# 计数时的 p 值应当是 P(X >= count)。差的那一项在这种极端尾巴上就是尾巴本身，
# 逐候选低估 (count+1)/lambda 倍（实测中位 1.16e4）。这里把 fa 改对再用。
cols["fa_corr"] = (poisson.sf(cols["count"] - 1, cols["mean"] if "mean" in cols else 0)
                   * YEAR / (cols["bin_us"] * 1e-6))
day_of = np.concatenate([np.full(len(rows), d) for d, (_, rows) in days.items()])
EXP = sum(e for e, _ in days.values())

print(f"{'day':12} {'曝光s':>8} {'候选':>8}")
for d, (e, rows) in days.items():
    print(f"{d:12} {e:8.0f} {len(rows):8d}")
print(f"{'合计':12} {EXP:8.0f} {len(day_of):8d}\n")

ok = cols["n_obs"] == cols["count"]
print(f"对账 n_obs == count: {int(ok.sum())}/{ok.size} = {ok.mean()*100:.2f}%")
if not ok.all():
    dd = (cols["n_obs"] - cols["count"])[~ok]
    print(f"  差值 {np.unique(dd, return_counts=True)}")

N = len(day_of)
print(f"\n{'=' * 72}\n判据链（纯计数）\n{'=' * 72}")
print(f"{'步骤':34} {'幸存':>8} {'留存%':>8} {'率 /s':>10} {'count中位':>9} {'bin中位us':>10}")


def line(name, sel):
    n = int(sel.sum())
    r = n / EXP
    cm = np.median(cols["count"][sel]) if n else float("nan")
    bm = np.median(cols["bin_us"][sel]) if n else float("nan")
    print(f"{name:34} {n:8d} {n/N*100:7.2f}% {r:10.3e} {cm:9.1f} {bm:10.3g}")
    return n, r


steps = []
sel = np.ones(N, bool)
steps.append(("原始池（已含 fa <= 20）", line("原始池（已含 fa <= 20）", sel)))
sel = sel & (cols["n_keep"] >= MIN_N)
steps.append(("+ 准入上界后仍 >= 8 个计数", line("+ 准入上界后仍 >= 8 个计数", sel)))
s_nobin = sel.copy()
sel = sel & (cols["bin_us"] >= BIN_THR)
steps.append((f"+ bin_size_best >= {BIN_THR:.0f} us", line(f"+ bin_size_best >= {BIN_THR:.0f} us", sel)))
sel = sel & (cols["f3"] == 0)
steps.append(("+ f3 == 0", line("+ f3 == 0", sel)))
final = sel & (cols["f2"] <= 0.3)
steps.append(("+ f2 <= 0.3", line("+ f2 <= 0.3", final)))

n_end, r_end = steps[-1][1]
print(f"\nbin 这一刀单独的贡献：{steps[1][1][0]} -> {steps[2][1][0]}，"
      f"砍掉 {(1-steps[2][1][0]/max(steps[1][1][0],1))*100:.2f}%（B 星池上是 98.0%）")
print(f"整条链末端 {n_end} 个 / {EXP:.0f} s = {r_end:.3e} /s")
print(f"  离 6.20e-6 /s 还差 {r_end/TARGET:.1f} 倍")
r_eff = r_end / TGF_KEEP
print(f"  按真 TGF 保留率 {TGF_KEEP:.1%} [{TGF_KEEP_LO:.1%}, {TGF_KEEP_HI:.1%}] 折算成"
      f"「等效全效率率」{r_eff:.3e} /s（区间 {r_end/TGF_KEEP_HI:.3e} .. {r_end/TGF_KEEP_LO:.3e}）"
      f"，差 {r_eff/TARGET:.1f} 倍")

print(f"\n不加 bin 这一刀的同一条链（对照）")
alt = s_nobin & (cols["f3"] == 0) & (cols["f2"] <= 0.3)
n_alt, r_alt = line("准入 + f3==0 + f2<=0.3", alt)
print(f"  bin 这一刀在链末端的额外增益 {n_alt} -> {n_end}，{n_alt/max(n_end,1):.1f} 倍")

print(f"\n{'=' * 72}\ncount 分布：过滤有没有把人群换掉（B 星真 TGF count 中位 {TGF_COUNT_MED}）\n{'=' * 72}")
print(f"{'样本':34} {'n':>8} {'p10':>6} {'p50':>6} {'p90':>6} {'max':>6}")
for name, s in [("原始池", np.ones(N, bool)),
                (f"bin < {BIN_THR:.0f} us（被切掉的）", cols["bin_us"] < BIN_THR),
                (f"bin >= {BIN_THR:.0f} us（留下的）", cols["bin_us"] >= BIN_THR),
                ("链末端", final)]:
    if s.sum() == 0:
        continue
    c = cols["count"][s]
    print(f"{name:34} {int(s.sum()):8d} " + " ".join(f"{np.percentile(c,q):6.0f}" for q in (10, 50, 90))
          + f" {c.max():6.0f}")

print(f"\n{'=' * 72}\nf3：实测 ÷ 它自己的偶然期望（逐候选按自己的 (n, W, q) 算）\n{'=' * 72}")
print(f"{'样本':34} {'n':>7} {'实测f3>0':>9} {'偶然期望三重数':>13} {'实测三重数':>10} {'实测/偶然':>10}")
for name, s in [("原始池", np.ones(N, bool)),
                ("准入后", s_nobin),
                (f"准入 + bin>={BIN_THR:.0f}us", s_nobin & (cols["bin_us"] >= BIN_THR)),
                ("链末端（按构造 f3=0）", final)]:
    if s.sum() == 0:
        continue
    obs = cols["n_trip"][s].sum()
    exp = cols["e_n_trip"][s].sum()
    print(f"{name:34} {int(s.sum()):7d} {(cols['f3'][s]>0).mean()*100:8.2f}% "
          f"{exp:13.2f} {obs:10d} {obs/max(exp,1e-9):10.1f}")

print("\n逐候选余量（准入 + bin>=50us 的子集，按 e_n_trip 升序看最像真信号的几个）")
sub = np.where(s_nobin & (cols["bin_us"] >= BIN_THR))[0]
if sub.size:
    e = cols["e_n_trip"][sub]
    print(f"  偶然三重期望 e_n_trip: p50 {np.median(e):.4f}  p90 {np.percentile(e,90):.4f}  max {e.max():.4f}")
    print(f"  其中实测 f3 > 0 的占 {(cols['f3'][sub]>0).mean()*100:.2f}%"
          f"——偶然期望这么小而实测这么高，只能是粒子")

print(f"\n{'=' * 72}\n链末端逐候选\n{'=' * 72}")
idx = np.where(final)[0]
order = idx[np.argsort(-cols["bin_us"][idx])]
print(f"{'day':12} {'bin_us':>9} {'count':>6} {'n_keep':>7} {'n_det':>6} "
      f"{'f2':>6} {'f3':>5} {'fa':>10} {'lat':>7} {'bin余量x':>9}")
for i in order[:40]:
    print(f"{day_of[i]:12} {cols['bin_us'][i]:9.1f} {cols['count'][i]:6d} {cols['n_keep'][i]:7d} "
          f"{cols['n_det'][i]:6d} {cols['f2'][i]:6.3f} {cols['f3'][i]:5.2f} {cols['fa'][i]:10.2e} "
          f"{cols['lat'][i]:7.2f} {cols['bin_us'][i]/BIN_THR:9.2f}")

print(f"\n{'=' * 72}\n验收线：按 B 星已发表率，这批曝光里该有几个真 TGF\n{'=' * 72}")
print(f"  {EXP:.0f} s × {TARGET:.2e} /s = {EXP*TARGET:.2f} 个。"
      f"链末端是 {n_end} 个，**多出 {n_end/(EXP*TARGET):.0f} 倍**。")
print("  也就是说目标不是把它降到几十，是降到 0 个量级。")

print(f"\n{'=' * 72}\n还剩哪些把手：在链末端的样本上继续扫\n{'=' * 72}")
print(f"{'再加的一刀':28} {'幸存':>7} {'率 /s':>10} {'离目标':>9} {'count中位':>9}")
for name, extra in [("（不加）", np.ones(N, bool)),
                    ("count >= 18（B 真 TGF p5）", cols["count"] >= 18),
                    ("count >= 30", cols["count"] >= 30),
                    ("count >= 64（B 真 TGF 中位）", cols["count"] >= 64),
                    ("fa(报出的) <= 1", cols["fa"] <= 1),
                    ("fa(报出的) <= 1e-4", cols["fa"] <= 1e-4),
                    ("fa(报出的) <= 1e-7", cols["fa"] <= 1e-7),
                    ("**fa(改对的) <= 20**", cols["fa_corr"] <= 20),
                    ("**fa(改对的) <= 1**", cols["fa_corr"] <= 1),
                    ("**fa(改对的) <= 1e-2**", cols["fa_corr"] <= 1e-2),
                    ("**fa(改对的) <= 1e-4**", cols["fa_corr"] <= 1e-4),
                    ("fa(改对的) <= 1 且 count >= 18",
                     (cols["fa_corr"] <= 1) & (cols["count"] >= 18))]:
    s = final & extra
    n = int(s.sum())
    cm = np.median(cols["count"][s]) if n else float("nan")
    print(f"{name:28} {n:7d} {n/EXP:10.3e} {n/EXP/TARGET:8.0f}x {cm:9.1f}")

print(f"\nfa 在清过的池上恢复成泊松尾巴了吗（对照：原始池上同一把刀）")
print(f"{'fa 阈':>10} {'原始池幸存':>10} {'留存%':>8} {'链末端幸存':>10} {'留存%':>8}")
for t in (20, 1, 1e-2, 1e-4, 1e-7):
    a = int((cols["fa_corr"] <= t).sum())
    b = int((final & (cols["fa_corr"] <= t)).sum())
    print(f"{t:10.0e} {a:10d} {a/N*100:7.2f}% {b:10d} {b/max(n_end,1)*100:7.2f}%")
print("（上表用的是改对的 fa = P(X >= count) · 一年秒数 / bin）")
