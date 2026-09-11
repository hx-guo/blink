"""GECAM-A：把搜索自己报出来的 bin_size_best 当判别量事后过滤。

外部真值定标（gecamB 在 147 个已发表 TGF 上做的）：真 TGF 的 bin_size_best
中位 111.4 us、bin >= 50 us 留 82.1% [75.0, 87.6]；候选池中位 0.134 us、
同一刀切掉 98.0%。这里只做一件事：在 A 星的候选表上量同一刀的效果。

注意口径：不抬 min_duration（那只会让搜索改报一个不同的 bin_size_best），
过滤的是搜索照常报出来的值。
"""
import csv
import json
import sys
import numpy as np

ROOT = "/scratchfs2/gecam/guohx/gecam_a"
DAYS = ["2023-06-01", "2023-12-16", "2024-01-11", "2024-06-02", "2026-06-01"]
THR = [0, 1, 5, 10, 20, 50, 100, 200, 500]
TGF_KEEP = 0.821  # bin >= 50 us 对真 TGF 的保留率（B 星 147 个定标）


def load(day):
    tag = day.replace("-", "")
    y, m, _ = day.split("-")
    hp = f"{ROOT}/data/GECAM-A/{y}/{m}/{tag}_hours.json"
    cp = f"{ROOT}/jit_{tag}_cand.csv"
    hrs = json.load(open(hp))
    rows = list(csv.DictReader(open(cp)))
    return hrs, rows


print(f"{'day':12} {'曝光s':>9} {'池':>8} {'率/s':>9} "
      + " ".join(f"{'b>=%d' % t:>8}" for t in THR))
tot = {t: 0 for t in THR}
tot_exp = 0.0
allbin = []
for day in DAYS:
    try:
        hrs, rows = load(day)
    except FileNotFoundError as e:
        print(f"{day:12} 缺文件 {e.filename}")
        continue
    exp = float(hrs["searched_seconds"])
    b = np.array([float(r["bin_us"]) for r in rows])
    allbin.append(b)
    n = len(rows)
    cells = []
    for t in THR:
        k = int((b >= t).sum())
        tot[t] += k
        cells.append(f"{k:8d}")
    tot_exp += exp
    print(f"{day:12} {exp:9.0f} {n:8d} {n/exp:9.3f} " + " ".join(cells))

print(f"{'合计':12} {tot_exp:9.0f} {tot[0]:8d} {tot[0]/tot_exp:9.3f} "
      + " ".join(f"{tot[t]:8d}" for t in THR))
print(f"{'率 /s':12} {'':9} {'':8} {'':9} "
      + " ".join(f"{tot[t]/tot_exp:8.2e}" for t in THR))
print(f"{'留存%':12} {'':9} {'':8} {'':9} "
      + " ".join(f"{tot[t]/tot[0]*100:7.2f}%" for t in THR))

b = np.concatenate(allbin)
qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\n全体候选 bin_us 分位：" + "  ".join(f"p{q}={np.percentile(b,q):.3g}" for q in qs))
print(f"  < 1 us 占 {(b<1).mean()*100:.1f}%   < 0.3 us 占 {(b<0.3).mean()*100:.1f}%")
r50 = tot[50] / tot_exp
print(f"\nbin >= 50 us 之后：{tot[50]} 个 / {tot_exp:.0f} s = {r50:.3e} /s")
print(f"  效率修正（真 TGF 只留 {TGF_KEEP:.1%}）后的等效率 {r50/TGF_KEEP:.3e} /s")
print(f"  离 B 星已发表率 6.20e-6 /s 还差 {r50/6.20e-6:.1f} 倍"
      f"（效率修正后 {r50/TGF_KEEP/6.20e-6:.1f} 倍）")
