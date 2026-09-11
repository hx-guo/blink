"""重跑还没完，先从旧池 + 逐候选改正因子预测新池长什么样。

要回答的是一个设计问题：**新批自己的本底档够不够用**。LST 腿与构成腿都要一批
「本底主导」的候选当本底样本；新口径下触发阈 20.0 的物理含义收紧约 17 倍，
旧批 fa > 1 的那 22205 个大部分不会被写出来。

预测 = 旧 fa × 改正因子，只留 <= 20 的（新批的存储阈）。
"""
import json
import math
import sys


def pmf(c, lam):
    if lam <= 0:
        return 1.0 if c == 0 else 0.0
    return math.exp(-lam + c * math.log(lam) - math.lgamma(c + 1.0))


recs = json.load(open(sys.argv[1]))
rows = []
for r in recs:
    s = r["signal"]
    if (r.get("train") or {}).get("is_train"):
        continue
    sf, fa = float(s["sf"]), float(s["false_positive_per_year"])
    if sf <= 0 or fa <= 0:
        continue
    f = 1.0 + pmf(int(s["count"]), float(s["mean"])) / sf
    rows.append((fa, fa * f, f))

print("旧池（去 train）%d 个。" % len(rows))
kept = [r for r in rows if r[1] <= 20.0]
print("新批会写出来的（新 fa <= 20）：**%d 个 = %.1f%%**" % (len(kept), 100.0 * len(kept) / len(rows)))
print("被写出来的那批，对应的旧 fa 上限约 %.3f" % max(r[0] for r in kept))

print("\n新批自己的 fa 分档（预测）：")
edges = [0, 1e-5, 1e-3, 0.1, 1.0, 5.0, 20.0]
for i in range(len(edges) - 1):
    n = sum(1 for r in kept if edges[i] < r[1] <= edges[i + 1])
    o = sum(1 for r in rows if edges[i] < r[0] <= edges[i + 1])
    print("  fa %8.0e .. %8.0e   新 %6d   （旧批同一档 %6d）" % (edges[i], edges[i + 1], n, o))

print("\n新批里「本底主导」能有多少个：")
for lo in (1.0, 5.0, 10.0):
    n = sum(1 for r in kept if r[1] > lo)
    print("  新 fa > %-4g ：%6d 个" % (lo, n))
print("对照：旧批 fa > 1 的本底样本是 22205 个（LST 腿与构成腿都用它）。")

sig_old = sum(1 for r in rows if r[0] <= 1e-5)
sig_new = sum(1 for r in rows if r[1] <= 1e-5)
print("\n显著池：旧 %d → 新 %d（掉出 %d，这是上界——新批会重选最佳格）"
      % (sig_old, sig_new, sig_old - sig_new))
