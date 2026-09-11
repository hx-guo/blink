"""冒烟检验：新二进制产出的 sf 是不是 P(X >= count)，并与旧口径同一天逐候选比。"""
import json
import math
import sys


def pmf(c, lam):
    if lam <= 0:
        return 1.0 if c == 0 else 0.0
    return math.exp(-lam + c * math.log(lam) - math.lgamma(c + 1.0))


def sf_gt(c, lam):
    if lam <= 0:
        return 0.0
    t = pmf(c + 1, lam)
    s, k = 0.0, c + 1
    while t > 0 and (s == 0 or t > s * 1e-17) and k < c + 60000:
        s += t
        k += 1
        t *= lam / k
    return s


def load(p):
    return {r["start"]: r for r in json.load(open(p))}


new, old = load(sys.argv[1]), load(sys.argv[2])
print("新 %d 个候选，旧 %d 个（同一天）。" % (len(new), len(old)))
d_ge, d_gt = [], []
for r in new.values():
    c, lam, sf = int(r["count"]), float(r["mean"]), float(r["sf"])
    if sf <= 0:
        continue
    d_ge.append(abs((sf_gt(c - 1, lam)) / sf - 1.0))
    d_gt.append(abs((sf_gt(c, lam)) / sf - 1.0))
d_ge.sort(); d_gt.sort()
print("新产物的 sf 对 P(X>=count) 的相对偏差：中位 %.2e，最大 %.2e" % (d_ge[len(d_ge)//2], d_ge[-1]))
print("新产物的 sf 对 P(X> count) 的相对偏差：中位 %.2e，最大 %.2e" % (d_gt[len(d_gt)//2], d_gt[-1]))
print("⇒ %s" % ("新口径生效（sf == P(X>=count)）" if d_ge[-1] < 1e-9 else "**口径不对，别提交**"))

both = set(new) & set(old)
print("\n同一天共有候选 %d 个（新 %d / 旧 %d）" % (len(both), len(new), len(old)))
bad = [k for k in both if float(new[k]["false_positive_per_year"])
       < float(old[k]["false_positive_per_year"]) * (1 - 1e-9)]
print("  fa_new < fa_old 的：%d 个（应当为 0）" % len(bad))
rat = sorted(float(new[k]["false_positive_per_year"]) / float(old[k]["false_positive_per_year"])
             for k in both if float(old[k]["false_positive_per_year"]) > 0)
if rat:
    print("  fa_new/fa_old：中位 %.2f，最小 %.2f，最大 %.1f" % (rat[len(rat)//2], rat[0], rat[-1]))
reb = [k for k in both if abs(float(new[k]["bin_size_best"]) / float(old[k]["bin_size_best"]) - 1) > 1e-9]
print("  最佳格被重选的：%d 个 = %.1f%%" % (len(reb), 100.0 * len(reb) / max(len(both), 1)))
print("  新池里旧池没有的：%d（应当为 0）" % len(set(new) - set(old)))
