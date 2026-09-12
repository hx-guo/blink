"""svomrun8（旧口径 P(X>count)）与 svomrun9（新口径 P(X>=count)）逐候选对账。

README 里最硬的一条：**旧池应当是新池的超集，且逐候选 fa_new >= fa_old**。

**匹配必须带时间容差,不能按 `start` 字符串精确匹配。** 冒烟检验实测：同一个物理候选
在两批里 `count` 与 `bin_size_best` 逐位相同,但 `start` 差 0.06–0.3 ms——sf 变了,
亚格最佳 delay 的 argmin 跟着挪。按字符串匹配会误报「新池冒出候选」。

再给两个先前只能估上界的量的**严格值**：真实掉出多少个（离线的 115 只是上界,
因为离线用的是旧的最佳格,而 sf 也参与选格）、最佳格被重选了多少个。

用法: python3 cmp89.py <old/tgfs.json> <new/tgfs.json>
"""
import bisect
import datetime as dt
import json
import sys

TOL = 0.005          # 5 ms 容差，远大于实测的 0.3 ms 漂移、远小于候选间距


def parse(s):
    d = dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    frac = float("0." + s[20:].rstrip("Z")) if len(s) > 20 else 0.0
    return d.timestamp() + frac


def load(p):
    out = []
    for r in json.load(open(p)):
        s = r["signal"]
        out.append(dict(t=parse(s["start"]), start=s["start"],
                        fa=float(s["false_positive_per_year"]), sf=float(s["sf"]),
                        count=int(s["count"]), mean=float(s["mean"]),
                        bb=float(s["bin_size_best"]),
                        train=bool((r.get("train") or {}).get("is_train")),
                        assoc=bool((r.get("lightning") or {}).get("associated")),
                        cov=bool((r.get("lightning") or {}).get("in_coverage", True))))
    out.sort(key=lambda x: x["t"])
    return out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))] if v else float("nan")


old, new = load(sys.argv[1]), load(sys.argv[2])
ot = [r["t"] for r in old]
print("旧池 %d 个，新池 %d 个（新/旧 = %.1f%%）。" % (len(old), len(new), 100.0 * len(new) / len(old)))

pairs, orphan = [], []
for r in new:
    i = bisect.bisect_left(ot, r["t"] - TOL)
    best = None
    while i < len(ot) and ot[i] <= r["t"] + TOL:
        c = old[i]
        # 同一物理候选：时间在容差内，且 count 与 bin_size_best 一致
        if c["count"] == r["count"] and abs(c["bb"] / r["bb"] - 1) < 1e-9:
            if best is None or abs(c["t"] - r["t"]) < abs(best["t"] - r["t"]):
                best = c
        i += 1
    if best is None:
        i = bisect.bisect_left(ot, r["t"] - TOL)
        while i < len(ot) and ot[i] <= r["t"] + TOL:
            if best is None or abs(old[i]["t"] - r["t"]) < abs(best["t"] - r["t"]):
                best = old[i]
            i += 1
    (pairs if best else orphan).append((r, best) if best else r)

print("新池里在旧池 ±%.0f ms 内找不到对应的：**%d 个（应当是 0，新池必须是旧池的子集）**"
      % (1000 * TOL, len(orphan)))
for r in orphan[:5]:
    print("    %s fa %.4g count %d" % (r["start"], r["fa"], r["count"]))

shift = [abs(a["t"] - b["t"]) for a, b in pairs]
print("\n配上的 %d 对。start 漂移：中位 %.3f ms，最大 %.3f ms"
      % (len(pairs), 1000 * pct(shift, 50), 1000 * max(shift)))
bad = [(a, b) for a, b in pairs if a["fa"] < b["fa"] * (1 - 1e-9)]
print("  fa_new < fa_old 的：%d 个 —— **不为 0 就是混了口径**" % len(bad))
for a, b in bad[:5]:
    print("    %s 旧 %.4g 新 %.4g" % (a["start"], b["fa"], a["fa"]))
ratio = [a["fa"] / b["fa"] for a, b in pairs if b["fa"] > 0]
print("  fa_new / fa_old：中位 %.2f，5–95%% %.2f–%.2f，最大 %.1f"
      % (pct(ratio, 50), pct(ratio, 5), pct(ratio, 95), max(ratio)))
reb = [(a, b) for a, b in pairs if abs(a["bb"] / b["bb"] - 1) > 1e-9]
print("  最佳格被重选的：%d 个 = %.2f%%（这正是离线重算做不到的那一半）"
      % (len(reb), 100.0 * len(reb) / max(len(pairs), 1)))

print("\n判选阈上的严格值（此前离线只能给上界 115）：")
for thr in (1e-5, 1e-3, 1.0):
    a = [r for r in old if r["fa"] <= thr and not r["train"]]
    b = [r for r in new if r["fa"] <= thr and not r["train"]]
    kept = {id(y) for x, y in pairs if x["fa"] <= thr and y is not None and not x["train"]}
    out = sum(1 for r in a if id(r) not in kept)
    print("  fa <= %-6g 旧 %5d → 新 %5d（掉出 %4d）" % (thr, len(a), len(b), out))

for tag, ks in (("旧", [r for r in old if r["fa"] <= 1e-5 and not r["train"]]),
                ("新", [r for r in new if r["fa"] <= 1e-5 and not r["train"]])):
    cov = [r for r in ks if r["cov"]]
    print("  %s：显著 %d，覆盖内 %d，闪电证实 %d"
          % (tag, len(ks), len(cov), sum(1 for r in cov if r["assoc"])))
