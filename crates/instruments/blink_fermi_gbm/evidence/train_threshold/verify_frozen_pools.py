"""逐行对账：把 train 阈从绝对 34 换成按池缩放后，已冻结的池标记有没有变。

「形式上只会抬高所以不变」不算证据，这个脚本要的是逐行 diff 出零差异。

三个已冻结的池：
  HXMT v6   /scratchfs2/gecam/guohx/v6run/tgfs_v6.csv     （1,709,490 行，带 is_train）
  SVOM v8   scratch_gbm/tgfs_svom_v8.json                  （25,756 条，带 is_train）
  GBM 2014  scratch_gbm/run/cand_2014.csv                  （107,948 个，邻居数就地算）

算法与 blink_wwlln::train_threshold 逐字对应：
  median  取偏下的那个（sorted[(n-1)//2]），不插值
  round   远离零取整（Rust f64::round），不是 Python 的 banker's rounding

用法：
  python3 verify_frozen_pools.py hxmt <tgfs_v6.csv>
  python3 verify_frozen_pools.py svom <tgfs_svom_v8.json>
  python3 verify_frozen_pools.py gbm2014 <cand_2014.csv>
"""
import csv, datetime as dt, json, math, sys

FLOOR = 34
RATIO = 34.0 / 9.0
HALF_WINDOW_US = 600 * 1_000_000


def median_lower(values):
    s = sorted(values)
    return s[(len(s) - 1) // 2]


def rust_round(x):
    """Rust f64::round：远离零取整。Python 的 round() 是 banker's rounding，不能用。"""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def train_threshold(neighbors):
    if not neighbors:
        return FLOOR
    return max(FLOOR, rust_round(RATIO * median_lower(neighbors)))


def neighbor_counts_us(times_us):
    import bisect
    s = sorted(times_us)
    out = []
    for t in times_us:
        lo = bisect.bisect_left(s, t - HALF_WINDOW_US)
        hi = bisect.bisect_left(s, t + HALF_WINDOW_US)
        out.append(hi - lo - 1)
    return out


def report(pool, neighbors, frozen_flags):
    thr = train_threshold(neighbors)
    med = median_lower(neighbors)
    new = [n > thr for n in neighbors]
    diff = sum(1 for a, b in zip(frozen_flags, new) if a != b)
    print("%s: N=%d  邻居中位=%d  旧阈=%d 新阈=%d" % (pool, len(neighbors), med, FLOOR, thr))
    print("  冻结标记 %d (%.4f%%)  新标记 %d (%.4f%%)"
          % (sum(frozen_flags), 100 * sum(frozen_flags) / len(neighbors),
             sum(new), 100 * sum(new) / len(neighbors)))
    print("  逐行差异: %d  ->  %s" % (diff, "零差异，通过" if diff == 0 else "不通过"))
    return diff


def main():
    which, path = sys.argv[1], sys.argv[2]
    if which == "hxmt":
        nb = []; fl = []
        with open(path) as f:
            for r in csv.DictReader(f):
                nb.append(int(r["neighbors_10min"])); fl.append(r["is_train"] == "1")
        return report("HXMT v6", nb, fl)
    if which == "svom":
        d = json.load(open(path))
        nb = [x["train"]["neighbors_10min"] for x in d]
        fl = [x["train"]["is_train"] for x in d]
        return report("SVOM v8", nb, fl)
    if which == "gbm2014":
        REF = dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc)
        times = []
        with open(path) as f:
            for r in csv.DictReader(f):
                b = r["start"].rstrip("Z"); h, _, fr = b.partition(".")
                t = dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
                times.append(int(round((((t - REF).total_seconds() + (float("0." + fr) if fr else 0.0))) * 1e6)))
        nb = neighbor_counts_us(times)
        # GBM 2014 还没进过 tgfs.json，冻结标记就是旧阈的结果
        fl = [n > FLOOR for n in nb]
        return report("Fermi/GBM 2014", nb, fl)
    raise SystemExit("unknown pool: " + which)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
