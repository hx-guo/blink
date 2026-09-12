"""`frac_ev_in3` 全任务扫描的收口表（共帧三星）。

四件事，按 `tgf` 要的口径：

1. **过境级 `frac_ev_in3` 的完整分布**，**脉冲过境单列**——尾部几乎全是片上标定脉冲
   （500 Hz，见 `pulser.py`），混在一起报会把"这一类有多少"答错。
2. **逐 |偶极磁纬| 带的 p50/p95**，并按 `pos_src` 分列（posatt / orbit_fit / none）——
   拟合轨道给的位置误差中位 73–174 km，不能与位姿给的混着报。
3. **阈的余量**：正常（非脉冲）过境的 p99 离候选阈有多远、脉冲过境离它多远。
4. 是不是一道磁纬门：**控制速率之后**逐带的 frac 还差多少。

用法: python3 scan_final.py <trip_pass_*.csv> [更多]
"""

import csv
import glob
import sys

import numpy as np

MLAT_EDGES = [0, 15, 25, 35, 45, 90]
RATE_BINS = [(400, 800), (800, 1600), (1600, 3200)]


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return np.nan


def main():
    rows = []
    for pat in sys.argv[1:]:
        for p in sorted(glob.glob(pat)):
            with open(p) as f:
                rows.extend(r for r in csv.DictReader(f) if r.get("frac_ev_in3"))
    sat = np.array([r["sat"] for r in rows])
    src = np.array([r.get("pos_src", "") for r in rows])
    frac = np.array([num(r, "frac_ev_in3") for r in rows])
    rate = np.array([num(r, "rate_cps") for r in rows])
    mlat = np.array([num(r, "mlat_med") for r in rows])
    k3 = np.array([num(r, "k3") for r in rows])
    k4 = np.array([num(r, "k4") for r in rows])
    dur = np.array([num(r, "dur_s") for r in rows])
    pulse = (k4 >= 200) & (k4 / np.maximum(k3, 1) > 0.8)

    for s in sorted(set(sat.tolist())):
        m = (sat == s) & np.isfinite(frac)
        if m.sum() < 50:
            continue
        pm, nm = m & pulse, m & ~pulse
        print("\n" + "=" * 78)
        print("%s  过境 %d（脉冲 %d = %.2f%%，其余 %d）  GTI 合计 %.1f h"
              % (s, m.sum(), pm.sum(), 100 * pm.sum() / m.sum(), nm.sum(),
                 dur[m].sum() / 3600.0))
        print("  位置来源: %s"
              % {k: int((src[m] == k).sum()) for k in sorted(set(src[m].tolist()))})
        q = [1, 25, 50, 75, 90, 95, 99, 100]
        print("  %-12s %s" % ("分位", " ".join("%9d%%" % x for x in q)))
        for tag, mm in (("全部过境", m), ("**非脉冲**", nm), ("脉冲过境", pm)):
            if mm.sum() < 5:
                continue
            print("  %-12s %s" % (tag, " ".join("%10.2e" % v
                                                for v in np.percentile(frac[mm], q))))
        print("  阈的余量：")
        for thr in (0.05, 0.02, 0.01, 0.005):
            a = int((frac[nm] > thr).sum())
            b = int((frac[pm] > thr).sum())
            print("    > %-6s 非脉冲 %4d 次（%.2f%%）  脉冲 %4d 次（%.1f%%）"
                  % (thr, a, 100 * a / max(nm.sum(), 1), b, 100 * b / max(pm.sum(), 1)))
        p99 = np.percentile(frac[nm], 99)
        print("    非脉冲过境 p99 = %.3e；脉冲过境中位 = %.3e；**分离 %.0f 倍**"
              % (p99, np.median(frac[pm]) if pm.any() else np.nan,
                 (np.median(frac[pm]) / p99) if pm.any() and p99 > 0 else np.nan))
        # 逐磁纬带，只用非脉冲过境，按 pos_src 分列
        print("  逐 |偶极磁纬| 带（**只用非脉冲过境**）：")
        print("    %-10s %-10s %7s %9s %10s %10s" % ("来源", "磁纬带", "过境", "速率中位", "frac p50", "frac p95"))
        for sc in ("posatt", "orbit_fit"):
            for a, b in zip(MLAT_EDGES[:-1], MLAT_EDGES[1:]):
                q2 = nm & (src == sc) & np.isfinite(mlat) & (mlat >= a) & (mlat < b)
                if q2.sum() < 10:
                    continue
                print("    %-10s %-10s %7d %9.0f %10.2e %10.2e"
                      % (sc, "%d-%d" % (a, b), q2.sum(), np.median(rate[q2]),
                         np.median(frac[q2]), np.percentile(frac[q2], 95)))
        # 控制速率之后还差多少
        print("  控制速率后逐带 frac 中位（非脉冲、posatt）：")
        for lo, hi in RATE_BINS:
            out = []
            for a, b in zip(MLAT_EDGES[:-1], MLAT_EDGES[1:]):
                q2 = (nm & (src == "posatt") & np.isfinite(mlat) & (mlat >= a) & (mlat < b)
                      & (rate >= lo) & (rate < hi))
                out.append("|ml|%d-%d: %s" % (a, b, "%.2e(n=%d)" % (np.median(frac[q2]), q2.sum())
                                              if q2.sum() >= 10 else "n/a"))
            print("    速率 %d-%d  %s" % (lo, hi, "  ".join(out)))


if __name__ == "__main__":
    main()
