"""`frac_ev_in3` 全任务扫描的收口：分布、磁纬关系、能不能定一道阈。

三个问题（判据三步验证里缺的那一条"这一类有多少、长什么样"）：

1. **分布**：`frac_ev_in3` 与 `ratio_frame`（实测 ÷ 纯帧偶然期望）的逐过境分布，
   正常带在哪、尾巴在哪、有没有干净的空隙。
2. **是不是磁纬门**：把 `ratio_frame` 对 |偶极磁纬| 画一遍。共帧星上这个量先验上
   可能退化成速率代理（速率随磁纬陡变），若真如此就等于偷偷加了一道磁纬门，
   **必须显式说出来**。
3. **阈与余量**：正对照是 03B 那个真实缺陷实例（所在过境 `frac_ev_in3` = 0.78，
   正常过境 8e-4–1.7e-2）。阈要抓得住它，并报它离阈的余量。

用法: python3 scan_summary.py <trip_pass_*.csv> [更多]
"""

import csv
import sys

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    if not r.get("rate_cps"):
                        continue
                    rows.append(r)
                except Exception:
                    continue
    return rows


def col(rows, k, default=np.nan):
    out = []
    for r in rows:
        v = r.get(k, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(default)
    return np.array(out)


def main():
    rows = load(sys.argv[1:])
    sat = np.array([r["sat"] for r in rows])
    src = np.array([r.get("pos_src", "") for r in rows])
    frac = col(rows, "frac_ev_in3")
    ratio = col(rows, "ratio_frame")
    rate = col(rows, "rate_cps")
    mlat = col(rows, "mlat_med")
    f40 = col(rows, "frac_t_mlat40")
    k4 = col(rows, "k4")
    k3 = col(rows, "k3")
    eq4 = col(rows, "e_q4")
    emed = col(rows, "e_med")
    for s in sorted(set(sat.tolist())):
        m = (sat == s) & np.isfinite(frac)
        if m.sum() < 10:
            continue
        f = frac[m]
        r = ratio[m][np.isfinite(ratio[m])]
        print("\n=== %s  过境 %d 次 ===" % (s, m.sum()))
        print("  位置来源: %s" % {k: int((src[m] == k).sum()) for k in sorted(set(src[m].tolist()))})
        print("  frac_ev_in3 分位 (1/25/50/75/95/99/max): %s"
              % np.array2string(np.percentile(f, [1, 25, 50, 75, 95, 99, 100]),
                                precision=2, formatter={"float": lambda x: "%.2e" % x}))
        print("  ratio_frame 分位 (1/25/50/75/95/99/max): %s"
              % np.array2string(np.percentile(r, [1, 25, 50, 75, 95, 99, 100]),
                                precision=1, formatter={"float": lambda x: "%.1f" % x}))
        for thr in (0.05, 0.02, 0.01, 5e-3, 2e-3):
            print("    frac_ev_in3 > %-7s: %4d 次 (%.2f%%)"
                  % (thr, int((f > thr).sum()), 100.0 * (f > thr).mean()))
        # 磁纬关系
        mm = m & np.isfinite(mlat)
        if mm.sum() > 30:
            print("  ratio_frame 对 |偶极磁纬|（逐过境中位磁纬分档）：")
            print("    %-12s %6s %10s %10s %10s" % ("磁纬档", "过境", "速率中位", "frac 中位", "ratio 中位"))
            for a, b in ((0, 15), (15, 25), (25, 35), (35, 45), (45, 90)):
                q = mm & (mlat >= a) & (mlat < b)
                if q.sum() < 5:
                    continue
                print("    %-12s %6d %10.0f %10.2e %10.1f"
                      % ("%d-%d" % (a, b), q.sum(), np.median(rate[q]),
                         np.median(frac[q]), np.median(ratio[q][np.isfinite(ratio[q])])))
            print("  ratio_frame 对**过境内高磁纬时长占比** frac_t_mlat40：")
            for a, b in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 1.01)):
                q = mm & np.isfinite(f40) & (f40 >= a) & (f40 < b)
                if q.sum() < 5:
                    continue
                print("    %-12s %6d %10.0f %10.2e %10.1f"
                      % ("%.2f-%.2f" % (a, b), q.sum(), np.median(rate[q]),
                         np.median(frac[q]), np.median(ratio[q][np.isfinite(ratio[q])])))
        # 尾巴长什么样
        top = np.flatnonzero(m)[np.argsort(frac[m])[::-1][:8]]
        print("  frac_ev_in3 最高的 8 次过境：")
        print("    %-12s %-42s %8s %10s %8s %8s %7s %7s"
              % ("日期", "文件", "速率", "frac", "k3", "k4", "e_q4", "磁纬"))
        for i in top:
            print("    %-12s %-42s %8.0f %10.2e %8d %8d %7.0f %7s"
                  % (rows[i]["day"], rows[i]["pass_file"][:42], rate[i], frac[i],
                     int(k3[i]), int(k4[i]), eq4[i] if np.isfinite(eq4[i]) else -1,
                     rows[i].get("mlat_med", "")))
        print("  参照：本底能量中位 %.0f keV；四重簇事例能量中位（尾部过境）见上列 e_q4"
              % np.median(emed[m]))


if __name__ == "__main__":
    main()
