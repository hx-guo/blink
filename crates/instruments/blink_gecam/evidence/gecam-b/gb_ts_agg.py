"""汇总 `gb_tau_scan.py`：τ 的取舍曲线。

每个 τ 并排两条：
* **代价** = 141 个已发表 TGF 的净计数掉多少（固定窗）；
* **收益** = 整小时里「10 µs 格中 ≥ 8 个计数」的格数掉多少（`fa` 真正吃饭的尾巴）。

**τ = 0 是"时戳精确相等才合并"，不是基线**；基线那一列是 `base`。

⚠️ **τ = 0 与 τ > 0 跨纪元不可比**：τ = 0 等于量化步 `q` 本身，而 B 星有三个 q
纪元（7.45 / 14.9 / 29.8 ns，第 18 条）。**偶然同戳率随 q 成正比，物理没变**，
所以 τ=0 那一列要连同 `(n−1)q/W` 一起看，别让量化冒充符合。

用法: python3 gb_ts_agg.py [目录=tauscan]
"""

import csv
import datetime as dt
import glob
import os
import sys

import numpy as np

EPOCH = (2019, 1, 1)
CATALOG = os.environ.get("GB_CATALOG", "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")
MATCH_LO, MATCH_HI = -1000e-6, 3000e-6
TAGS = ("base", "0", "30", "50", "100", "150", "300", "1000")


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "tauscan"
    acc, tot = {}, {t: [0.0, 0.0] for t in TAGS}     # [事例数, 尾巴格数]
    files = sorted(glob.glob(f"{directory}/*.npz"))
    for f in files:
        z = np.load(f)
        for k in ("t0", "bin_s", "fa_archive"):
            acc.setdefault(k, []).append(z[k])
        for t in TAGS:
            acc.setdefault(f"n_{t}", []).append(z[f"n_{t}"])
            tot[t][0] += float(z[f"nev_{t}"][0])
            tot[t][1] += float(z[f"tail_{t}"][0])
    D = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"小时 {len(files)}，候选 {D['t0'].size}")

    catalog = list(csv.DictReader(open(CATALOG)))
    ut = np.array([met(c["UT"]) for c in catalog])
    bw = D["bin_s"] * 1e6
    sel, sel50 = [], []
    for k in range(len(catalog)):
        d = D["t0"] - ut[k]
        m = (d >= MATCH_LO) & (d <= MATCH_HI)
        if not m.any():
            continue
        j = np.flatnonzero(m)
        sel.append(int(j[np.argmin(D["fa_archive"][j])]))
        jj = j[bw[j] >= 50]
        if jj.size:
            sel50.append(int(jj[np.argmin(D["fa_archive"][jj])]))
    sel, sel50 = np.array(sel), np.array(sel50)
    print(f"匹配上：最显著口径 {sel.size}，`bin ≥ 50 µs` 口径 {sel50.size}")
    print()
    base_ev, base_tail = tot["base"]
    print(f"{'τ (ns)':>8}{'整小时并掉%':>13}{'尾巴格数':>12}{'尾巴剩':>9}"
          f"{'TGF净计数掉%(最显著)':>22}{'(bin≥50µs)':>14}{'池中位掉%':>11}")
    for t in TAGS:
        ev, tail = tot[t]
        merged = (base_ev - ev) / base_ev * 100
        d1 = (D["n_base"][sel] - D[f"n_{t}"][sel]) / np.maximum(D["n_base"][sel], 1)
        d2 = ((D["n_base"][sel50] - D[f"n_{t}"][sel50]) / np.maximum(D["n_base"][sel50], 1)
              if sel50.size else np.array([np.nan]))
        dp = (D["n_base"] - D[f"n_{t}"]) / np.maximum(D["n_base"], 1)
        lab = "基线" if t == "base" else t
        print(f"{lab:>8}{merged:>13.3f}{tail:>12.0f}{tail / max(base_tail, 1):>9.4f}"
              f"{np.median(d1) * 100:>22.3f}{np.median(d2) * 100:>14.3f}"
              f"{np.median(dp) * 100:>11.3f}")
    print()
    print("说明：『尾巴格数』= 整小时里 10 µs 格中计数 ≥ 8 的格数（`min_number = 8`），")
    print("      基线那一格就是没合并时的值；『尾巴剩』= 该 τ 下剩下的比例。")


if __name__ == "__main__":
    main()
