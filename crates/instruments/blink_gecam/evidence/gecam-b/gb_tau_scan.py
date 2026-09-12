"""GECAM-B：τ 的取舍曲线——每个 τ 同时量「吃掉多少真 TGF」和「压掉多少假阳性尾巴」。

门槛 (1) 在 τ = 150 ns 上不通过（真 TGF 净计数中位掉 14.8–16.7%，还掉了 1–11 个）。
但 τ = 150 ns 这个值是从**同一路的两个增益支路**量出来的（第 32 条那个约 100 ns
固定延迟），**跨探头符合要的未必是同一个尺度**。所以把 τ 扫一遍，两条曲线并排：

* **代价**：141 个已发表 TGF 的净计数掉多少（固定窗、本底同步重算）；
* **收益**：整小时里「10 µs 格子中出现 ≥ 8 个计数」的格数掉多少——那是 `fa` 真正
  吃饭的那条尾巴（`min_number = 8`），**Fano 看不见它**。

两条曲线的交叉点就是能谈的 τ；若**在任何 τ 上代价都先于收益出现**，那这条路线本身
要换（不是调参数）。

用法: python3 gb_tau_scan.py <小时清单> <输出目录> <worker> <workers>
"""

import json
import os
import sys

import numpy as np

import gb_feat as gf

# −1 是"不合并"的基线；**0 是"时戳精确相等才合并"**，不是不合并——
# 一次带电粒子穿越在多路上留下的是完全相同的时戳（第 32 条：`mf` 与 `f₂` 逐位相等
# ⇒ 同戳簇全部跨探头），而真 TGF 的光子是真正不同的时刻到达的。
TAUS = (-1.0, 0.0, 30e-9, 50e-9, 100e-9, 150e-9, 300e-9, 1000e-9)
TAIL_BIN, TAIL_K = 10e-6, 8      # 尾巴的口径：10 µs 格、≥ min_number = 8


def merge_cross_det(t, det, tau):
    """tau < 0 = 不合并（基线）；tau = 0 = 时戳精确相等才合并；tau > 0 = 单链接容差。"""
    if t.size == 0 or tau < 0:
        return np.arange(t.size, dtype=np.int64)
    link = (np.diff(t) <= tau) & (det[1:] != det[:-1])
    return np.concatenate(([0], np.flatnonzero(~link) + 1))


def tail_count(t, bw, k):
    """`t` 落在宽 `bw` 的格子里、计数 ≥ k 的格数。不开满数组（一小时是 3.6e8 格）。"""
    if t.size == 0:
        return 0
    idx = np.floor(t / bw).astype(np.int64)
    _, c = np.unique(idx, return_counts=True)
    return int((c >= k).sum())


def main():
    hours_file, outdir = sys.argv[1], sys.argv[2]
    worker = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(outdir, exist_ok=True)
    hours = [x.strip() for x in open(hours_file) if x.strip()]

    for idx, iso in enumerate(hours):
        if idx % workers != worker:
            continue
        path = gf.hour_file(iso, "grd")
        if path is None:
            continue
        ev = gf.read_grd(path)
        if ev is None:
            continue
        keep = ev["keep"]
        time, det = ev["time"][keep], ev["det"][keep]

        day = iso[:10]
        sig_path = (f"{gf.SIGNAL_ROOT}/{day[:4]}/{day[5:7]}/"
                    f"{day.replace('-', '')}_signals.json")
        if not os.path.exists(sig_path):
            continue
        signals = [s for s in json.load(open(sig_path)) if s["start"][:13] == iso]
        t0 = np.array([gf.met(s["start"]) + s["delay"] for s in signals])
        wid = np.array([s["bin_size_best"] for s in signals])
        fa0 = np.array([s["false_positive_per_year"] for s in signals])

        out = {"t0": t0, "bin_s": wid, "fa_archive": fa0}
        for tau in TAUS:
            head = merge_cross_det(time, det, tau)
            mt = time[head]
            lo = np.searchsorted(mt, t0, "left")
            hi = np.searchsorted(mt, t0 + wid, "right")
            tag = "base" if tau < 0 else f"{tau * 1e9:.0f}"
            out[f"n_{tag}"] = (hi - lo).astype(float)
            out[f"nev_{tag}"] = np.array([float(mt.size)])
            out[f"tail_{tag}"] = np.array([float(tail_count(mt, TAIL_BIN, TAIL_K))])
        np.savez_compressed(
            f"{outdir}/ts_{iso.replace('-', '').replace('T', '_')}.npz", **out)
        print(f"{iso}: 候选 {len(signals)}，事例 {time.size}，"
              + "  ".join(
                  "τ=%s 尾 %d" % ("base" if t < 0 else "%.0fns" % (t * 1e9),
                                  int(out["tail_" + ("base" if t < 0 else "%.0f" % (t * 1e9))][0]))
                  for t in TAUS), flush=True)


if __name__ == "__main__":
    main()
