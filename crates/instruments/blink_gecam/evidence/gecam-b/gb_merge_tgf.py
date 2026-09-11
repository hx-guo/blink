"""GECAM-B 门槛 (1)：τ = 150 ns 跨探头合并对 147 个已发表 TGF 做了什么。

**不止报"在不在"。** 147 个全活着但净计数系统性掉 15%，说明合并在吃真光子——
那也是不能用。所以逐个报**合并前后的计数、本底、显著性**。

**窗口口径（要害）**：合并前后用**同一个窗**（原来的最佳格 `[t0, t0+bin]`）。
若各用各的最佳格，量到的是"最佳格也变了"而不是"光子被吃了"。最佳格重算要跑
搜索，不在本脚本范围内——本脚本给的是**固定窗下的纯损失**。

**本底也要一起改。** 合并把本底率一起降了约 6%，这对显著性是**有利**方向；只扣
分子不扣分母会把损失夸大。所以本底逐候选从事例流现算（候选窗两侧各 1 s、留
10 ms 保护带、夹 GTI），合并前后各算一遍。

**显著性用与搜索同一条公式**：`fa = P(X ≥ count | λ) × 一年秒数 / 窗宽`。
（注意 `blink_algorithms` 现版用的是 `P(X > count)`，差最后一项；这里用
`P(X ≥ count)`，两边一致地用，所以比较仍然成立，但绝对值与落盘的 `fa` 不同口径。）

用法: python3 gb_merge_tgf.py <小时清单> <输出目录> <worker> <workers>
"""

import json
import os
import sys

import numpy as np
from scipy import stats as sps

import gb_feat as gf

TAU = 150e-9
YEAR = 365.25 * 86400.0
BASELINE, GUARD = 1.0, 0.010     # 本底窗半宽 / 紧贴候选窗扣掉的保护带


def merge_cross_det(t, det, tau):
    """跨探头单链接合并，返回分量首事例的下标（与 `gb_poisson.merge_cross_det` 同）。"""
    if t.size == 0:
        return np.zeros(0, np.int64)
    link = (np.diff(t) <= tau) & (det[1:] != det[:-1])
    return np.concatenate(([0], np.flatnonzero(~link) + 1))


def rate_around(t_sorted, t0, t1, gti):
    """候选窗两侧各 BASELINE 秒的本底率（夹 GTI、扣保护带）。"""
    lo0, lo1 = t0 - GUARD - BASELINE, t0 - GUARD
    hi0, hi1 = t1 + GUARD, t1 + GUARD + BASELINE
    n = (np.searchsorted(t_sorted, lo1) - np.searchsorted(t_sorted, lo0)
         + np.searchsorted(t_sorted, hi1) - np.searchsorted(t_sorted, hi0))
    secs = gf.gti_overlap(lo0, lo1, gti) + gf.gti_overlap(hi0, hi1, gti)
    return (n / secs if secs > 0 else np.nan), secs


def fa_of(count, lam, width):
    """与搜索同口径的假阳性率，但用 P(X ≥ count) 而不是 P(X > count)。"""
    if not np.isfinite(lam) or lam <= 0 or width <= 0:
        return np.nan
    return float(sps.poisson.sf(count - 1, lam)) * YEAR / width


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
            print(f"{iso}: 没有 GRD 文件", flush=True)
            continue
        ev = gf.read_grd(path)
        if ev is None:
            print(f"{iso}: 事例为空", flush=True)
            continue
        keep = ev["keep"]
        time, det, gti = ev["time"][keep], ev["det"][keep], ev["gti"]
        head = merge_cross_det(time, det, TAU)
        mtime = time[head]

        day = iso[:10]
        sig_path = (f"{gf.SIGNAL_ROOT}/{day[:4]}/{day[5:7]}/"
                    f"{day.replace('-', '')}_signals.json")
        if not os.path.exists(sig_path):
            print(f"{iso}: 没有候选文件", flush=True)
            continue
        signals = [s for s in json.load(open(sig_path)) if s["start"][:13] == iso]

        cols = {k: [] for k in ("t0", "bin_s", "fa_archive", "count_archive",
                                "n_pre", "n_post", "lam_pre", "lam_post",
                                "bg_secs", "fa_pre", "fa_post")}
        for s in signals:
            t0 = gf.met(s["start"]) + s["delay"]
            w = s["bin_size_best"]
            t1 = t0 + w
            n_pre = int(np.searchsorted(time, t1, "right") - np.searchsorted(time, t0, "left"))
            n_post = int(np.searchsorted(mtime, t1, "right") - np.searchsorted(mtime, t0, "left"))
            r_pre, secs = rate_around(time, t0, t1, gti)
            r_post, _ = rate_around(mtime, t0, t1, gti)
            lam_pre, lam_post = r_pre * w, r_post * w
            cols["t0"].append(t0)
            cols["bin_s"].append(w)
            cols["fa_archive"].append(s["false_positive_per_year"])
            cols["count_archive"].append(s["count"])
            cols["n_pre"].append(n_pre)
            cols["n_post"].append(n_post)
            cols["lam_pre"].append(lam_pre)
            cols["lam_post"].append(lam_post)
            cols["bg_secs"].append(secs)
            cols["fa_pre"].append(fa_of(n_pre, lam_pre, w))
            cols["fa_post"].append(fa_of(n_post, lam_post, w))
        out = f"{outdir}/mt_{iso.replace('-', '').replace('T', '_')}.npz"
        np.savez_compressed(out, **{k: np.array(v, float) for k, v in cols.items()},
                            n_event_pre=np.array([time.size]),
                            n_event_post=np.array([mtime.size]))
        kept = 1 - mtime.size / max(time.size, 1)
        print(f"{iso}: 候选 {len(signals)}，整小时合并掉 {kept * 100:.3f}%", flush=True)


if __name__ == "__main__":
    main()
