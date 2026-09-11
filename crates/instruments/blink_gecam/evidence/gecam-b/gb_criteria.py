"""GECAM-B：拿找回的已发表 TGF 当正样本，把判据逐条反向定标。

**限定先写死，结论怎么读取决于它。** 147 个是 Zhao et al. 2023 的触发式亮样本
（P < 2e-21、`NetCounts` 中位 66.9），而我们的候选计数中位约 10。**完备性高只证明
亮端不漏，证明不了暗端**，所以正样本只能用来**证伪**判据（哪条切掉了真 TGF 就不能
用），不能用来证明判据够用。

**每一条判据都给两个数：正样本保留率 和 对照组保留率。** 单独一个百分数不含信息。
比例一律带 Wilson 95% 区间（`0/142` 的上限不是 0）。

对照组 = 同一批 138 小时里的全部候选，与正样本**同一个二进制、同一份归档、同一条
去重规则**跑出来的——旧的 e34f560 口径候选池不能当对照组。

用法: python3 gb_criteria.py <npz 目录> <recall.csv> [输出目录]
"""

import csv
import glob
import os
import sys

import numpy as np

CPD_LABELS = ("±10 µs", "±100 µs", "±1 ms", "±10 ms")


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 95% 区间。k=0 时给出真正的上限，不是 0。"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n * 100:5.2f}% [{lo * 100:.2f}, {hi * 100:.2f}]" if n else "n=0"


def load(npz_dir):
    """把所有小时的特征表拼起来。kind: 0 候选，1 目录窗，2/3 构造无关的固定窗。"""
    files = sorted(glob.glob(os.path.join(npz_dir, "feat_*.npz")))
    print(f"特征表 {len(files)} 个小时")
    cols, det_w, det_b, det_pos, det_pos_bg, kinds = {}, [], [], [], [], []
    meta = dict(n_event=0, n_merged=0, gti=0.0, hours=len(files))
    skip = {"kind", "det_sum_window", "det_sum_baseline", "det_pos_window",
            "det_pos_baseline", "n_merged_hour", "n_event_hour", "gti_seconds"}
    for path in files:
        z = np.load(path)
        kinds.append(z["kind"])
        for key in z.files:
            if key in skip:
                continue
            cols.setdefault(key, []).append(z[key])
        det_w.append(z["det_sum_window"])
        det_b.append(z["det_sum_baseline"])
        if z["det_pos_window"].size:
            det_pos.append(z["det_pos_window"])
            det_pos_bg.append(z["det_pos_baseline"])
        meta["n_event"] += int(z["n_event_hour"][0])
        meta["n_merged"] += int(z["n_merged_hour"][0])
        meta["gti"] += float(z["gti_seconds"][0])
    allcols = {k: np.concatenate(v) for k, v in cols.items()}
    allcols["kind"] = np.concatenate(kinds)
    meta["det_window"] = np.sum(det_w, axis=0)
    meta["det_baseline"] = np.sum(det_b, axis=0)
    meta["det_pos"] = np.concatenate(det_pos) if det_pos else np.zeros((0, 26), np.int64)
    meta["det_pos_bg"] = np.concatenate(det_pos_bg) if det_pos_bg else np.zeros((0, 26), np.int64)
    return allcols, meta


def main():
    npz_dir, recall_path = sys.argv[1], sys.argv[2]
    allcols, meta = load(npz_dir)
    kind = allcols["kind"]
    pool = {k: v[kind == 0] for k, v in allcols.items()}     # 候选（= 对照组 + 正样本）
    fixed = {kk: {k: v[kind == kk] for k, v in allcols.items()} for kk in (1, 2, 3)}
    rows = list(csv.DictReader(open(recall_path)))
    want = np.array([float(r["t0_met"]) for r in rows if r["matched"] == "1" and r["t0_met"]])
    n_cat = len(rows)

    t0 = pool["t0"]
    order = np.argsort(t0)
    slot = np.clip(np.searchsorted(t0[order], want), 0, t0.size - 1)
    idx = order[slot]
    ok = np.abs(t0[idx] - want) < 1e-9
    pos = idx[ok]
    print(f"\n正样本：目录 {n_cat} 个，找回 {len(want)} 个，对回特征表 {ok.sum()} 个"
          f"{'（有 %d 个对不回去，先查口径）' % (~ok).sum() if (~ok).any() else ''}")
    mask_pos = np.zeros(t0.size, bool)
    mask_pos[pos] = True
    ctrl = ~mask_pos
    print(f"对照组：{ctrl.sum():,} 个候选（同 138 小时、同二进制、同去重规则）")
    print(f"事例合计 {meta['n_event']:,}，双增益并掉 {meta['n_merged']:,} "
          f"({meta['n_merged'] / max(meta['n_event'], 1) * 100:.2f}%)，GTI 合计 {meta['gti']:.0f} s")

    # 对账：重算的 n_core 必须与搜索报的 count 逐条相等
    diff = pool["n_core"] - pool["count"]
    print(f"\n【对账】n_core == count 的占 "
          f"{(diff == 0).mean() * 100:.4f}%（{int((diff == 0).sum()):,}/{diff.size:,}）"
          f"，差值中位 {np.median(diff):+.0f}，非零的 5–95% "
          f"{np.percentile(diff[diff != 0], 5) if (diff != 0).any() else 0:+.0f} .. "
          f"{np.percentile(diff[diff != 0], 95) if (diff != 0).any() else 0:+.0f}")
    if (diff != 0).mean() > 0.001:
        print("！对账不到 99.9%，下面的量先别信，回去修窗口口径")

    P, C = mask_pos, ctrl

    # ---- 1. f₃：落在 ≥3 重跨探头同戳簇里的计数占比 ----
    print("\n=== 1. f₃（≥3 重同戳簇里的计数占比，双增益合并之后算）===")
    binus = pool["bin_s"] * 1e6
    for lo in (0.0, 10.0):
        sel = binus >= lo
        p, c = P & sel, C & sel
        kp = int((pool["f3"][p] > 0).sum())
        kc = int((pool["f3"][c] > 0).sum())
        print(f"  窗长 ≥ {lo:4.0f} µs：正样本 f₃>0 {pct(kp, int(p.sum()))}   "
              f"对照组 f₃>0 {pct(kc, int(c.sum()))}")
    print(f"  偶然三重的期望（Σ exp_trip）：正样本 {np.nansum(pool['exp_trip'][P]):.3e}，"
          f"对照组 {np.nansum(pool['exp_trip'][C]):.3e}")
    print(f"  最大同戳探头数 max_d：正样本中位 {np.median(pool['max_d'][P]):.0f}、"
          f"最大 {np.nanmax(pool['max_d'][P]):.0f}；对照组中位 {np.median(pool['max_d'][C]):.0f}、"
          f"最大 {np.nanmax(pool['max_d'][C]):.0f}")
    print("  【判据 f₃ == 0 的保留率】正样本 %s   对照组 %s"
          % (pct(int((pool["f3"][P] == 0).sum()), int(P.sum())),
             pct(int((pool["f3"][C] == 0).sum()), int(C.sum()))))

    # ---- 2. multiplet_frac ----
    print("\n=== 2. multiplet_frac（参与任何同戳簇的事例占比）===")
    for name, m in (("正样本", P), ("对照组", C)):
        v = pool["mf"][m]
        print(f"  {name} n={m.sum():,}：中位 {np.median(v):.4f}，"
              f"5–95% {np.percentile(v, 5):.4f} .. {np.percentile(v, 95):.4f}，"
              f"== 0 占 {(v == 0).mean() * 100:.2f}%")
    for thr in (0.0, 0.25, 0.30, 0.50):
        print(f"  【mf ≤ {thr:.2f} 的保留率】正样本 %s   对照组 %s"
              % (pct(int((pool["mf"][P] <= thr).sum()), int(P.sum())),
                 pct(int((pool["mf"][C] <= thr).sum()), int(C.sum()))))

    # ---- 3. CPD 符合 ----
    print("\n=== 3. CPD 符合（分子分母都夹 GTI；期望 = 当地实测率 × 符合窗∩GTI）===")
    print(f"  当地 CPD 率：正样本中位 {np.nanmedian(pool['cpd_rate'][P]):.1f} c/s，"
          f"对照组中位 {np.nanmedian(pool['cpd_rate'][C]):.1f} c/s")
    for i, label in enumerate(CPD_LABELS):
        line = []
        for name, m in (("正样本", P), ("对照组", C)):
            o = np.nansum(pool[f"obs{i}"][m])
            e = np.nansum(pool[f"exp{i}"][m])
            r = o / e if e > 0 else np.nan
            s = np.sqrt(max(o, 1)) / e if e > 0 else np.nan
            hit = int((pool[f"obs{i}"][m] > 0).sum())
            line.append(f"{name} obs/exp = {r:6.3f} ± {s:.3f} (obs {o:.0f}, exp {e:.1f})"
                        f"  命中 {pct(hit, int(m.sum()))}")
        print(f"  {label:8s} " + "\n           ".join(line))

    # ---- 4. 成串尺度扫描 ----
    print("\n=== 4. 成串判据的尺度扫描（阈 = 池率给的泊松期望 + 5σ，不看幸存者，无循环）===")
    ts = np.sort(t0)
    span = meta["gti"]           # 分母用活时间，不是墙钟
    density = ts.size / span
    print(f"  池率 {density:.3f} 个/s（{ts.size:,} 个候选 / {span:.0f} s 活时间）")
    for W in (0.5, 5.0, 30.0, 60.0, 600.0):
        lo_i = np.searchsorted(ts, ts - W, "left")
        hi_i = np.searchsorted(ts, ts + W, "right")
        nb_sorted = hi_i - lo_i - 1
        nb = np.empty_like(nb_sorted)
        nb[np.argsort(t0)] = nb_sorted
        mu = density * 2 * W
        thr = mu + 5 * np.sqrt(mu)
        print(f"  ±{W:<6g} s 期望 {mu:9.1f}，阈 {thr:9.1f} | "
              f"正样本邻居中位 {np.median(nb[P]):8.0f}，超阈 {pct(int((nb[P] > thr).sum()), int(P.sum()))} | "
              f"对照组中位 {np.median(nb[C]):8.0f}，超阈 {pct(int((nb[C] > thr).sum()), int(C.sum()))}")

    # ---- 5. bin_size_best ----
    print("\n=== 5. bin_size_best 分布（与目录 Duration_us 口径不同：后者是 Bayesian Block 块长）===")
    for name, m in (("正样本", P), ("对照组", C)):
        v = pool["bin_s"][m] * 1e6
        q = np.percentile(v, [0, 1, 5, 10, 25, 50, 75, 95])
        print(f"  {name} n={m.sum():,}：min {q[0]:.4f}  p1 {q[1]:.4f}  p5 {q[2]:.4f}  "
              f"p10 {q[3]:.4f}  p25 {q[4]:.3f}  中位 {q[5]:.3f}  p75 {q[6]:.3f}  p95 {q[7]:.3f} µs")
        print(f"       < 0.3 µs: {pct(int((v < 0.3).sum()), int(m.sum()))}   "
              f"< 1 µs: {pct(int((v < 1.0).sum()), int(m.sum()))}   "
              f"≥ 900 µs（顶到 1 ms 上限）: {pct(int((v >= 900).sum()), int(m.sum()))}")
    dur = np.array([float(r["Duration_us"]) for r in rows])
    print(f"  目录 Duration_us：中位 {np.median(dur):.1f}，5–95% {np.percentile(dur, 5):.1f} .. "
          f"{np.percentile(dur, 95):.1f}，最大 {dur.max():.1f}")

    # ---- 6. PI >= 368 ----
    print("\n=== 6. PI ≥ 368（第 19 条的超量程堆积包）===")
    for name, m in (("正样本", P), ("对照组", C)):
        n = pool["n_core"][m]
        f = np.divide(pool["n_pi368"][m], n, out=np.zeros_like(n, float), where=n > 0)
        nb = pool["n_bg_grd"][m]
        fb = np.divide(pool["n_pi368_bg"][m], nb, out=np.zeros_like(nb, float), where=nb > 0)
        print(f"  {name}：窗内占比中位 {np.median(f):.5f}，>0 的占 {(f > 0).mean() * 100:.2f}%；"
              f"本底窗占比中位 {np.median(fb):.5f}；窗/本底 "
              f"{np.nansum(pool['n_pi368'][m]) / max(np.nansum(pool['n_core'][m]), 1) / max(np.nansum(pool['n_pi368_bg'][m]) / max(np.nansum(pool['n_bg_grd'][m]), 1), 1e-12):.3f}")

    # ---- 7. 逐路探头分布 ----
    print("\n=== 7. 逐路 GRD 分布（25 路朝向不同；正样本 vs 本底）===")
    w, b = meta["det_window"][1:26].astype(float), meta["det_baseline"][1:26].astype(float)
    pw = meta["det_pos"][:, 1:26].sum(axis=0).astype(float) if meta["det_pos"].size else np.zeros(25)
    pb = meta["det_pos_bg"][:, 1:26].sum(axis=0).astype(float) if meta["det_pos_bg"].size else np.zeros(25)
    print("  探头  对照窗占比  对照本底占比  比值 | 正样本窗占比  正样本本底占比  比值")
    for d in range(25):
        rc = (w[d] / w.sum()) / (b[d] / b.sum()) if b.sum() and w.sum() else np.nan
        rp = (pw[d] / pw.sum()) / (pb[d] / pb.sum()) if pb.sum() and pw.sum() else np.nan
        print(f"   {d + 1:2d}   {w[d] / max(w.sum(), 1) * 100:7.3f}%    "
              f"{b[d] / max(b.sum(), 1) * 100:7.3f}%   {rc:5.3f} | "
              f"{pw[d] / max(pw.sum(), 1) * 100:7.3f}%     {pb[d] / max(pb.sum(), 1) * 100:7.3f}%   {rp:5.3f}")
    zb = pool["n_det_zero_bg"]
    print(f"  基线零格（25 路里有几路在该候选的基线窗内一个计数都没有，第 4c 条）：")
    print(f"    正样本 有零格的 {pct(int((zb[P] > 0).sum()), int(P.sum()))}；"
          f"对照组 有零格的 {pct(int((zb[C] > 0).sum()), int(C.sum()))}")

    # ---- 8. 构造无关的固定窗复测 ----
    print("\n=== 8. 构造无关复测（窗只由目录 UT 定，不含我们的最佳格选择）===")
    print("  正样本（最佳格窗）  vs  固定窗 [UT−0.1, UT+0.9] ms  vs  固定窗 [UT−0.1, UT+2.0] ms")
    for label, src, m in (("最佳格", pool, P),
                          ("固定 1.0 ms", fixed[2], np.ones(fixed[2]["t0"].size, bool)),
                          ("固定 2.1 ms", fixed[3], np.ones(fixed[3]["t0"].size, bool)),
                          ("目录窗", fixed[1], np.ones(fixed[1]["t0"].size, bool))):
        n = src["n_core"][m]
        good = np.isfinite(src["mf"][m])
        if good.sum() == 0:
            print(f"  {label:12s} 窗内无事例")
            continue
        print(f"  {label:12s} n={int(m.sum()):3d}  窗内计数中位 {np.median(n):5.1f}  "
              f"mf 中位 {np.nanmedian(src['mf'][m]):.4f}  "
              f"f₃>0 {pct(int(np.nansum(src['f3'][m] > 0)), int(good.sum()))}  "
              f"mf ≤ 0.3 {pct(int(np.nansum(src['mf'][m] <= 0.3)), int(good.sum()))}")
    print("  若最佳格与两个固定窗给出同一结论，说明同戳量不是最佳格选择造出来的；"
          "不一致本身就是信息——那意味着最佳格系统性落在同戳簇上。")


if __name__ == "__main__":
    main()
