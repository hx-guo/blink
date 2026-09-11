"""GECAM-A：逐候选重算窗内量，事例流口径与 Rust 侧 `search` 逐条对齐。

**先对账再下结论。** 这里同时对两个量：
  * `count` —— 最佳格 `[start+delay, start+delay+bin_size_best]` 闭区间内的事例数；
  * `mean` —— 本底窗计数 `(±0.5 s) − (±5 ms)` 乘 `bin / (活时长差)`。
`mean` 这一路把 GTI 活时长也一起验了，比只对 count 硬。两个量都逐条命中才往下走。

事例流四步，与 `search()` 同序：
  1. `Event::keep`：`EVT_TYPE == 1` 且 `54 <= PI < 448`；
  2. GTI 过滤（闭区间）；
  3. 按时间排序；
  4. `dedupe_gain_pairs`：死时间（事例自带 `DEAD_TIME`，µs）内同探头**一高一低**
     并成一条，留 `GAIN_TYPE` 大的那条（低增益），**配过的两条都不再参与配对**。

Rust 的外层循环走全局时间序，但只配同探头的对，所以逐探头贪心与它逐条等价。
先前 `recut.py` 只把相邻对标掉、没有「配过就不再配」这个状态、也没上 GTI，
对账只有 98.43%。

顺带在同一趟里量：
  * 同戳簇（`f2`/`f3`/`f4`/`max_d`）与它们的偶然期望（`q` 逐候选取 `np.spacing`）；
  * 逐小时逐探头逐增益档的满量程道 `edge` 与堆积包起点 `cut`（从数据本身求，不查表）；
  * 按 `cut` / `edge−5` / `edge−10` / `edge−15` 四种上界重切之后的窗内计数与本底计数。

用法: python3 ga_window.py <signals.json> <输出 CSV> [小时前缀 如 2024-01-11T03]
"""

import csv
import datetime as dt
import glob
import json
import os
import sys
from math import comb

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
MIN_CHANNEL, OVERFLOW, NORMAL = 54, 448, 1
NEIGHBOR_HALF = 0.5          # 搜索本底窗半宽（SEARCH_NEIGHBOR_SECONDS / 2）
HOLLOW_HALF = 0.005          # 搜索空窗半宽（SEARCH_HOLLOW_MILLISECONDS / 2）
MARGINS = (5, 10, 15)        # 固定余量，与自适应 cut 并排报
# 同戳簇的容差（秒）。0 = 时戳精确相等。
#
# **A 星上「精确相等」是不够的。** 一次带电粒子穿越点亮多路 GRD，各路的时戳不
# 完全相同：双增益对里实测有整整一批差 89 ns 与 119 ns（正好 3 个和 4 个量化步），
# 跨探头同样有传播与成形的延迟。q = 29.8 ns 时，一个粒子的 7 路响应会落在 5 个
# 不同的格子上，`f₂`/`f₃` 按精确相等算就**系统性漏掉**这个签名。所以逐候选按
# 几个容差各算一遍，把整条曲线报出来，不挑一个。
TOLERANCES = (0.0, 60e-9, 150e-9, 300e-9)


def met(iso):
    """ISO → MET 秒（f64），逐位还原 Rust 侧那个 f64。

    **不能先折成整数纳秒再乘 1e−9**：MET 约 1.6e8 s，纳秒整数是 1.6e17，超过
    2⁵³，转 f64 时先被舍到 32 ns 的格子上，再乘一次又舍一次，合计可以差一个
    ulp（29.8 ns）——而 A 星的 `bin_size_best` 中位只有 150 ns，差一个 ulp 就
    整段丢事例。正确做法是整秒与纳秒分开：整秒是小整数、精确；`ns * 1e-9` 只
    有一次舍入，再与整秒相加落回同一个 f64 格子。
    `to_utc()` 把 f64 截断到纳秒（实测 …961.704 → …961），截断误差 0.7 ns 远
    小于半个 ulp 14.9 ns，所以这样加回来正好是原来那个 f64。
    """
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    whole = int(round((stamp - EPOCH).total_seconds()))
    nano = int((frac + "000000000")[:9]) if frac else 0
    return float(whole) + nano * 1e-9


def hour_file(iso):
    """A 星归档：子目录小写，文件名有 `_HH_` 与 `_HHMMSS_` 两种写法。"""
    day = iso[:10].replace("-", "/")
    directory = f"{ROOT}/{day}/GECAM_A/GRD_evt"
    stem = f"gag_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}"
    files = sorted(f for f in glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


def bump(hist):
    """返回 (edge, cut, continuum)。

    `edge` = 最后一个 ≥10 计数的道，即该(探头,增益档)的满量程道。
    `continuum` = `hist[edge-80:edge-40]` 的中位，取在还在正常幂律下降段上的地方。
    `cut` = 超量程堆积包的起点：在 `[edge-80, edge]` 里找连续 ≥3×continuum 的
    最后一段（段内允许 ≤3 道的小缺口），取它的起点。看不出包就返回 −1（不切）。

    两道护栏，缺一条就会误切整档：
    * **本底段不能伸到能阈以下。** 高增益档 `edge ≈ 118`，`edge−80 = 38` 落在
      ch54 以下的恒零区，中位被零拖到近 0，于是「≥3×continuum」在 ch54 就成立、
      `cut` 返回 54 = 整档全切。实测 2024-01-11 h03 有 3 路高增益踩到这一条，
      白白多切 2.9% 的事例。所以下界一律夹到 `MIN_CHANNEL`。
    * **包必须挨着上限。** 离 `edge` 超过 25 道的「包」不是超量程堆积，判为没有包。
    """
    nz = np.flatnonzero(hist >= 10)
    if nz.size == 0:
        return -1, -1, 0.0
    edge = int(nz[-1])
    lo, hi = max(MIN_CHANNEL, edge - 80), max(MIN_CHANNEL + 1, edge - 40)
    if hi <= lo or edge <= lo:
        return edge, -1, 0.0
    cont = float(np.median(hist[lo:hi])) if hi > lo else 0.0
    if cont < 5:
        return edge, -1, cont
    seg = hist[lo:edge + 1]
    idx = np.flatnonzero(seg >= 3 * cont)
    if idx.size == 0:
        return edge, -1, cont
    gaps = np.flatnonzero(np.diff(idx) > 3)
    start = int(lo + (idx[gaps[-1] + 1] if gaps.size else idx[0]))
    if start < edge - 25:
        return edge, -1, cont
    return edge, start, cont


def dedupe(time, gain, dead):
    """`dedupe_gain_pairs` 的逐探头贪心复刻，返回「留下」的布尔掩模。

    链：相邻间隔 ≤ 死时间的极大连续段。链外的事例不可能配对。
    链长 1 直接留；链长 2 且两档不同则丢 `GAIN_TYPE` 小的那条（留低增益）；
    链长 ≥3 才走逐条状态机（实测占比千分之几）。
    """
    n = time.size
    keep = np.ones(n, bool)
    if n < 2:
        return keep
    step = float(dead.max())
    heads = np.flatnonzero(np.diff(time) > step) + 1
    bounds = np.concatenate(([0], heads, [n]))
    size = np.diff(bounds)

    two = np.flatnonzero(size == 2)
    if two.size:
        a = bounds[two]
        differ = gain[a] != gain[a + 1]
        drop = np.where(gain[a] > gain[a + 1], a + 1, a)[differ]
        keep[drop] = False

    for k in np.flatnonzero(size >= 3):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        state = np.zeros(hi - lo, np.int8)          # 0 alive, 1 paired, 2 dropped
        for a in range(lo, hi):
            if state[a - lo]:
                continue
            deadline = time[a] + dead[a]
            b = a + 1
            while b < hi and time[b] <= deadline:
                if state[b - lo] == 0 and gain[b] != gain[a]:
                    held, gone = (a, b) if gain[a] > gain[b] else (b, a)
                    state[held - lo], state[gone - lo] = 1, 2
                    keep[gone] = False
                    break
                b += 1
    return keep


def load_hour(path):
    """按 Rust 口径载入一小时，返回排好序的事例流、GTI 段、逐(探头,增益)的上限表。"""
    parts_t, parts_p, parts_d, parts_g = [], [], [], []
    edges = {}
    with fits.open(path, memmap=True) as hdus:
        gti = hdus["GTI"].data
        segments = np.stack([np.asarray(gti["START"], float), np.asarray(gti["STOP"], float)], 1)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            det = int(hdu.name[-2:])
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            pi = np.asarray(data["PI"], np.int32)
            keep = (np.asarray(data["EVT_TYPE"], np.int32) == NORMAL) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW)
            t = np.asarray(data["TIME"], float)[keep]
            if t.size == 0:
                continue
            pi = pi[keep]
            gain = np.asarray(data["GAIN_TYPE"], np.int32)[keep]
            dead = np.asarray(data["DEAD_TIME"], float)[keep] * 1e-6
            inside = np.zeros(t.size, bool)
            for lo, hi in segments:
                inside |= (t >= lo) & (t <= hi)
            t, pi, gain, dead = t[inside], pi[inside], gain[inside], dead[inside]
            if t.size == 0:
                continue
            order = np.argsort(t, kind="stable")
            t, pi, gain, dead = t[order], pi[order], gain[order], dead[order]
            alive = dedupe(t, gain, dead)
            t, pi, gain = t[alive], pi[alive], gain[alive]
            for g in (0, 1):
                sel = gain == g
                if not sel.any():
                    edges[(det, g)] = (-1, -1, 0.0, 0)
                    continue
                hist = np.bincount(pi[sel], minlength=OVERFLOW)
                e, c, cont = bump(hist)
                edges[(det, g)] = (e, c, cont, int(sel.sum()))
            parts_t.append(t)
            parts_p.append(pi.astype(np.int16))
            parts_d.append(np.full(t.size, det, np.int8))
            parts_g.append(gain.astype(np.int8))
    time = np.concatenate(parts_t)
    order = np.argsort(time, kind="stable")
    return (time[order], np.concatenate(parts_p)[order], np.concatenate(parts_d)[order],
            np.concatenate(parts_g)[order], segments, edges)


def right_end(time, ref, half):
    """快照窗右端（开区间下标）。判据与 Rust 一致，是 `t − ref < half` 而不是
    `t < ref + half`——`ref + half` 那一步自己会舍入，边沿上正好有事例时两者
    差一条，实测 7149 个候选里踩到 1 次。先二分给个近似位置，再逐条修边。"""
    i = int(np.searchsorted(time, ref + half, "left"))
    while i > 0 and not time[i - 1] - ref < half:
        i -= 1
    while i < time.size and time[i] - ref < half:
        i += 1
    return i


def left_begin(time, ref, half, head):
    """快照窗左端（闭区间下标）。Rust 前进的判据是 `ref − t[start+1] > half`，
    停下来时 `start` 这一条还在窗里、而它已经在半宽之外，所以左端**多含一条**。
    左端同时夹到 `head`（chunk 起点对应的下标），见下面的注释。"""
    i = int(np.searchsorted(time, ref - half, "left"))
    while i > 0 and ref - time[i - 1] <= half:
        i -= 1
    while i < time.size and not ref - time[i] <= half:
        i += 1
    return max(head, i - 1)


def live(segments, lo, hi, span):
    """GTI 与 [lo, hi] 的交集长度，再夹到 chunk 边界，与 `live_length` 同口径。"""
    a = np.maximum(lo, np.maximum(segments[:, 0], span[0]))
    b = np.minimum(hi, np.minimum(segments[:, 1], span[1]))
    return float(np.maximum(b - a, 0.0).sum())


def clusters(times, detectors, tol):
    """按容差单链接分簇，返回每条事例所在簇的**探头数**。tol = 0 即时戳精确相等。

    数探头数而不是事例数：「一个粒子同时点亮 ≥3 路」才是粒子签名，同一路里的
    两条（同档未并的对，实测 0.18%）不算。tol = 0 时两者本来就一样——同探头同
    时戳的已经在 `dedupe` 里并掉了。
    """
    if times.size == 0:
        return np.zeros(0, int)
    order = np.argsort(times, kind="stable")
    st, sd = times[order], detectors[order].astype(np.int64)
    if tol <= 0:
        heads = np.flatnonzero(np.diff(st) != 0) + 1
    else:
        heads = np.flatnonzero(np.diff(st) > tol) + 1
    label = np.zeros(times.size, np.int64)
    label[heads] = 1
    label = np.cumsum(label)
    per_cluster = np.bincount(np.unique(label * 256 + sd) // 256, minlength=int(label[-1]) + 1)
    out = np.empty(times.size, int)
    out[order] = per_cluster[label]
    return out


def ceiling_masks(pi, det, gain, edges):
    """逐事例的「低于上界」掩模，四种上界各一份。上界 −1 表示这一档看不出包、不切。"""
    table = {}
    for name in ("cut",) + tuple(f"m{m}" for m in MARGINS):
        limit = np.full((26, 2), OVERFLOW, np.int32)
        for (d, g), (edge, cut, _cont, _n) in edges.items():
            if name == "cut":
                limit[d, g] = cut if cut > 0 else OVERFLOW
            else:
                margin = int(name[1:])
                limit[d, g] = (edge - margin) if edge > 0 else OVERFLOW
        table[name] = pi < limit[det, gain]
    return table


def main():
    sig_path, out_path = sys.argv[1], sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else None

    signals = json.load(open(sig_path))
    if want:
        signals = [s for s in signals if s["start"].startswith(want)]
    signals.sort(key=lambda s: s["start"])
    print(f"候选 {len(signals)} 个", flush=True)

    names = ("cut",) + tuple(f"m{m}" for m in MARGINS)
    fh = open(out_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(
        ["start", "hour", "fa", "sf", "count", "mean", "bin_ns", "lon", "lat",
         "n_core", "mean_chk", "n_det", "f2", "f3", "f4", "n_trip", "max_d",
         "q_ns", "exp_pair", "exp_trip", "det_frac_max", "pi_med", "pi_max",
         "n_bkg", "n_hi", "n_m2", "n_m2_hi", "f2_cut", "f3_cut", "max_d_cut", "n_det_cut"]
        + [f"n_{x}" for x in names] + [f"b_{x}" for x in names]
        + [f"{c}_t{int(t * 1e9)}" for t in TOLERANCES for c in ("f2", "f3", "maxd", "exp3")]
    )

    key = None
    stream = None
    hit_count = hit_mean = written = 0
    for signal in signals:
        iso = signal["start"]
        if iso[:13] != key:
            key = iso[:13]
            path = hour_file(iso)
            print(f"  载入 {key} {os.path.basename(path)}", flush=True)
            time, pi, det, gain, segments, edges = load_hour(path)
            masks = ceiling_masks(pi, det, gain, edges)
            cums = {x: np.concatenate(([0], np.cumsum(masks[x], dtype=np.int64))) for x in names}
            bumped = ~masks["cut"]
            cum_bump = np.concatenate(([0], np.cumsum(bumped, dtype=np.int64)))
            hour_met = met(iso[:13] + ":00:00")
            span = (hour_met, hour_met + 3600.0)
            # `search_new` 的 cursor 从「第一个 ≥ chunk 起点的事例」起步，两个
            # 本底快照窗都初始化在 cursor 上、只向前拖，**不会回看 chunk 起点
            # 之前的事例**。A 星的小时文件常越过整点往前伸（本例 GTI 起点在整点
            # 前 100 s），所以落在头 0.5 s 里的候选，其本底计数的左端被钉在
            # chunk 起点上；活时长同样夹在 chunk 起点，分子分母一致。重算必须
            # 照做，否则这批候选的 mean 会偏高近两倍。
            lows = [f"{d}/{g}:{v[0]},{v[1]},{v[3]}" for (d, g), v in sorted(edges.items())]
            print("    edge/cut/n 逐(探头,档): " + " ".join(lows), flush=True)
            head = int(np.searchsorted(time, span[0], "left"))
            stream = (time, pi, det, gain, segments, span, head)
        time, pi, det, gain, segments, span, head = stream

        t0 = met(iso) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]
        i0 = int(np.searchsorted(time, t0, "left"))
        i1 = int(np.searchsorted(time, t1, "right"))
        n_core = i1 - i0
        if n_core == 0:
            continue
        if n_core == signal["count"]:
            hit_count += 1

        # 本底：索引窗 [t0−0.5, t1+0.5) 减 [t0−0.005, t1+0.005)，与 search_new 同口径。
        # 两个左端各**多含一个事例**：快照窗前进的判据是
        # `t_cursor − t[start+1] > 半宽`，停下来的时候 `start` 这一条还留在窗里，
        # 而它本身已经在半宽之外。两边同时多一条，`mean − hollow` 通常正好抵消，
        # 只有左端顶到 chunk 起点、mean 那侧退无可退时才差一条——实测 7149 个候选
        # 里有 5 个（都在小时头 0.5 s 内）。左端一律夹到 `head`。
        m0 = left_begin(time, t0, NEIGHBOR_HALF, head)
        m1 = right_end(time, t1, NEIGHBOR_HALF)
        h0 = left_begin(time, t0, HOLLOW_HALF, head)
        h1 = right_end(time, t1, HOLLOW_HALF)
        n_bkg = (m1 - m0) - (h1 - h0)
        pure = live(segments, t0 - NEIGHBOR_HALF, t1 + NEIGHBOR_HALF, span) \
            - live(segments, t0 - HOLLOW_HALF, t1 + HOLLOW_HALF, span)
        width = signal["bin_size_best"]
        mean_chk = n_bkg * (width / pure) if pure > 0 else float("nan")
        if signal["mean"] > 0 and abs(mean_chk - signal["mean"]) <= 1e-9 * max(signal["mean"], 1e-9):
            hit_mean += 1

        q = float(np.spacing(t0))
        ct, cp, cd = time[i0:i1], pi[i0:i1], det[i0:i1]
        uniq, inverse, sizes = np.unique(ct, return_counts=True, return_inverse=True)
        f2 = float(sizes[sizes >= 2].sum()) / n_core
        f3 = float(sizes[sizes >= 3].sum()) / n_core
        f4 = float(sizes[sizes >= 4].sum()) / n_core
        n_trip = int((sizes >= 3).sum())
        max_d = int(sizes.max())
        _, per_det = np.unique(cd, return_counts=True)
        # 事例级 2×2：窗内事例「在不在 ≥2 重同戳簇里」× 「在不在超量程包里」。
        # 候选级的 bump_frac 与 mf 只看得见两个占比，看不出是不是同一批事例。
        above = bumped[i0:i1]
        in_m2 = sizes[inverse] >= 2
        n_hi, n_m2, n_m2_hi = int(above.sum()), int(in_m2.sum()), int((above & in_m2).sum())
        # 准入上界改对之后再算同戳量：组合判据是有先后的，先修口径再上判据
        alive = ~above
        tol_cols = []
        m = int(alive.sum())
        for tol in TOLERANCES:
            if m == 0:
                tol_cols += ["0.0000", "0.0000", 0, "inf"]
                continue
            size = clusters(ct[alive], cd[alive], tol)
            tol_cols.append(f"{float((size >= 2).sum()) / m:.4f}")
            tol_cols.append(f"{float((size >= 3).sum()) / m:.4f}")
            tol_cols.append(int(size.max()))
            # 三重偶然期望：n 个事例落在窗 W 里，任三个两两在 tol 内的期望簇数
            # ~C(n,3)·(2·tol/W)²；tol = 0 时退回量化格 q。
            reach = (2.0 * tol) if tol > 0 else q
            tol_cols.append(f"{comb(m, 3) * (reach / width) ** 2 if width > 0 else float('inf'):.4g}")
        if m:
            size0 = clusters(ct[alive], cd[alive], 0.0)
            f2c = float((size0 >= 2).sum()) / m
            f3c = float((size0 >= 3).sum()) / m
            maxdc, ndetc = int(size0.max()), int(np.unique(cd[alive]).size)
        else:
            f2c = f3c = 0.0
            maxdc = ndetc = 0

        slots = width / q if width > 0 else 1.0
        exp_pair = comb(n_core, 2) / slots if slots > 0 else float("inf")
        exp_trip = comb(n_core, 3) / slots ** 2 if slots > 0 else float("inf")

        row = [iso[:23], iso[11:13], f"{signal['false_positive_per_year']:.4e}",
               f"{signal['sf']:.4e}", signal["count"], f"{signal['mean']:.8g}",
               f"{width * 1e9:.2f}", f"{signal['position']['longitude']:.3f}",
               f"{signal['position']['latitude']:.3f}", n_core, f"{mean_chk:.8g}",
               len(per_det), f"{f2:.4f}", f"{f3:.4f}", f"{f4:.4f}", n_trip, max_d,
               f"{q * 1e9:.4f}", f"{exp_pair:.4g}", f"{exp_trip:.4g}",
               f"{per_det.max() / n_core:.4f}", int(np.median(cp)), int(cp.max()),
               n_bkg, n_hi, n_m2, n_m2_hi,
               f"{f2c:.4f}", f"{f3c:.4f}", maxdc, ndetc]
        for x in names:
            row.append(int(cums[x][i1] - cums[x][i0]))
        for x in names:
            row.append(int((cums[x][m1] - cums[x][m0]) - (cums[x][h1] - cums[x][h0])))
        row += tol_cols
        writer.writerow(row)
        written += 1

    fh.close()
    print(f"写出 {written}")
    print(f"对账 count: {hit_count}/{written} = {hit_count / max(written, 1) * 100:.2f}%")
    print(f"对账 mean : {hit_mean}/{written} = {hit_mean / max(written, 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
