"""GECAM-B：138 个"已发表 TGF 所在小时"的逐候选特征，一次过完事例流。

**为什么重算而不是用落盘的 `acd` / `detectors` 字段。** Rust 侧那两个字段的基线窗
是标称 2 s、不夹 GTI（见 `chunk::search::CPD_BASELINE_SECONDS` 的注释），拿它当
分母会在 GTI 边缘系统性低估本底率、跟着高估"观测/期望"。这里分子分母都夹 GTI。

**双增益去重必须照搬 Rust 的死时间判据，不能用"时戳相等"。** 二进制从 `e34f560`
（时戳相等）换到 `3c017a2`（死时间内、一高一低）之后，多并掉的那一批事例正落在
候选窗尺度上。判据是**同探头、b 在 (a, t_a + dead_time_a] 内、增益档不同**，保留
`gain_type` 大的那条（低增益，量程撑到 ch≈379 而高增益 ch≈157 就到顶），配对成功
后两条都退出配对池。去重是**逐探头独立**的——配对要求同探头，所以按探头分组做，
组内再按"相邻间隔 > 该探头最大死时间"切段，段间不可能配对。

事例顺序也要与 Rust 一致：`EvtFile::into_iter` 是按时间的 k 路归并、同时戳按探头
下标升序（`BinaryHeap<Reverse<(Event, index)>>`，`Event: Ord` 只比时间），之后
`events.sort()` 是稳定排序。numpy 侧等价写法是 `np.lexsort((det, time))`，数组事先
按 EVENTS01..25 顺序拼好。

对账纪律：重算的 `n_core` 必须与搜索报的 `count` 逐条相等，不到 100% 先修窗口。
窗口一律取最佳格 `[start + delay, start + delay + bin_size_best]`；`met()` 的小数秒
单独加，不让 `datetime` 截到微秒（截一次就差几十个窗宽）。

用法: python3 gb_feat.py <小时清单> <输出目录> <worker> <workers>
小时清单每行一个 `YYYY-MM-DDTHH`。
"""

import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ARCHIVE = os.environ.get("GECAM_ARCHIVE", "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily")
EPOCH = (2019, 1, 1)  # GECAM-A/B 的 MET 历元
SIGNAL_ROOT = os.environ.get("GB_SIGNALS", "/scratchfs2/gecam/guohx/gecambrun/b2/data/GECAM-B")
CATALOG = os.environ.get("GB_CATALOG", "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")

MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE = 54, 448, 1
# 第 19 条那个超量程堆积包的下沿（A 星实测低增益 edge 中位 379、包在 ch368–383）
PILEUP_CHANNEL = 368
# B 星实测：2022 年起低增益 edge 就在能量梯顶道 ch447，满量程沉积堆成**单道尖峰**
# （2024-01-11 实测 ch447 = 372,911 而 ch446 = 3,194，117 倍），而 `PI < 448` 正好
# 把这一道放进来。所以 B 星要单独数这一道，ch368 那个口径是 A 星的形态。
TOP_CHANNEL = 447
CPD_HALF_WIDTHS = (1e-5, 1e-4, 1e-3, 1e-2)
BASELINE, GUARD = 1.0, 0.01  # CPD 本底窗半宽 / 紧贴候选窗扣掉的保护带
TOLERANCE = 1e-7  # 同戳容差版用 100 ns


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def hour_file(iso, kind):
    """归档里这一小时的最高版本。两种时间字段写法（`_07_` 与 `_070000_`）都认，
    版本号单独解析出来比大小，与 `io/file.rs::version_of` 同规则。"""
    sub, prefix = {"grd": ("GRD_evt", "gbg_evt"), "cpd": ("CPD_evt", "gbc_evt")}[kind]
    directory = f"{ARCHIVE}/{iso[:4]}/{iso[5:7]}/{iso[8:10]}/GECAM_B/{sub}"
    stem = f"{prefix}_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}"
    best, best_version = None, -1
    for path in glob.glob(f"{directory}/{stem}*.fits"):
        rest = os.path.basename(path)[len(stem):-len(".fits")]
        if not rest.startswith("_v"):
            if len(rest) < 4 or not rest[:4].isdigit():
                continue
            rest = rest[4:]
            if not rest.startswith("_v"):
                continue
        try:
            version = int(rest[2:])
        except ValueError:
            continue
        if version > best_version:
            best, best_version = path, version
    return best


def read_gti(hdus):
    for hdu in hdus:
        if hdu.name == "GTI":
            return (np.asarray(hdu.data["START"], float), np.asarray(hdu.data["STOP"], float))
    return None


def inside_gti(times, gti):
    if gti is None:
        return np.ones(times.size, bool)
    keep = np.zeros(times.size, bool)
    for start, stop in zip(*gti):
        keep |= (times >= start) & (times <= stop)
    return keep


def gti_overlap(a, b, gti):
    if gti is None:
        return max(b - a, 0.0)
    starts, stops = gti
    return float(np.clip(np.minimum(b, stops) - np.maximum(a, starts), 0, None).sum())


def dedupe_one_detector(t, g, dead):
    """一路探头的死时间双增益去重，返回"丢弃"的布尔掩模。t 已按时间排好。"""
    drop = np.zeros(t.size, bool)
    if t.size < 2:
        return drop
    dt_s = np.where(np.isfinite(dead) & (dead > 0), dead.astype(float) * 1e-6, 0.0)
    dt_max = float(dt_s.max())
    if dt_max <= 0:
        dt_max = 0.0
    # 段边界：下一个事例已在上一个的最大死时间之外，段与段之间不可能配对
    cut = np.flatnonzero(t[1:] > t[:-1] + dt_max) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [t.size]))
    sizes = ends - starts

    # 两元段：向量化。配对条件 = 增益档不同 且 后一条落在前一条的死时间内
    two = starts[sizes == 2]
    if two.size:
        a, b = two, two + 1
        ok = (g[a] != g[b]) & (t[b] <= t[a] + dt_s[a])
        a, b = a[ok], b[ok]
        # 丢掉增益档小的那条（高增益），保留低增益
        drop[np.where(g[a] < g[b], a, b)] = True

    # 三元及以上：少见，按 Rust 的顺序贪心逐段做
    for s, e in zip(starts[sizes >= 3], ends[sizes >= 3]):
        alive = np.ones(e - s, bool)
        for i in range(e - s):
            if not alive[i]:
                continue
            deadline = t[s + i] + dt_s[s + i]
            for j in range(i + 1, e - s):
                if t[s + j] > deadline:
                    break
                if alive[j] and g[s + j] != g[s + i]:
                    drop[s + (i if g[s + i] < g[s + j] else j)] = True
                    alive[i] = alive[j] = False
                    break
    return drop


def read_grd(path):
    """一小时 GRD：准入 + GTI 过滤 + 逐探头双增益去重，最后按 (时间, 探头) 排好。"""
    times, channels, detectors, keeps = [], [], [], []
    with fits.open(path, memmap=True) as hdus:
        gti = read_gti(hdus)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            pi = np.asarray(data["PI"])
            keep = (np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            t = np.asarray(data["TIME"], float)[keep]
            inside = inside_gti(t, gti)
            t = t[inside]
            c = pi[keep][inside]
            g = np.asarray(data["GAIN_TYPE"])[keep][inside].astype(np.int8)
            dead = np.asarray(data["DEAD_TIME"], np.float32)[keep][inside]
            # 单路内部可能有量化步级的时间回跳；Rust 侧全局稳定排序后同样是时间序
            order = np.argsort(t, kind="stable")
            t, c, g, dead = t[order], c[order], g[order], dead[order]
            drop = dedupe_one_detector(t, g, dead)
            times.append(t)
            channels.append(c)
            keeps.append(~drop)
            detectors.append(np.full(t.size, int(hdu.name[-2:]), np.int8))
    if not times or sum(x.size for x in times) == 0:
        return None
    time = np.concatenate(times)
    detector = np.concatenate(detectors)
    # 与 Rust 的 k 路归并 + 稳定排序等价：主键时间、次键探头下标
    order = np.lexsort((detector, time))
    return {
        "time": time[order],
        "pi": np.concatenate(channels)[order],
        "det": detector[order],
        "keep": np.concatenate(keeps)[order],
        "gti": gti,
    }


def cluster_stats(time, det, tol=0.0):
    """分簇，返回 (每簇事例数 m, 每簇探头数 d)。tol=0 时按时戳精确相等分。"""
    if time.size == 0:
        return np.array([]), np.array([])
    boundary = np.flatnonzero(np.diff(time) > tol) + 1 if tol > 0 else np.flatnonzero(np.diff(time) != 0) + 1
    groups = np.split(np.arange(time.size), boundary)
    return (np.array([g.size for g in groups]),
            np.array([np.unique(det[g]).size for g in groups]))


def count(sorted_times, a, b):
    return int(np.searchsorted(sorted_times, b, "right") - np.searchsorted(sorted_times, a, "left"))


FIELDS = ["t0", "bin_s", "fa", "count", "mean", "lon", "lat", "alt",
          "n_raw", "n_core", "same_det_extra", "mf", "f2", "f3", "f4",
          "n_trip", "max_d", "f3_tol", "max_d_tol", "exp_pair", "exp_trip", "q_ns",
          "n_det_hit", "det_frac_max", "pi_med", "n_pi368", "pi_bg_med", "n_pi368_bg", "n_bg_grd",
          "n_det_zero_bg", "n_pi447", "n_pi447_bg",
          "cpd_rate", "cpd_nbg", "cpd_tbg"]
FIELDS += [f"obs{i}" for i in range(len(CPD_HALF_WIDTHS))]
FIELDS += [f"exp{i}" for i in range(len(CPD_HALF_WIDTHS))]


def features(ev, cpd_t, cpd_gti, t0, t1, det_sum=None):
    time, pi, det = ev["time"], ev["pi"], ev["det"]
    lo = int(np.searchsorted(time, t0, "left"))
    hi = int(np.searchsorted(time, t1, "right"))
    sl = slice(lo, hi)
    kept = ev["keep"][sl]
    n_raw = hi - lo
    wt, wd, wpi = time[sl][kept], det[sl][kept], pi[sl][kept]
    n_core = wt.size
    row = {f: np.nan for f in FIELDS}
    row["t0"], row["bin_s"] = t0, t1 - t0
    row["n_raw"], row["n_core"], row["same_det_extra"] = n_raw, n_core, n_raw - n_core
    if n_core == 0:
        return row, None

    m, d = cluster_stats(wt, wd)
    row["mf"] = float(m[m >= 2].sum()) / n_core          # 参与任何同戳簇的事例占比
    row["f2"] = float(d[d >= 2].sum()) / n_core
    row["f3"] = float(d[d >= 3].sum()) / n_core          # ≥3 重跨探头同戳的计数占比
    row["f4"] = float(d[d >= 4].sum()) / n_core
    row["n_trip"] = int((d >= 3).sum())
    row["max_d"] = int(d.max())
    _, dt_ = cluster_stats(wt, wd, TOLERANCE)
    row["f3_tol"] = float(dt_[dt_ >= 3].sum()) / n_core
    row["max_d_tol"] = int(dt_.max())

    q = float(np.spacing(t0))
    row["q_ns"] = q * 1e9
    slots = (t1 - t0) / q if t1 > t0 else 1.0
    row["exp_pair"] = n_core * (n_core - 1) / 2 / slots if slots > 0 else np.inf
    row["exp_trip"] = n_core * (n_core - 1) * (n_core - 2) / 6 / slots ** 2 if slots > 0 else np.inf

    ids, counts = np.unique(wd, return_counts=True)
    row["n_det_hit"] = ids.size
    row["det_frac_max"] = counts.max() / n_core
    row["pi_med"] = float(np.median(wpi))
    row["n_pi368"] = int((wpi >= PILEUP_CHANNEL).sum())
    row["n_pi447"] = int((wpi == TOP_CHANNEL).sum())
    detvec = np.zeros(26, np.int64)
    detvec[ids] = counts

    blo = int(np.searchsorted(time, t0 - BASELINE, "left"))
    bhi = int(np.searchsorted(time, t1 + BASELINE, "right"))
    bsl = slice(blo, bhi)
    bt = time[bsl]
    bkeep = ev["keep"][bsl] & ((bt < t0 - GUARD) | (bt > t1 + GUARD))
    bpi, bdet = pi[bsl][bkeep], det[bsl][bkeep]
    row["n_bg_grd"] = int(bpi.size)
    bdetvec = np.zeros(26, np.int64)
    if bpi.size:
        row["pi_bg_med"] = float(np.median(bpi))
        row["n_pi368_bg"] = int((bpi >= PILEUP_CHANNEL).sum())
        row["n_pi447_bg"] = int((bpi == TOP_CHANNEL).sum())
        bids, bcounts = np.unique(bdet, return_counts=True)
        bdetvec[bids] = bcounts
    # 逐路基线零格：GTI 是全仪器概念，25 路里掉一路仪器照样"活着"，
    # 基线向量里的零格是唯一看得见探头级停机的地方（第 4c 条）
    row["n_det_zero_bg"] = int((bdetvec[1:26] == 0).sum())
    if det_sum is not None:
        det_sum[0] += detvec
        det_sum[1] += bdetvec

    if cpd_t is not None and cpd_t.size:
        left = (t0 - BASELINE, t0 - GUARD)
        right = (t1 + GUARD, t1 + BASELINE)
        n_bg = count(cpd_t, *left) + count(cpd_t, *right)
        t_bg = gti_overlap(*left, cpd_gti) + gti_overlap(*right, cpd_gti)
        rate = n_bg / t_bg if t_bg > 0 else np.nan
        row["cpd_rate"], row["cpd_nbg"], row["cpd_tbg"] = rate, n_bg, t_bg
        for i, half in enumerate(CPD_HALF_WIDTHS):
            a, b = t0 - half, t1 + half
            row[f"obs{i}"] = count(cpd_t, a, b)
            row[f"exp{i}"] = rate * gti_overlap(a, b, cpd_gti) if np.isfinite(rate) else np.nan
    return row, (detvec, bdetvec)


def main():
    hours = [line.strip() for line in open(sys.argv[1]) if line.strip()]
    outdir, worker, workers = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(outdir, exist_ok=True)
    mine = [h for i, h in enumerate(hours) if i % workers == worker]
    print(f"worker {worker}/{workers}: {len(mine)} 小时", flush=True)
    catalog = list(csv.DictReader(open(CATALOG)))

    for iso_hour in mine:
        out = os.path.join(outdir, f"feat_{iso_hour.replace('-', '').replace('T', '_')}.npz")
        if os.path.exists(out):
            print("  跳过已有", iso_hour, flush=True)
            continue
        day = iso_hour[:10]
        sig_path = f"{SIGNAL_ROOT}/{day[:4]}/{day[5:7]}/{day.replace('-', '')}_signals.json"
        if not os.path.exists(sig_path):
            print("  没有候选文件", iso_hour, sig_path, flush=True)
            continue
        signals = [s for s in json.load(open(sig_path)) if s["start"][:13] == iso_hour]

        grd_path = hour_file(iso_hour + ":00:00", "grd")
        if grd_path is None:
            print("  没有 GRD 文件", iso_hour, flush=True)
            continue
        ev = read_grd(grd_path)
        if ev is None:
            print("  GRD 为空", iso_hour, flush=True)
            continue
        n_merged = int((~ev["keep"]).sum())

        cpd_path = hour_file(iso_hour + ":00:00", "cpd")
        cpd_t, cpd_gti = None, None
        if cpd_path is not None:
            with fits.open(cpd_path, memmap=True) as hdus:
                cpd_gti = read_gti(hdus)
                parts = []
                for hdu in hdus:
                    if hdu.name.startswith("EVENTS"):
                        data = hdu.data
                        keep = np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE
                        parts.append(np.asarray(data["TIME"], float)[keep])
            if parts:
                allt = np.sort(np.concatenate(parts))
                cpd_t = allt[inside_gti(allt, cpd_gti)]

        rows, kinds, det_pos, det_pos_bg = [], [], [], []
        det_sum = [np.zeros(26, np.int64), np.zeros(26, np.int64)]
        for s in signals:
            t0 = met(s["start"]) + s["delay"]
            row, _ = features(ev, cpd_t, cpd_gti, t0, t0 + s["bin_size_best"], det_sum)
            row["fa"] = s["false_positive_per_year"]
            row["count"] = s["count"]
            row["mean"] = s["mean"]
            row["lon"] = s["position"]["longitude"]
            row["lat"] = s["position"]["latitude"]
            row["alt"] = s["position"].get("altitude", np.nan)
            rows.append(row)
            kinds.append(0)

        # 目录里落在这一小时的已发表 TGF，两个只锚目录 UT 的窗：
        #
        # kind=1 —— [UT, UT + Duration_us]。**实测这个窗系统性偏早**：目录 UT 比
        #   暴发在我们时基里的起点早几十微秒到两毫秒（2021-01-20 那个实测 +54.5 µs，
        #   窗内只捞到 7 个计数而真候选是 66 个），所以它**不能当 TGF 窗用**，
        #   只作口径对照留着。
        # kind=2 —— [UT − 0.1 ms, UT + 0.9 ms] 的固定 1 ms 窗。**构造无关**：窗长
        #   与位置都只由目录 UT 定，不含我们搜索的最佳格选择，所以拿它复测同戳量
        #   可以排除"最佳格优先落在同戳簇上"这个自我实现的嫌疑。上一轮实测偏移
        #   5–95% 是 −0.038 .. +1.82 ms，**这个 1 ms 窗只覆盖约 90%**，代价是漏掉
        #   偏移最大的那一成；好处是背景稀释小（约 10 个本底事例对暴发的几十个）。
        # kind=3 —— [UT − 0.1 ms, UT + 2.0 ms] 的固定 2.1 ms 窗，**覆盖整个偏移分布**，
        #   代价是本底事例多一倍、同戳占比被稀释。两个都算，结论要一致才算稳。
        for c in catalog:
            if c["UT"][:13] != iso_hour:
                continue
            t0 = met(c["UT"])
            row, vecs = features(ev, cpd_t, cpd_gti, t0, t0 + float(c["Duration_us"]) * 1e-6)
            row["count"] = -1
            rows.append(row)
            kinds.append(1)
            det_pos.append(vecs[0] if vecs else np.zeros(26, np.int64))
            det_pos_bg.append(vecs[1] if vecs else np.zeros(26, np.int64))
            row2, _ = features(ev, cpd_t, cpd_gti, t0 - 1e-4, t0 + 9e-4)
            row2["count"] = -1
            rows.append(row2)
            kinds.append(2)
            row3, _ = features(ev, cpd_t, cpd_gti, t0 - 1e-4, t0 + 2e-3)
            row3["count"] = -1
            rows.append(row3)
            kinds.append(3)

        arrays = {f: np.array([r[f] for r in rows], float) for f in FIELDS}
        arrays["kind"] = np.array(kinds, np.int8)
        arrays["det_sum_window"] = det_sum[0]
        arrays["det_sum_baseline"] = det_sum[1]
        arrays["det_pos_window"] = np.array(det_pos, np.int64) if det_pos else np.zeros((0, 26), np.int64)
        arrays["det_pos_baseline"] = np.array(det_pos_bg, np.int64) if det_pos_bg else np.zeros((0, 26), np.int64)
        arrays["n_merged_hour"] = np.array([n_merged])
        arrays["n_event_hour"] = np.array([ev["time"].size])
        arrays["gti_seconds"] = np.array([gti_overlap(-1e18, 1e18, ev["gti"]) if ev["gti"] is not None else np.nan])
        np.savez_compressed(out, **arrays)
        print(f"  {iso_hour}: 候选 {len(signals)}，目录 {sum(kinds)}，事例 {ev['time'].size}，"
              f"并掉 {n_merged} ({n_merged / max(ev['time'].size, 1) * 100:.3f}%)", flush=True)


if __name__ == "__main__":
    main()
