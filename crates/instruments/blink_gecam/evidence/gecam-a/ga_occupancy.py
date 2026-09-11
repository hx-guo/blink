"""A 星本底的占据数分布 ÷ 泊松——搜索用错了零假设，这里把低估的倍数量出来。

搜索的 `fa` 假定本底是泊松的。但 25 路 GRD 上一次宇宙线穿越会同时点亮好几路，
于是"窄窗里挤进 k 个计数"的实际概率远高于泊松。**这个比值就是 `fa` 在阈值处
被低估的倍数**，也是"候选率比配置的假阳性率高四个数量级"的来源。

做法：把一小时的事例流（已按 `Event::keep` 准入、按 GTI 过滤、双增益去重，
与 Rust 侧逐条对齐）按固定宽度 W 铺成格子，数"出现 >= k 个计数的格数"，
与 `Poisson(λ = r·W)` 的解析尾概率比。

**几条口径**：
* **必须在对数空间算**。泊松尾在 k = 12、λ = 1e-3 时是 1e-60 量级，先算成
  浮点再相除会得到 inf/nan，而 nan 传进分位数看着像"算不出来"，其实是把最
  极端那批悄悄丢掉了。
* **报一条随 k 的曲线，不是一个点**；并且必须报"候选 `count` 分布中位处"
  那个 k（A 星是 9），那才是对目录实际生效的数。
* **W 要标，而且要给对 W 的依赖**。A 星候选 `bin_size_best` 中位只有
  0.149 µs，与 B 星取的 10 µs 差两个数量级，两边的数不能直接比。
* **不剔除候选窗**。候选本身就是这些簇——它们不是外来的信号，就是本底过离散
  的那条尾巴，剔掉等于把要量的东西量掉。这几个小时按已发表率期望的真 TGF 数
  是 1e-2 量级，整条流就是本底。为免这句被当成回避，同时给出"剔除候选窗
  之后"的对照。

用法: python3 ga_occupancy.py <YYYY-MM-DD> <signals.json> <输出前缀>
"""

import datetime as dt
import glob
import json
import sys

import numpy as np
from scipy.stats import poisson

ROOT = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_CHANNEL, OVERFLOW, NORMAL = 54, 448, 1
EPOCH = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
LN10 = np.log(10.0)
WIDTHS_US = [0.0298, 0.149, 0.5, 1.0, 10.0, 100.0, 1000.0]
KS = list(range(2, 21))
K_MEDIAN = 9          # A 星候选 count 的中位


def met_exact(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    whole = int((stamp - EPOCH).total_seconds())
    return float(f"{whole}.{frac}" if frac else str(whole))


def hour_files(day):
    d = f"{ROOT}/{day.replace('-', '/')}/GECAM_A/GRD_evt"
    stems = {}
    for f in sorted(glob.glob(f"{d}/gag_evt_*_v*.fits")):
        stems.setdefault(f.rsplit("_v", 1)[0], []).append(f)
    return [sorted(v)[-1] for _, v in sorted(stems.items())]


def dedupe(t, gt, dead):
    """与 `dedupe_gain_pairs` 同规则：保留 gain_type 大的那条（低增益）。"""
    n = t.size
    if n < 2:
        return np.zeros(n, bool)
    cand = np.where((t[1:] <= t[:-1] + dead) & (gt[1:] != gt[:-1]))[0]
    drop = np.zeros(n, bool)
    used = np.zeros(n, bool)
    for i in cand:
        j = i + 1
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        drop[i if gt[i] < gt[j] else j] = True
    return drop


def load_hour(path):
    from astropy.io import fits
    times = []
    with fits.open(path, memmap=True) as hdus:
        gti = hdus["GTI"].data
        lo = np.asarray(gti["START"], float)
        hi = np.asarray(gti["STOP"], float)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            d = hdu.data
            if d is None or len(d) == 0:
                continue
            pi = np.asarray(d["PI"], np.int32)
            et = np.asarray(d["EVT_TYPE"], np.int32)
            gt = np.asarray(d["GAIN_TYPE"], np.int32)
            t = np.asarray(d["TIME"], float)
            dtc = np.asarray(d["DEAD_TIME"], float)
            keep = (et == NORMAL) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW)
            k = np.searchsorted(lo, t, "right") - 1
            keep &= (k >= 0) & (t <= hi[np.clip(k, 0, hi.size - 1)])
            t, gt, dtc = t[keep], gt[keep], dtc[keep]
            if t.size == 0:
                continue
            o = np.argsort(t, kind="stable")
            t, gt, dtc = t[o], gt[o], dtc[o]
            dead = float(dtc[0]) * 1e-6 if np.isfinite(dtc[0]) and dtc[0] > 0 else 0.0
            t = t[~dedupe(t, gt, dead)]
            times.append(t)
    if not times:
        return None
    time = np.concatenate(times)
    time.sort()
    live = float(np.sum(np.clip(hi - lo, 0.0, None)))
    return time, live


def occupancy(rel, w, ncells, drop_cells=None):
    """宽度 w 的格子里的占据数分布。

    只有非空格会出现在 np.unique 里；空格数 = ncells − 非空格数，对 k >= 2 的
    尾巴没有影响，所以不必把它们造出来（W = 0.0298 µs 时 ncells 是 1.2e11，
    造出来会直接撑爆内存）。
    """
    cid = np.floor(rel / w).astype(np.int64)
    uid, cnt = np.unique(cid, return_counts=True)
    if drop_cells is not None and drop_cells.size:
        keep = ~np.isin(uid, drop_cells)
        uid, cnt = uid[keep], cnt[keep]
        ncells = ncells - float(drop_cells.size)
    return np.bincount(cnt), ncells, int(cnt.sum())


def report(tag, occ, ncells, n_ev, w, quiet=False):
    lam = n_ev / ncells
    tail = np.array([occ[k:].sum() if k < occ.size else 0 for k in KS], float)
    # 对数空间：泊松尾在 k=12 时是 1e-60 量级，先化成浮点再除会得到 inf/nan
    log_obs = np.where(tail > 0, np.log10(np.maximum(tail, 1.0)) - np.log10(ncells), np.nan)
    log_poi = poisson.logsf(np.array(KS) - 1, lam) / LN10
    rl = log_obs - log_poi
    print(f"\n[{tag}] W = {w*1e6:g} µs  格数 {ncells:.3e}  事例 {n_ev:,}  λ = {lam:.4g}")
    if not quiet:
        print(f"{'k':>4} {'实测>=k的格数':>14} {'实测尾概率':>12} {'泊松尾概率':>12} {'实测/泊松':>13}")
        for i, k in enumerate(KS):
            if tail[i] <= 0:
                print(f"{k:4d} {0:14d} {'0':>12} {10.0**log_poi[i]:12.3e} {'— 实测已空':>13}")
                continue
            mark = "   <- 候选 count 中位" if k == K_MEDIAN else ""
            print(f"{k:4d} {int(tail[i]):14d} {10.0**log_obs[i]:12.3e} "
                  f"{10.0**log_poi[i]:12.3e} {10.0**rl[i]:13.3e}{mark}")
    return rl, lam


def main():
    day, sig_path, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    signals = json.load(open(sig_path))
    by_hour = {}
    for s in signals:
        by_hour.setdefault(s["start"][11:13], []).append(s)

    out = {"day": day, "k_list": KS, "widths_us": WIDTHS_US, "hours": {}}
    for path in hour_files(day):
        hh = path.rsplit("_", 2)[-2]
        res = load_hour(path)
        if res is None:
            continue
        time, live = res
        t0 = float(time[0])
        rel = time - t0            # 同一 ulp 格上的差，精确
        print(f"\n{'=' * 78}\n小时 {hh}：事例 {time.size:,}  活时间 {live:.0f} s  "
              f"率 {time.size / live:.0f} c/s\n{'=' * 78}", flush=True)
        hrec = {}
        for w_us in WIDTHS_US:
            w = w_us * 1e-6
            occ, nc, n_ev = occupancy(rel, w, live / w)
            rl, lam = report(f"{hh} 全段", occ, nc, n_ev, w)
            hrec[str(w_us)] = {"lambda": lam, "n_ev": n_ev, "ncells": nc,
                               "log10_ratio": {str(k): (None if not np.isfinite(rl[i]) else float(rl[i]))
                                               for i, k in enumerate(KS)}}
        # 对照：把候选最佳格覆盖到的格子整格剔掉
        sig = by_hour.get(hh, [])
        if sig:
            c0 = np.array([met_exact(s["start"]) + s["delay"] - t0 for s in sig])
            c1 = c0 + np.array([s["bin_size_best"] for s in sig])
            for w_us in (0.149, 10.0):
                w = w_us * 1e-6
                cells = np.unique(np.concatenate([np.floor(c0 / w), np.floor(c1 / w)])).astype(np.int64)
                occ, nc, n_ev = occupancy(rel, w, live / w, drop_cells=cells)
                report(f"{hh} 剔除候选窗（{cells.size} 格）", occ, nc, n_ev, w)
        out["hours"][hh] = hrec
        del time, rel

    json.dump(out, open(prefix + "_occ.json", "w"))
    print(f"\n写出 {prefix}_occ.json")


if __name__ == "__main__":
    main()
