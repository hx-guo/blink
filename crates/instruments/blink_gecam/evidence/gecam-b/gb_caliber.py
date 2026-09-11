"""GECAM-B：双增益去重换口径之后，`mf` / `f₂` / `f₃` 往哪个方向偏、偏多少。

旧口径 `e34f560` 按「同探头**同时戳**」并，新口径 `3c017a2` 按「同探头、死时间内、
一高一低」并。实测旧口径只抓到全部对的 85.8–88.4%，漏掉的那 11.6–14.2% 相距约
100 ns（第 32 条）。

**两个簇口径必须分开量，否则比不了**：`tol = 0`（同戳）与 `tol = 150 ns`（容差单链接）
对这批漏并事例的反应完全不同——约 100 ns 落在前者之外、后者之内。所以每个窗算四套：
{新口径, 旧口径} × {同戳, 容差}。

**先前在这里写"分母涨、分子不涨 ⇒ 三个量被压低"是错的，实测方向相反**（旧口径把
`mf` 抬 2.0%、`f₃` 抬 3.7–4.8%），而且**抬升倍数在窗宽跨四个数量级上几乎不变**，
所以也不是"偶然撞同戳"（那个概率 `(n−1)·q/W` 在同一区间里变了三个数量级）。
真正的机制是：约 100 ns 是**固定**延迟，所以一次跨探头符合的高增益孪生条会落在
**彼此相同**的时戳上，在旧口径下形成真簇的一个影子副本。

窗一律取最佳格 `[start + delay, start + delay + bin_size_best]`，两个口径用**同一个窗**，
所以差出来的就是去重口径本身。

用法: python3 gb_caliber.py <小时清单> <输出目录> <worker> <workers>
"""

import json
import os
import sys

import numpy as np

import gb_feat as gf

TOL = 150e-9   # 与第 22 条的 τ 同值：容差簇口径


def dedupe_same_timestamp(t, g):
    """旧口径 `e34f560`：同探头**同时戳**、增益档不同则并，丢高增益（`gain_type` 小的）。
    返回"丢弃"的布尔掩模。t 已按时间排好。"""
    drop = np.zeros(t.size, bool)
    if t.size < 2:
        return drop
    pair = (t[1:] == t[:-1]) & (g[1:] != g[:-1])
    a = np.flatnonzero(pair)
    # 连续的同戳串里，一条只能配一次：从左往右贪心，配上的两条都退出
    used = np.zeros(t.size, bool)
    for i in a:
        if used[i] or used[i + 1]:
            continue
        used[i] = used[i + 1] = True
        drop[i if g[i] < g[i + 1] else i + 1] = True
    return drop


def read_grd_two_calibers(path):
    """一小时 GRD，同时给出新旧两套去重掩模。除去重那一步，其余与 `gb_feat.read_grd` 同。"""
    from astropy.io import fits
    times, dets, keep_new, keep_old = [], [], [], []
    with fits.open(path, memmap=True) as hdus:
        gti = gf.read_gti(hdus)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            pi = np.asarray(data["PI"])
            sel = ((np.asarray(data["EVT_TYPE"]) == gf.NORMAL_EVT_TYPE)
                   & (pi >= gf.MIN_CHANNEL) & (pi < gf.OVERFLOW_CHANNEL))
            t = np.asarray(data["TIME"], float)[sel]
            inside = gf.inside_gti(t, gti)
            t = t[inside]
            g = np.asarray(data["GAIN_TYPE"])[sel][inside].astype(np.int8)
            dead = np.asarray(data["DEAD_TIME"], np.float32)[sel][inside]
            order = np.argsort(t, kind="stable")
            t, g, dead = t[order], g[order], dead[order]
            keep_new.append(~gf.dedupe_one_detector(t, g, dead))
            keep_old.append(~dedupe_same_timestamp(t, g))
            times.append(t)
            dets.append(np.full(t.size, int(hdu.name[-2:]), np.int8))
    if not times or sum(x.size for x in times) == 0:
        return None
    time = np.concatenate(times)
    det = np.concatenate(dets)
    order = np.lexsort((det, time))
    return {"time": time[order], "det": det[order],
            "new": np.concatenate(keep_new)[order],
            "old": np.concatenate(keep_old)[order], "gti": gti}


def cluster_arrays(time, det, tol=0.0):
    """分簇的向量化版：返回每簇的事例数 m 与探头数 d。

    `tol = 0` 是"同戳"（时戳精确相等），`tol > 0` 是单链接容差簇。
    **这两个不是同一个量，比较时必须对齐**——约 100 ns 的支路延迟落在
    `tol = 0` 的外面、`tol = 150 ns` 的里面，所以换口径的影响在两种簇上
    完全不同（见第 32 条）。

    `np.split` 那版在 250 万个窗上要跑几十分钟。事例已按 (时间, 探头) 排好，
    所以簇内探头也是升序的（`tol = 0` 时），"换探头"只要看相邻两条是否同探头；
    `tol > 0` 时簇内探头未必有序，改用逐簇 unique 的等价向量化写法。
    """
    n = time.size
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    head = np.empty(n, bool)
    head[0] = True
    head[1:] = (time[1:] - time[:-1]) > tol if tol > 0 else time[1:] != time[:-1]
    cid = np.cumsum(head) - 1
    m = np.bincount(cid)
    if tol > 0:
        # 簇内探头无序：按 (簇, 探头) 去重再数
        key = cid.astype(np.int64) * 32 + det.astype(np.int64)
        uniq = np.unique(key)
        d = np.bincount(uniq // 32, minlength=m.size)
    else:
        new_det = head.copy()
        new_det[1:] |= det[1:] != det[:-1]
        d = np.bincount(cid, weights=new_det.astype(np.float64)).astype(np.int64)
    return m, d


def stats(time, det, tol=0.0):
    n = time.size
    if n == 0:
        return dict(n=0, mf=np.nan, f2=np.nan, f3=np.nan, f3pos=0, maxd=0)
    m, d = cluster_arrays(time, det, tol)
    return dict(n=int(n),
                mf=float(m[m >= 2].sum()) / n,
                f2=float(d[d >= 2].sum()) / n,
                f3=float(d[d >= 3].sum()) / n,
                f3pos=int((d >= 3).any()),
                maxd=int(d.max()))


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
        ev = read_grd_two_calibers(path)
        if ev is None:
            print(f"{iso}: 事例为空", flush=True)
            continue
        day = iso[:10]
        sig_path = (f"{gf.SIGNAL_ROOT}/{day[:4]}/{day[5:7]}/"
                    f"{day.replace('-', '')}_signals.json")
        if not os.path.exists(sig_path):
            print(f"{iso}: 没有候选文件", flush=True)
            continue
        signals = [s for s in json.load(open(sig_path)) if s["start"][:13] == iso]
        time = ev["time"]
        keys = ["t0", "bin_s", "fa", "count"]
        for cal in ("new", "old"):
            for suf in ("", "t"):          # "" = 同戳 tol=0，"t" = 容差 TOL 单链接
                keys += [f"{x}{suf}_{cal}" for x in ("n", "mf", "f2", "f3", "f3pos", "maxd")]
        cols = {k: [] for k in keys}
        for s in signals:
            t0 = gf.met(s["start"]) + s["delay"]
            t1 = t0 + s["bin_size_best"]
            lo = int(np.searchsorted(time, t0, "left"))
            hi = int(np.searchsorted(time, t1, "right"))
            sl = slice(lo, hi)
            wt, wd = time[sl], ev["det"][sl]
            cols["t0"].append(t0)
            cols["bin_s"].append(s["bin_size_best"])
            cols["fa"].append(s["false_positive_per_year"])
            cols["count"].append(s["count"])
            for cal in ("new", "old"):
                kt, kd = wt[ev[cal][sl]], wd[ev[cal][sl]]
                for suf, tol in (("", 0.0), ("t", TOL)):
                    for k, v in stats(kt, kd, tol).items():
                        cols[f"{k}{suf}_{cal}"].append(v)
        out = f"{outdir}/cal_{iso.replace('-', '').replace('T', '_')}.npz"
        np.savez_compressed(out, **{k: np.array(v, float) for k, v in cols.items()},
                            n_event_new=np.array([int(ev["new"].sum())]),
                            n_event_old=np.array([int(ev["old"].sum())]),
                            n_event_raw=np.array([int(time.size)]))
        extra = (ev["old"].sum() - ev["new"].sum()) / max(ev["new"].sum(), 1)
        print(f"{iso}: 候选 {len(signals)}，整小时准入后 新 {ev['new'].sum()} / "
              f"旧 {ev['old'].sum()}（旧口径多留 {extra * 100:.3f}%）", flush=True)


if __name__ == "__main__":
    main()
