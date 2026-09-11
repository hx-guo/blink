"""GECAM-A：逐候选把整条判据链的输入量一次算齐，事例流按 Rust 侧的规则重建。

**窗口是事例定义的、两端闭合。** 搜索里 `duration = data[cursor+step].time() -
data[cursor].time()`、窗内事例数恒等于 `step+1`，所以报出来的 `start + delay`
就是首个事例的时刻、`+ bin_size_best` 就是末个事例的时刻。离线重算必须取
`[t0, t1]` 闭区间，取半开区间会把末个事例漏掉——A 星中位窗只有 5 个 ulp，
漏一个就是漏掉一成的计数。

**时刻用整数 ulp 算，不经 f64 加法。** A 星 2023-04-03 之后 MET > 2^27，
float64 在这里的 ulp 就是 29.8 ns，而 `bin_size_best` 中位只有 149 ns。
把 MET 折成整数纳秒再乘 1e-9（1.6e17 > 2^53）会前后各舍一次、差一个 ulp，
窗口整体错位。归档 TIME 列本身就落在这个 ulp 格子上，所以把所有时刻换算成
"第几个 ulp"的整数，窗口边界用整数加法定。

**事例流要与 Rust 侧逐条对齐**，否则对账停在 98% 而且差的那部分说不清：
  * 先按 `Event::keep` 准入，**再按 GTI 过滤**（`search()` 里两道都做了）；
  * 双增益去重保留的是**低增益那条**（`gain_type` 大的），不是时间靠前那条
    ——一对的两条时戳差 0–119 ns，留错一条就把窗边界挪一个 ulp。

逐候选算出来的量：
  n_obs          窗内事例数（与搜索报的 count 对账）
  n_lo / n_hi    窗左右端点上同戳的事例数——搜索的窗是按下标取的，端点上有并列
                 时它只取到其中一条，时间口径取不到，这两列用来把残差定量圈住
  n_keep         扣掉超量程堆积包之后还剩几个（准入上界）
  f2 / f3        落在 >=2 / >=3 重严格同戳簇里的计数占比
  e_f2 / e_f3    同一候选自己的 (n, W, q) 下的偶然期望——判据要跟它自己的偶然
                 比，不是跟别的候选比
用法: python3 ga_chain.py <YYYY-MM-DD> <signals.json> <输出 csv> [new|old]
"""

import csv
import datetime as dt
import glob
import json
import sys

import numpy as np
from scipy.special import gammaln

ROOT = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_CHANNEL, OVERFLOW, NORMAL = 54, 448, 1
EPOCH = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)


def met_exact(iso):
    """把 ISO 串还原成归档里那个 float64，只舍一次。

    整秒是精确整数，小数部分连成一个十进制串交给 float() 正确舍入；分两步相加
    （total_seconds() + float("0."+frac)）会多舍一次，在几个 ulp 宽的窗上足以
    让窗口整体错位。
    """
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


def bump_start(hist):
    """超量程堆积包的起点，逐探头从数据自己求（与 recut.py 同一套）。"""
    nz = np.where(hist >= 10)[0]
    if nz.size == 0:
        return -1
    edge = int(nz[-1])
    lo, hi = max(0, edge - 80), max(1, edge - 40)
    cont = float(np.median(hist[lo:hi])) if hi > lo else 0.0
    if cont < 5:
        return -1
    seg = hist[lo:edge + 1]
    idx = np.where(seg >= 3 * cont)[0]
    if idx.size == 0:
        return -1
    gaps = np.where(np.diff(idx) > 3)[0]
    return int(lo + (idx[gaps[-1] + 1] if gaps.size else idx[0]))


def dedupe(t, gt, dead):
    """双增益去重，与 `dedupe_gain_pairs` 同规则：保留 `gain_type` 大的那条。

    Rust 侧是"对每个还活着的 a，在 [t_a, t_a+dead] 里找第一条还活着、同探头、
    增益档不同的 b，留 gain_type 大的、丢另一条"。同探头 4 µs 内出现第三条的
    概率可以忽略（实测一小时几百对），所以只看相邻对；但**留哪条按增益判，
    不能按先后判**——留错一条窗边界就差一个 ulp。
    """
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


def load_hour(path, mode="new"):
    """mode="new"：按 Rust 现行规则（同探头、死时间内、一高一低，留低增益）去重。
    mode="old"：按 e34f560 的老规则（同探头、**时戳完全相等**）去重，用来量
    f2/f3 这类同戳量对去重口径有多敏感——老口径漏并的是相距约 100 ns 的那批，
    它们会把窗内 n 抬高、也会让同戳重数虚高。
    """
    from astropy.io import fits
    times, dets, overs = [], [], []
    with fits.open(path, memmap=True) as hdus:
        gti = hdus["GTI"].data
        gti_lo = np.asarray(gti["START"], float)
        gti_hi = np.asarray(gti["STOP"], float)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            det = int(hdu.name[-2:])
            d = hdu.data
            if d is None or len(d) == 0:
                continue
            pi = np.asarray(d["PI"], np.int32)
            et = np.asarray(d["EVT_TYPE"], np.int32)
            gt = np.asarray(d["GAIN_TYPE"], np.int32)
            t = np.asarray(d["TIME"], float)
            dtc = np.asarray(d["DEAD_TIME"], float)
            keep = (et == NORMAL) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW)
            # GTI 闭区间，与 GtiHdu::contains 同口径
            k = np.searchsorted(gti_lo, t, "right") - 1
            keep &= (k >= 0) & (t <= gti_hi[np.clip(k, 0, gti_hi.size - 1)])
            t, pi, gt, dtc = t[keep], pi[keep], gt[keep], dtc[keep]
            if t.size == 0:
                continue
            o = np.argsort(t, kind="stable")
            t, pi, gt, dtc = t[o], pi[o], gt[o], dtc[o]
            dead = float(dtc[0]) * 1e-6 if np.isfinite(dtc[0]) and dtc[0] > 0 else 0.0
            drop = dedupe(t, gt, 0.0 if mode == "old" else dead)
            t, pi, gt = t[~drop], pi[~drop], gt[~drop]
            lowg = gt == 1
            hist = np.bincount(pi[lowg], minlength=OVERFLOW) if lowg.any() else np.zeros(OVERFLOW, int)
            cut = bump_start(hist)
            times.append(t)
            dets.append(np.full(t.size, det, np.int16))
            overs.append((pi >= cut) if cut > 0 else np.zeros(t.size, bool))
    if not times:
        return None
    time = np.concatenate(times)
    o = np.argsort(time, kind="stable")
    return time[o], np.concatenate(dets)[o], np.concatenate(overs)[o]


def chance_multiplicity(n, m):
    """n 个计数随机撒进 m 个 ulp 格子，落在 >=2 / >=3 重格里的计数期望占比。

    单格计数 ~ Binom(n, 1/m)，E[落在 >=k 重格里的计数] = m·Σ_{j>=k} j·P(X=j)，
    除以 n 即期望占比。闭区间 [k0, k1] 有 w+1 个格。
    """
    if n < 2 or m < 1:
        return 0.0, 0.0
    p = 1.0 / m
    jmax = min(n, 40)
    j = np.arange(0, jmax + 1)
    pmf = np.exp(gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
                 + j * np.log(p) + (n - j) * np.log1p(-p))
    e2 = m * float((j[2:] * pmf[2:]).sum()) / n
    e3 = m * float((j[3:] * pmf[3:]).sum()) / n if jmax >= 3 else 0.0
    return min(e2, 1.0), min(e3, 1.0)


def main():
    day, sig_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "new"
    signals = sorted(json.load(open(sig_path)), key=lambda s: s["start"])
    by_hour = {}
    for s in signals:
        by_hour.setdefault(s["start"][11:13], []).append(s)
    print(f"候选 {len(signals)}", flush=True)

    rows = []
    for path in hour_files(day):
        hh = path.rsplit("_", 2)[-2]
        sig = by_hour.get(hh, [])
        if not sig:
            continue
        res = load_hour(path, mode)
        if res is None:
            continue
        time, det, over = res
        # 量化步 q 是该时刻上 float64 的 ulp，跨 2 的整数次幂会整体跳一档。
        # A 星 MET 在 2023-04-03T10:42:08 前后跨 2^27（14.9 -> 29.8 ns），
        # 所以 q 不是全任务常数；逐小时取并硬断言这一小时没跨档。
        q = float(np.spacing(time[0]))
        assert q == float(np.spacing(time[-1])), f"{path}: 该小时跨了 ulp 档"
        key = np.rint(time / q).astype(np.int64)
        print(f"[{hh}] GTI 内去重后事例 {time.size:,}  q={q*1e9:.3f}ns  "
              f"超量程占 {over.mean()*100:.2f}%  候选 {len(sig)}", flush=True)
        del time
        for s in sig:
            k0 = int(np.rint(met_exact(s["start"]) / q)) + int(np.rint(s["delay"] / q))
            w = int(np.rint(s["bin_size_best"] / q))
            k1 = k0 + w
            i0 = int(np.searchsorted(key, k0, "left"))
            i1 = int(np.searchsorted(key, k1, "right"))
            kk, dd, oo = key[i0:i1], det[i0:i1], over[i0:i1]
            n = kk.size
            if n:
                _, inv, mult = np.unique(kk, return_inverse=True, return_counts=True)
                per = mult[inv]
                f2 = float((per >= 2).sum()) / n
                f3 = float((per >= 3).sum()) / n
                n_trip = int((per >= 3).sum())
                n_lo = int((kk == k0).sum())
                n_hi = int((kk == k1).sum()) if w else n_lo
                ndet = int(np.unique(dd).size)
                nkeep = int((~oo).sum())
            else:
                f2 = f3 = 0.0
                n_trip = ndet = nkeep = n_lo = n_hi = 0
            e2, e3 = chance_multiplicity(n, w + 1)
            rows.append([s["start"][:26], hh, s["false_positive_per_year"], s["count"], s["mean"],
                         s["bin_size_best"] * 1e6, w, n, n_lo, n_hi, nkeep, ndet,
                         round(f2, 6), round(f3, 6), n_trip, round(e2, 8), round(e3, 8),
                         round(e3 * n, 6), s["position"]["longitude"], s["position"]["latitude"]])
        del key, det, over

    hdr = ["start", "hour", "fa", "count", "mean", "bin_us", "n_cells", "n_obs", "n_lo", "n_hi",
           "n_keep", "n_det", "f2", "f3", "n_trip", "e_f2", "e_f3", "e_n_trip", "lon", "lat"]
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)

    cnt = np.array([r[3] for r in rows], float)
    obs = np.array([r[7] for r in rows], float)
    lo = np.array([r[8] for r in rows], float)
    hi = np.array([r[9] for r in rows], float)
    hit = int((cnt == obs).sum())
    print(f"\n对账 n_obs == count: {hit}/{len(rows)} = {hit/len(rows)*100:.2f}%", flush=True)
    bad = cnt != obs
    if bad.any():
        print(f"  差值 {np.unique((obs-cnt)[bad], return_counts=True)}")
        # 端点并列能解释多少：按下标取的窗在每个端点上只收一条，
        # 所以 count 只能落在 [n_obs-(n_lo-1)-(n_hi-1), n_obs] 里
        floor = obs - np.maximum(lo - 1, 0) - np.maximum(hi - 1, 0)
        expl = bad & (cnt >= floor) & (cnt <= obs)
        print(f"  端点并列可解释 {int(expl.sum())}/{int(bad.sum())} = "
              f"{expl.sum()/bad.sum()*100:.2f}%；剩 {int((bad & ~expl).sum())} 个说不清")


if __name__ == "__main__":
    main()
