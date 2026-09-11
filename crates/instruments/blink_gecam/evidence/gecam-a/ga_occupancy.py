"""A 星本底的占据数分布 ÷ 泊松——搜索用错了零假设，这里把低估的倍数量出来。

搜索的 `fa` 假定本底是泊松的。25 路 GRD 上一次宇宙线穿越会同时点亮好几路，
于是"窄窗里挤进 k 个计数"的实际概率远高于泊松。**这个比值就是 `fa` 在阈值处
被低估的倍数**，也是"候选率比配置的假阳性率高四个数量级"的定量来源。

做法：把一小时的事例流（已按 `Event::keep` 准入、按 GTI 过滤、双增益去重，
与 Rust 侧逐条对齐）按固定宽度 W 铺成格子，数"出现 >= k 个计数的格数"，
与泊松的期望格数比。

**λ 必须逐小段取，不能整小时一个。** 搜索用的 λ 是局部的（候选两侧各半秒的
本底窗）。整小时一个 λ 会把一小时之内的速率漂移算成过离散 —— 而且**正是在大
W 上高估最多**，因为窗越宽、一个格子越能跨过速率的变化。这里把每小时切成
SEG = 10 s 的段，逐段用该段自己的 λ 算期望再相加：

    E[>=k 的格数] = Σ_s  m_s · P(X >= k | λ_s)，  λ_s = n_s / m_s，m_s = live_s / W

两套口径并排报，**"逐段"那一版才是对搜索实际生效的数**；整小时那一版留着，
是为了把"漂移被算成过离散"这件事本身量出来。

**其它几条口径**：
* **必须在对数空间算**。泊松尾在 k = 20、λ = 1e-4 时是 1e-94 量级，先算成
  浮点再相除会得到 inf/nan，而 nan 传进分位数看着像"算不出来"，其实是把最
  极端那批悄悄丢掉了。期望格数用 `logsumexp` 累加。
* **报一条随 k 的曲线，不是一个点**；并且必须报"候选 `count` 分布中位处"
  那个 k（A 星是 9），那才是对目录实际生效的数。
* **W 要标，而且要给对 W 的依赖**。A 星候选 `bin_size_best` 中位只有
  0.149 µs，与 B 星取的 10 µs 差两个数量级，两边的数不能直接比。
* **不剔除候选窗**。候选本身就是这些簇——它们不是外来的信号，就是本底过离散
  的那条尾巴，剔掉等于把要量的东西量掉。为免这句被当成回避，同时给出"剔除
  候选窗之后"的对照。
* **有一条结论完全不用 λ**：实测"≥8 的格数"随 W 怎么变。它不受本条任何口径
  争议影响，对外首选它。

用法: python3 ga_occupancy.py <YYYY-MM-DD> <signals.json> <输出前缀>
"""

import datetime as dt
import glob
import json
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson

ROOT = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_CHANNEL, OVERFLOW, NORMAL = 54, 448, 1
EPOCH = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
LN10 = np.log(10.0)
WIDTHS_US = [0.0298, 0.149, 0.5, 1.0, 10.0, 100.0, 1000.0]
KS = list(range(2, 21))
K_MEDIAN = 9          # A 星候选 count 的中位
SEG = 10.0            # 逐段 λ 的段长，秒。与 gecamB 的安静段取同一个值才能跨星比


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
        lo = np.array(gti["START"], float)
        hi = np.array(gti["STOP"], float)
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
    return time, lo, hi


def live_before(x, lo, hi):
    """[起点, x] 落在 GTI 里的秒数。段数只有个位数，逐段累加即可。"""
    tot = np.zeros_like(x)
    for a, b in zip(lo, hi):
        tot += np.clip(np.minimum(b, x) - a, 0.0, b - a)
    return tot


def segment_rates(rel, lo, hi):
    """把这一小时切成 SEG 秒的段，给出逐段活时间与事例数。"""
    edges = np.arange(0.0, float(rel[-1]) + SEG, SEG)
    seg_live = np.diff(live_before(edges, lo, hi))
    idx = np.floor(rel / SEG).astype(np.int64)
    seg_n = np.bincount(idx, minlength=seg_live.size)[:seg_live.size].astype(float)
    ok = seg_live > 0.5              # 半秒以下的边角段不参加
    return seg_live[ok], seg_n[ok]


def expected_log10(w, seg_live, seg_n):
    """逐段泊松期望的 ">=k 格数"，对数空间累加。返回 log10 的数组，对齐 KS。"""
    m = seg_live / w
    lam = seg_n / m
    logm = np.log(m)
    out = np.empty(len(KS))
    for i, k in enumerate(KS):
        out[i] = logsumexp(logm + poisson.logsf(k - 1, lam)) / LN10
    return out, float(seg_n.sum() / m.sum())


def occupancy(rel, w, drop_cells=None):
    """宽度 w 的格子里的占据数分布。

    只有非空格会出现在 np.unique 里；空格数对 k >= 2 的尾巴没有影响，所以不必
    把它们造出来（W = 0.0298 µs 时格数是 1e11，造出来会直接撑爆内存）。
    """
    cid = np.floor(rel / w).astype(np.int64)
    uid, cnt = np.unique(cid, return_counts=True)
    ndrop = 0
    if drop_cells is not None and drop_cells.size:
        keep = ~np.isin(uid, drop_cells)
        ndrop = int((~keep).sum())
        uid, cnt = uid[keep], cnt[keep]
    return np.bincount(cnt), int(cnt.sum()), ndrop


def report(tag, occ, n_ev, w, live, seg_live, seg_n):
    obs = np.array([occ[k:].sum() if k < occ.size else 0 for k in KS], float)
    log_obs = np.where(obs > 0, np.log10(np.maximum(obs, 1.0)), np.nan)
    # 整小时一个 λ
    log_exp_g, lam_g = expected_log10(w, np.array([live]), np.array([float(n_ev)]))
    # 逐段 λ
    log_exp_s, lam_mean = expected_log10(w, seg_live, seg_n)
    print(f"\n[{tag}] W = {w * 1e6:g} µs  活时间 {live:.0f} s  事例 {n_ev:,}  "
          f"λ(整小时) = {lam_g:.4g}  段数 {seg_live.size}")
    print(f"{'k':>4} {'实测>=k格数':>12} {'期望(整小时λ)':>14} {'期望(逐段λ)':>13} "
          f"{'比(整小时)':>12} {'比(逐段)':>12} {'两者之比':>9}")
    for i, k in enumerate(KS):
        if obs[i] <= 0:
            print(f"{k:4d} {0:12d} {10.0 ** log_exp_g[i]:14.3e} {10.0 ** log_exp_s[i]:13.3e} "
                  f"{'— 实测已空':>12} {'':>12} {'':>9}")
            continue
        rg = 10.0 ** (log_obs[i] - log_exp_g[i])
        rs = 10.0 ** (log_obs[i] - log_exp_s[i])
        mark = "  <- 候选 count 中位" if k == K_MEDIAN else ""
        print(f"{k:4d} {int(obs[i]):12d} {10.0 ** log_exp_g[i]:14.3e} {10.0 ** log_exp_s[i]:13.3e} "
              f"{rg:12.3e} {rs:12.3e} {10.0 ** (log_exp_s[i] - log_exp_g[i]):9.2f}{mark}")
    return (log_obs - log_exp_g), (log_obs - log_exp_s), obs


def main():
    day, sig_path, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    signals = json.load(open(sig_path))
    by_hour = {}
    for s in signals:
        by_hour.setdefault(s["start"][11:13], []).append(s)

    out = {"day": day, "k_list": KS, "widths_us": WIDTHS_US, "seg_seconds": SEG, "hours": {}}
    for path in hour_files(day):
        hh = path.rsplit("_", 2)[-2]
        res = load_hour(path)
        if res is None:
            continue
        time, glo, ghi = res
        t0 = float(time[0])
        rel = time - t0
        lo, hi = glo - t0, ghi - t0
        live = float(np.sum(np.clip(hi - lo, 0.0, None)))
        seg_live, seg_n = segment_rates(rel, lo, hi)
        rate = seg_n / seg_live
        print(f"\n{'=' * 96}\n小时 {hh}：事例 {time.size:,}  活时间 {live:.0f} s  "
              f"整小时率 {time.size / live:.0f} c/s\n"
              f"  逐 {SEG:.0f} s 段的率：n={seg_live.size}  "
              f"p5/p50/p95 = {np.percentile(rate, 5):.0f} / {np.percentile(rate, 50):.0f} / "
              f"{np.percentile(rate, 95):.0f} c/s  最大/最小 = {rate.max() / rate.min():.2f}\n"
              f"{'=' * 96}", flush=True)
        # 把逐段率存下来：漂移污染的倍数应当等于 <λ_s^k>/<λ_s>^k（小 λ 极限下
        # P(X>=k) ~ λ^k/k!，W 在分子分母里同次幂约掉），存了才能验而不是断言。
        hrec = {"seg_live": seg_live.tolist(), "seg_n": seg_n.tolist(),
                "rate_seg": {"p5": float(np.percentile(rate, 5)),
                             "p50": float(np.percentile(rate, 50)),
                             "p95": float(np.percentile(rate, 95)),
                             "max_over_min": float(rate.max() / rate.min()),
                             "n_seg": int(seg_live.size)}}
        for w_us in WIDTHS_US:
            w = w_us * 1e-6
            occ, n_ev, _ = occupancy(rel, w)
            lg, ls, obs = report(f"{hh} 全段", occ, n_ev, w, live, seg_live, seg_n)
            hrec[str(w_us)] = {
                "n_ev": n_ev,
                "obs_tail": {str(k): int(obs[i]) for i, k in enumerate(KS)},
                "log10_ratio_global": {str(k): (None if not np.isfinite(lg[i]) else float(lg[i]))
                                       for i, k in enumerate(KS)},
                "log10_ratio_seg": {str(k): (None if not np.isfinite(ls[i]) else float(ls[i]))
                                    for i, k in enumerate(KS)},
            }
        sig = by_hour.get(hh, [])
        if sig:
            c0 = np.array([met_exact(s["start"]) + s["delay"] - t0 for s in sig])
            c1 = c0 + np.array([s["bin_size_best"] for s in sig])
            for w_us in (0.149, 10.0):
                w = w_us * 1e-6
                cells = np.unique(np.concatenate([np.floor(c0 / w), np.floor(c1 / w)])).astype(np.int64)
                occ, n_ev, nd = occupancy(rel, w, drop_cells=cells)
                report(f"{hh} 剔除候选窗（剔 {nd} 格）", occ, n_ev, w, live, seg_live, seg_n)
        out["hours"][hh] = hrec
        del time, rel

    json.dump(out, open(prefix + "_occ.json", "w"))
    print(f"\n写出 {prefix}_occ.json")


if __name__ == "__main__":
    main()
