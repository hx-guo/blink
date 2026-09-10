"""量每个显著候选的真实时长、真实硬度，以及硬度的噪声。

搜索给出的 `dur_us` 是窗长，顶在 1 ms 的搜索上限上，不是时长；窗内能道中位数同样
受窗长影响。这里在 ±15 ms 里先用 1 ms 格框出暴的范围，再对扣除本底的累积计数取
5%–95% 得到 T90，然后在 T90 区间内重新量沉积能量中位数与本底之比。

事例准入与搜索一致（`blink_grid::types::Event::keep`）：EVT_TYPE==1、道号在
[1, n_channels) 内（最高道是溢出道）、且该道下限能量 ≥ 30 keV。只筛 EVT_TYPE 会
把溢出道和 30 keV 以下的道也算进来，能道中位数和窗内计数都会偏。

硬度用能量而不是能道：四颗星的道—能对应各不相同（GRID-02 第 1 道 6.66 keV、
GRID-03B 5.23 keV，道宽也不同），能道中位数之比跨星不可比。能量取该道的
sqrt(E_MIN·E_MAX)。

硬度的噪声不能不管：候选只有 7–30 个计数，中位数本身就有几个能道的抖动。这里给
两组数——
  * 自举（hard_e_lo/hi/hard_e_95）：对暴内事例有放回重抽，取硬度的 16%/84% 分位当
    误差棒，95% 分位当上限。硬度中位 1.00 只说明"没测出硬"，`hard_e_95` 才说明
    "在这些计数下能排除硬度大于多少"——检验功效要摆出来，否则"与本底一致"可能
    只是没功效；
  * 零假设（null_p50/null_p95/null_p/null_p_at_1p6）：从本底能谱里抽同样多的事例算
    硬度，重复 `N_MC` 次。null_p95 是"谱与本底相同"时硬度能涨到的 95% 分位，null_p 是
    观测硬度的单侧 p 值。硬度高于 1 不等于谱硬，要高过 null_p95 才算。
    null_p_at_1p6 是零假设下硬度越过名义分界 1.6 的概率，对全表求和就得到"纯噪声
    能造出几个硬候选"——判据的富集要能被这个数解释干净，才敢说剩下的是真的。
另给一个用上全部事例（不只中位数）的判据：暴内与本底能量样本的 KS 检验 p 值。

用法: python3 grid_t90.py <候选 CSV: sat,start,...> <输出 CSV>
"""
from astropy.io import fits
from scipy.stats import poisson, ks_2samp
import glob, os, csv, sys, numpy as np, datetime as dt

G = "/gecamfs/Exchange/GSDC/missions/GRID"
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
ENERGY_THRESHOLD_KEV = 30.0   # 与 blink_grid::types::event::ENERGY_THRESHOLD_KEV 同步
# 疑似量程上限堆积的保守截止。GRID-03B 上实测每个探头有各自的满量程位置
# （det3 ch83–85、det0 ch84–87、det1 ch88–93、det2 ch88–94，约 1.24–1.81 MeV），
# 上限之下一个堆积包、之上基本为零，位置逐探头不同且逐次过境漂移；而 `Event::keep`
# 只把最高道当溢出道，这些沉积就以 1.2–1.9 MeV 硬光子的身份进了搜索与谱硬度。
# 这里按能量而不是道号截止（四星道—能对应不同，道号跨星不可比），取所有已知上限
# 里最低的那个当统一截止。截止之上不全是堆积——真暴本来就有 >1 MeV 光子——所以
# `hardness_e_cut` 是硬度的**下限**，用来检验结论稳不稳，不是修正值。
OVER_RANGE_KEV = 1200.0
N_MC = 4000
RNG = np.random.default_rng(20260910)


def met(iso):
    """ISO 时刻 → MET 秒。小数秒不能截到微秒：搜索产物的时刻带纳秒，候选窗的
    两端都是事例本身的时刻，截断把窗口整体左移不到 1 µs 就足以把落在窗末端的
    那个事例（连同与它同戳的几个）挤出窗外。整秒交给 datetime，小数部分按
    浮点单独加。"""
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - REF).total_seconds() + (float("0." + frac) if frac else 0.0)


def pass_files(sat, t0):
    out = []
    for back in (0, 1):
        day = (REF + dt.timedelta(seconds=t0) - dt.timedelta(days=back)).strftime("%Y/%m/%d")
        dd = f"{G}/{sat}/fits7/{day}"
        if not os.path.isdir(dd):
            continue
        v = sorted(os.listdir(dd))[-1]
        out += sorted(glob.glob(f"{dd}/{v}/*.fits"))
    return out


def ebounds(h):
    """道 → (下限能量, 几何中心能量)。道号从 1 起，索引 pi-1。"""
    e = h["EBOUNDS"].data
    emin = np.asarray(e["E_MIN"], float)
    emax = np.asarray(e["E_MAX"], float)
    return emin, np.sqrt(emin * emax)


def hardness_stats(e_in, e_bk):
    """观测硬度、自举误差棒、零假设分位与 p 值。e_in/e_bk 是能量样本。"""
    n = len(e_in)
    med_bk = float(np.median(e_bk))
    if n == 0 or med_bk <= 0:
        return (np.nan,) * 8
    obs = float(np.median(e_in)) / med_bk
    boot = np.median(RNG.choice(e_in, size=(N_MC, n), replace=True), axis=1) / med_bk
    null = np.median(RNG.choice(e_bk, size=(N_MC, n), replace=True), axis=1) / med_bk
    p = float((null >= obs).mean())
    # 零假设下硬度越过名义分界 1.6 的概率：拿它对全表求和，就得到"纯噪声能造出几个
    # 硬候选"，判据的富集要能被这个数解释干净才敢说剩下的是真的
    p16 = float((null >= 1.6).mean())
    return (obs, float(np.percentile(boot, 16)), float(np.percentile(boot, 84)),
            float(np.percentile(boot, 95)),
            float(np.percentile(null, 50)), float(np.percentile(null, 95)), p, p16)


rows = list(csv.DictReader(open(sys.argv[1])))
w = csv.writer(open(sys.argv[2], "w", newline=""))
COLS = ["sat", "start", "dur_search_us", "t90_us", "excess", "n_t90", "rate_bkg",
        "pi_med_t90", "pi_med_bkg", "hardness_t90", "n_det_t90", "det_frac_t90",
        "e_med_t90", "e_med_bkg", "hardness_e", "hard_e_lo", "hard_e_hi", "hard_e_95",
        "null_p50", "null_p95", "null_p", "null_p_at_1p6", "ks_p",
        "frac_over", "frac_over_bkg", "hardness_e_cut", "null_p_cut"]
w.writerow(COLS)
n_ok = 0
for r in rows:
    sat, t0 = r["sat"], met(r["start"])
    written = False
    for f in pass_files(sat, t0):
        with fits.open(f) as h:
            g = h["GTI"].data
            s0, s1 = float(g["START"][0]), float(g["STOP"][0])
            if not (s0 <= t0 <= s1):
                continue
            emin, ectr = ebounds(h)
            n_ch = len(emin)
            T, P, D = [], [], []
            for k in range(4):
                x = h[f"EVENTS{k}"].data
                t = np.asarray(x["TIME"], float)
                pi = np.asarray(x["PI"])
                # 与搜索同一道准入：非溢出道 + 道下限能量 ≥ 30 keV
                ok = (pi >= 1) & (pi < n_ch)
                keep = np.zeros(len(pi), bool)
                keep[ok] = emin[pi[ok] - 1] >= ENERGY_THRESHOLD_KEV
                m = (np.asarray(x["EVT_TYPE"]) == 1) & keep & (t >= t0 - 1.0) & (t <= t0 + 1.0)
                T.append(t[m]); P.append(pi[m]); D.append(np.full(int(m.sum()), k))
            T = np.concatenate(T); P = np.concatenate(P); D = np.concatenate(D)
            o = np.argsort(T); T, P, D = T[o], P[o], D[o]
            if len(T) < 3:
                break
            E = ectr[P - 1]
            live = min(t0 + 1.0, s1) - max(t0 - 1.0, s0)
            far = np.abs(T - t0) > 0.02
            rate = far.sum() / live

            edges = t0 + np.arange(-15.0, 15.01, 1.0) * 1e-3
            cnt, _ = np.histogram(T, bins=edges)
            sig = poisson.sf(cnt - 1, rate / 1000.0) < 1e-4
            peak = int(np.argmax(cnt))
            lo = hi = peak
            while lo - 1 >= 0 and sig[lo - 1]:
                lo -= 1
            while hi + 1 < len(sig) and sig[hi + 1]:
                hi += 1
            b0, b1 = edges[max(lo - 1, 0)], edges[min(hi + 2, len(edges) - 1)]
            ts = np.sort(T[(T >= b0) & (T < b1)])
            if len(ts) < 3:
                break
            cum = np.arange(1, len(ts) + 1) - rate * (ts - b0)
            total = cum[-1]
            if total <= 0:
                break
            t05 = float(np.interp(0.05 * total, cum, ts))
            t95 = float(np.interp(0.95 * total, cum, ts))
            inside = (T >= t05) & (T <= t95)
            t90 = (t95 - t05) * 1e6
            excess = inside.sum() - rate * (t95 - t05)
            pi_in = float(np.median(P[inside])) if inside.any() else np.nan
            pi_bk = float(np.median(P[far])) if far.any() else np.nan
            dets, cnts = np.unique(D[inside], return_counts=True)
            hard = hardness_stats(E[inside], E[far]) if (inside.any() and far.any()) else (np.nan,) * 8
            ks = ks_2samp(E[inside], E[far]).pvalue if (inside.sum() >= 3 and far.any()) else np.nan
            # 扣掉疑似量程堆积后的第二套硬度：暴内与本底用同一道截止
            below = E < OVER_RANGE_KEV
            f_over = 1.0 - float(below[inside].mean()) if inside.any() else np.nan
            f_over_bk = 1.0 - float(below[far].mean()) if far.any() else np.nan
            cut = (hardness_stats(E[inside & below], E[far & below])
                   if ((inside & below).any() and (far & below).any()) else (np.nan,) * 8)
            w.writerow([sat, r["start"], r["dur_us"], f"{t90:.0f}", f"{excess:.1f}", int(inside.sum()),
                        f"{rate:.0f}", f"{pi_in:.0f}", f"{pi_bk:.0f}",
                        f"{pi_in / pi_bk:.3f}" if pi_bk else "", len(dets),
                        f"{cnts.max() / inside.sum():.2f}" if inside.any() else "",
                        f"{np.median(E[inside]):.1f}" if inside.any() else "",
                        f"{np.median(E[far]):.1f}" if far.any() else ""]
                       + [f"{v:.3f}" if np.isfinite(v) else "" for v in hard]
                       + [f"{ks:.3e}" if np.isfinite(ks) else ""]
                       + [f"{f_over:.3f}" if np.isfinite(f_over) else "",
                          f"{f_over_bk:.4f}" if np.isfinite(f_over_bk) else "",
                          f"{cut[0]:.3f}" if np.isfinite(cut[0]) else "",
                          f"{cut[6]:.3f}" if np.isfinite(cut[6]) else ""])
            n_ok += 1; written = True
            break
    if not written:
        w.writerow([sat, r["start"], r["dur_us"]] + [""] * (len(COLS) - 3))
print("measured:", n_ok, "of", len(rows))
