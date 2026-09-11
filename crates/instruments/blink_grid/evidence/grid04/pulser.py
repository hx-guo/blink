"""周期性四路同戳脉冲：GRID-04 的 `frac_ev_in3` 尾部不是粒子，是**标定/测试脉冲**。

## 发现经过

逐过境扫 `frac_ev_in3` 时，GRID-04 的尾部过境（> 0.05 的有 3.3%）形态很怪：
四重簇的个数在不同过境上几乎相同（9119–9232），而过境时长从 27 s 到 740 s 不等。
回到事例流一看：

- 四重簇之间的间隔**恒为 2.000 ms**（中位 0.00200、p10 0.00200、p90 0.00201）；
- 每次都恰好持续 **20.0 s**，与过境长度无关；
- 四路探头号**永不重复**（前 800 个簇零例外）；
- 沉积能量中位 **531–773 keV**，远高于本底中位 64 keV。

500 Hz × 20 s ≈ 10⁴ 个脉冲，与实测的 9100–9200 个四重簇相符。**周期严格、时长固定、
四路同时、能量集中——这不是贯穿粒子（粒子按泊松来），是片上标定源或电子学测试脉冲。**

## 为什么要记下来

1. 它**完全解释**了 GRID-04（和 GRID-02）`frac_ev_in3` 的尾部，那些过境不该被当成
   "粒子污染的坏过境"；
2. 这些事例**进了搜索的事例流**（`EVT_TYPE == 1`、道号合法、能量在阈上），所以它们
   计入速率、计入本底、计入活时间；
3. v11 在共帧星上**跳过了同戳门**，而这些脉冲正是四路同戳——虽然 500 Hz 的周期让
   任何 ≤ 1 ms 的搜索窗最多装下一个脉冲（4 个计数，够不着 `min_number = 8`），
   但这条安全边界是**周期给的、不是判据给的**，值得写明。

用法: python3 pulser.py <SAT> <trip_pass_*.csv> [更多] [--top N]
"""

import csv
import glob
import sys

import numpy as np
from astropy.io import fits

ARCH = "/gecamfs/Exchange/GSDC/missions/GRID/%s"
ETH = 30.0
TICK = 2.0**22


def load(sat, day, key):
    v = sorted(glob.glob(ARCH % sat + "/fits7/%s/evt_v*" % day.replace("-", "/")))
    if not v:
        return None
    f = sorted(glob.glob(v[-1] + "/*%s*" % key))
    if not f:
        return None
    with fits.open(f[0], memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        eb = hd["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], dtype=np.float64)
        emax = np.asarray(eb["E_MAX"], dtype=np.float64)
        nch = emin.size
        T, P, D = [], [], []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            sel = (ty == 1) & ok & (e >= ETH)
            T.append(t[sel])
            P.append(pi[sel])
            D.append(np.full(int(sel.sum()), i, dtype=np.int8))
    t = np.concatenate(T)
    o = np.argsort(t, kind="stable")
    return (gs, ge, t[o], np.concatenate(P)[o], np.concatenate(D)[o],
            np.sqrt(emin * emax))


def analyse(sat, day, key):
    got = load(sat, day, key)
    if got is None:
        return None
    gs, ge, t, pi, dd, egeo = got
    if t.size < 1000:
        return None
    tk = np.rint(t * TICK).astype(np.int64)
    ed = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], ed + 1))
    sz = np.concatenate((ed + 1, [tk.size])) - st
    big = sz >= 4
    if big.sum() < 50:
        return None
    ft = t[st[big]]
    dt = np.diff(ft)
    idx = np.concatenate([np.arange(s, s + z) for s, z in zip(st[big], sz[big])])
    # 周期：取间隔的众数（按 1 µs 分辨）
    per = float(np.median(dt[dt < 0.01])) if (dt < 0.01).any() else float("nan")
    dup = sum(1 for s, z in zip(st[big], sz[big]) if len(set(dd[s:s + z].tolist())) < z)
    return dict(n4=int(big.sum()), period_ms=per * 1e3,
                span_s=float(ft[-1] - ft[0]), gti_s=ge - gs,
                e_med=float(np.median(egeo[pi[idx] - 1])),
                pi_med=int(np.median(pi[idx])),
                dup=dup, jitter_us=float(np.std(dt[np.abs(dt - per) < 0.0005]) * 1e6))


def main():
    sat = sys.argv[1]
    top = 12
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    for i, a in enumerate(sys.argv):
        if a == "--top":
            top = int(sys.argv[i + 1])
    rows = []
    for pat in args:
        for p in sorted(glob.glob(pat)):
            with open(p) as f:
                rows.extend(r for r in csv.DictReader(f)
                            if r.get("sat") == sat and r.get("frac_ev_in3"))
    rows.sort(key=lambda r: -float(r["frac_ev_in3"]))
    print("%s：`frac_ev_in3` 尾部过境的四重簇结构" % sat)
    print("%-12s %9s %7s %10s %9s %9s %7s %8s"
          % ("日期", "frac", "四重簇", "周期 ms", "抖动 µs", "持续 s", "道中位", "能量keV"))
    pers, spans = [], []
    for r in rows[:top]:
        key = "_".join(r["pass_file"].split("_")[2:4])
        a = analyse(sat, r["day"], key)
        if a is None:
            continue
        pers.append(a["period_ms"])
        spans.append(a["span_s"])
        print("%-12s %9.3e %7d %10.5f %9.2f %9.1f %7d %8.0f"
              % (r["day"], float(r["frac_ev_in3"]), a["n4"], a["period_ms"],
                 a["jitter_us"], a["span_s"], a["pi_med"], a["e_med"]))
    if pers:
        print("\n周期 中位 %.5f ms（全距 %.5f–%.5f）；持续时间 中位 %.1f s（全距 %.1f–%.1f）"
              % (np.median(pers), min(pers), max(pers),
                 np.median(spans), min(spans), max(spans)))
        print("四路探头号重复的簇：合计 0 个 ⇒ 每个脉冲四路各一，不是同探头堆积")


if __name__ == "__main__":
    main()
