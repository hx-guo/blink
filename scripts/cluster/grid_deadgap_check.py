"""读出空洞门在安静本底里的误触发率：随机取窗，完全照搬 search.rs 的 has_dead_gap。

判据（与 Rust 同）：本底窗 = 中心 ±0.5 s、裁到所在过境的 GTI；L = 窗内最长空段（含窗两端到
最近事例）；r = max(窗内计数率, 整次过境计数率)；r·L > 9.2 即否决。事例准入同 `Event::keep`。
过境计数率 = 过境内准入事例数 / 过境时长（与 Rust 一样不截到小时——Rust 截到小时，这里按整次
过境，差别只在跨整点的过境上）。

每个窗同时报：只用窗内计数率时是否触发；按泊松算的触发概率
    P = 1 − exp(−n · exp(−9.2 · r_w / r))，n = 窗内间隔数，r_w = 窗内计数率。

用法: python3 grid_deadgap_check.py <卫星目录名> <worker> <nworkers> <输出 CSV>
"""
import csv, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
THR = 9.2
ETH = 30.0
N_WIN = 200
rng = np.random.default_rng(20260927)


def all_days(sat):
    return sorted(glob.glob(f"{G}/{sat}/fits7/20*/*/*"))


def load(f):
    with fits.open(f, memmap=False) as h:
        g = h["GTI"].data
        gs, ge = float(g["START"][0]), float(g["STOP"][0])
        emin = np.asarray(h["EBOUNDS"].data["E_MIN"], float)
        T = []
        for k in range(4):
            e = h[f"EVENTS{k}"].data
            t = np.asarray(e["TIME"], float); pi = np.asarray(e["PI"], int); ty = np.asarray(e["EVT_TYPE"], int)
            ok = (pi >= 1) & (pi < emin.size)
            en = np.zeros(t.size); en[ok] = emin[pi[ok] - 1]
            T.append(t[(ty == 1) & ok & (en >= ETH)])
    t = np.sort(np.concatenate(T))
    return gs, ge, t[(t >= gs) & (t <= ge)]


sat, w, nw, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
days = all_days(sat)
pick = days[:: max(1, len(days) // 24)][:24]
rows = []
for d in pick[w::nw]:
    vs = sorted(os.listdir(d))
    if not vs:
        continue
    for f in sorted(glob.glob(f"{d}/{vs[-1]}/*.fits")):
        try:
            gs, ge, t = load(f)
        except Exception:
            continue
        if t.size < 1000 or ge - gs < 10:
            continue
        pass_rate = t.size / (ge - gs)
        for c in rng.uniform(gs, ge, N_WIN):
            lo, hi = max(c - 0.5, gs), min(c + 0.5, ge)
            a, b = np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
            win = t[a:b]
            span = hi - lo
            if win.size == 0:
                L = span
            else:
                L = max(win[0] - lo, hi - win[-1], np.diff(win).max() if win.size > 1 else 0.0)
            rw = win.size / span
            r = max(rw, pass_rate)
            p = 1 - np.exp(-max(win.size - 1, 0) * np.exp(-THR * rw / r)) if rw > 0 else 1.0
            rows.append(dict(file=os.path.basename(f), centre="%.6f" % c, span="%.3f" % span,
                             n=win.size, rate_win="%.1f" % rw, rate_pass="%.1f" % pass_rate,
                             L_ms="%.3f" % (L * 1e3), rL="%.2f" % (r * L), fire=int(r * L > THR),
                             fire_win_only=int(rw * L > THR), p_poisson="%.4g" % p))
with open(out, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print("worker", w, "windows", len(rows))
