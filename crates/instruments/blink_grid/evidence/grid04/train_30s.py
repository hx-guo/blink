"""B 角候选是不是一列而不是一个：候选前后 ±30 s 里同类毫秒尖峰的密度。

未决项 14 在 GRID-07 的 17 个 B 角候选上做过（每分钟中位 182 个同类尖峰，对 GRID-03B
的 29 个短硬暴 4.0）。**这里在 GRID-04 上独立做一遍**：04 有 18 个 B 角候选 + 3 个
中间带，是全队第二大的样本，而且是另一颗星、另一份数据。

**⚠️ 这条判据已撤回，本脚本只作记录与复现用。** 原意是"每一个候选都坐在一串同类
尖峰中间"是阳性证据（TGF 孤立、微暴成团）。**实测它整条来自读出成帧**：
±30 s 窗里 3 ms 格的空格占比中位 0.153（纯泊松 1.5e-4），最长空段 45–118 ms；
速率匹配的对照时刻尖峰比候选处还多（1890 对 454）；**限制到无成帧的干净段后，
尖峰率从 585 个/分掉到 0.0**（泊松期望 1.1）。**零假设用纯泊松是错的——高磁纬
高速率段的本底本身就非泊松，而非泊松的原因就是成帧。**
见 `spike_char.py` / `spike_clean.py` / `train_ctrl2.py` 与 `readout-and-b-corner.md` 4.2 节。

做法（照未决项 14 的口径）：±30 s 按 3 ms 分格，本底取该格前后 ±1 s 的局部均值，
数除候选自己那一格以外、逐格泊松 p < 1e-4 的格子。同时给出**纯泊松下的期望个数**，
因为在 1270 c/s、3 ms 格上 µ ≈ 3.8，偶然假尖峰本来就不是零。

用法: python3 train_30s.py <SAT> <tgfs.json> <out.csv> [fa 上限]
"""

import datetime as dt
import glob
import json
import math
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
EPOCH = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
HALF = 30.0
BIN = 0.003
LOCAL = 1.0
PTHR = 1e-4


def met_of(s):
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    whole = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (whole - EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def prev_day(day):
    y, m, d = (int(x) for x in day.split("/"))
    p = dt.date(y, m, d) - dt.timedelta(days=1)
    return "%04d/%02d/%02d" % (p.year, p.month, p.day)


def load(sat, day, t0, t1):
    """取 [t0, t1] 的事例，并返回实际被 GTI 覆盖的总时长（活时间分母要用它）。"""
    out, live = [], 0.0
    for dpath in (day, prev_day(day)):
        vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, dpath)))
        if not vers:
            continue
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            try:
                with fits.open(path, memmap=False) as hd:
                    g = hd["GTI"].data
                    gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
                    if ge < t0 or gs > t1:
                        continue
                    live += min(ge, t1) - max(gs, t0)
                    emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
                    nch = emin.size
                    for i in range(4):
                        d = hd["EVENTS%d" % i].data
                        t = np.asarray(d["TIME"], dtype=np.float64)
                        pi = np.asarray(d["PI"], dtype=np.int32)
                        ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
                        ok = (pi >= 1) & (pi < nch)
                        e = np.zeros(t.size)
                        e[ok] = emin[pi[ok] - 1]
                        out.append(t[(ty == 1) & ok & (e >= ETH) & (t >= t0) & (t <= t1)])
            except Exception:
                continue
    return (np.sort(np.concatenate(out)) if out else np.zeros(0)), live


def poisson_sf(k, mu):
    """P(K ≥ k | mu)。**直接加上尾，不用 1 − cdf。**

    `1 − Σ_{i<k} pmf` 在 P 很小时会被浮点静默吃掉（cdf 趋近 1.0，相减只剩
    机器精度的残渣）。本函数原来就是那个写法——在阈 1e-4 上还安全，但任何人
    把阈调到 1e-8 就会归零。上尾直接累加没有这个问题。
    """
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    k = int(k)
    # 从第 k 项开始往上加，加到贡献可忽略为止
    term = math.exp(-mu)
    for i in range(k):
        term *= mu / (i + 1)
    total = term
    i = k
    while term > total * 1e-16 and i < k + 10000:
        i += 1
        term *= mu / i
        total += term
    return total


def count_spikes(ev, t0, live):
    """±30 s 里逐格泊松 p < PTHR 的尖峰个数（不含候选所在格）与总格数。"""
    lo, hi = t0 - HALF, t0 + HALF
    nb = int((hi - lo) / BIN)
    idx = ((ev - lo) / BIN).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < nb)]
    cnt = np.bincount(idx, minlength=nb)
    # 局部本底：±1 s 的滑动均值，用累积和
    half = int(LOCAL / BIN)
    cs = np.concatenate(([0], np.cumsum(cnt)))
    a = np.maximum(np.arange(nb) - half, 0)
    b = np.minimum(np.arange(nb) + half + 1, nb)
    mu = (cs[b] - cs[a]) / (b - a)
    own = int((t0 - lo) / BIN)
    n_spike, kmax = 0, 0
    for i in np.flatnonzero(cnt >= 8):
        if abs(i - own) <= 1:
            continue
        if poisson_sf(cnt[i], mu[i]) < PTHR:
            n_spike += 1
            kmax = max(kmax, int(cnt[i]))
    # 纯泊松期望：对每格按它自己的 mu 算越阈概率之和（同一套阈）
    exp_chance = 0.0
    for i in range(0, nb, 37):          # 抽样 1/37 的格再乘回去，够给量级
        m = mu[i]
        if m <= 0:
            continue
        k = 8
        while poisson_sf(k, m) >= PTHR and k < 200:
            k += 1
        exp_chance += poisson_sf(k, m)
    exp_chance *= 37.0
    return n_spike, exp_chance, kmax, float(np.median(mu))


def main():
    sat, jpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fa_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    fh = open(out, "w")
    fh.write("sat,start,fa,count,live_s,n_spike,per_min,exp_chance_per_min,kmax,mu_med\n")
    for c in json.load(open(jpath)):
        s = c["signal"]
        if s.get("false_positive_per_year", 1e9) > fa_max:
            continue
        t0 = met_of(s["start"]) + s.get("delay", 0.0)
        ev, live = load(sat, s["start"][:10].replace("-", "/"), t0 - HALF, t0 + HALF)
        if ev.size < 1000 or live < 5.0:
            continue
        n, exp, kmax, mu = count_spikes(ev, t0, live)
        fh.write("%s,%s,%.3e,%d,%.1f,%d,%.1f,%.2f,%d,%.3f\n"
                 % (sat, s["start"], s["false_positive_per_year"], s["count"],
                    live, n, n / (live / 60.0), exp / (live / 60.0), kmax, mu))
        print("%s  活时间 %5.1f s  其他尖峰 %5d  = %7.1f /分钟（纯泊松期望 %5.2f）最大格 %d"
              % (s["start"][:19], live, n, n / (live / 60.0), exp / (live / 60.0), kmax))
    fh.close()


if __name__ == "__main__":
    main()
