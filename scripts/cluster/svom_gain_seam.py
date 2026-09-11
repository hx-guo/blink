"""两档增益的交接处在能谱里留下的缺口——用本底的大统计量量出来。

L1B 的 `GAIN_TYPE` 把事例分成低增益（PI 小）与高增益（PI 大）两支，在某一道
交接。交接不是平滑过渡：低增益支在某道戛然而止、高增益支从下一道开始，
中间留 1–5 个计数明显偏低的道，2025 年起其中一道几乎为零。EBOUNDS 与 CALDB
的响应都不知道这件事，于是任何折叠模型在那几道上系统性高估。

判据不依赖模型：跨过缺口、紧挨两侧各 3 道做对数线性插值，比实测。整小时三路
合计上千万个事例，缺口是 10² σ 级的，不是统计涨落。

用法: python3 svom_gain_seam.py <out_prefix> [epoch_index]
"""
import glob
import sys

import numpy as np
from astropy.io import fits

D = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
EPOCHS = [("2024/07/03", "12"), ("2024/08/15", "06"), ("2024/10/20", "18"),
          ("2024/12/05", "04"), ("2025/05/03", "04"), ("2025/09/19", "00"),
          ("2026/02/13", "03")]
NCH = 256


def latest(pat):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    seen = {}
    for f in g:
        seen[f.rsplit("_v", 1)[0]] = f
    return sorted(seen.values())[-1]


def hist_hour(f):
    h0 = np.zeros(NCH, np.int64)
    h1 = np.zeros(NCH, np.int64)
    with fits.open(f) as hd:
        for i in (3, 4, 5):
            d = hd[i].data
            pi = np.asarray(d["PI"])
            k = ((np.asarray(d["EVT_TYPE"]) == 0) & (np.asarray(d["ANTI_COIN"]) == 0)
                 & (pi >= 0) & (pi < NCH))
            gt = np.asarray(d["GAIN_TYPE"])
            h0 += np.bincount(pi[k & (gt == 0)], minlength=NCH)[:NCH]
            h1 += np.bincount(pi[k & (gt == 1)], minlength=NCH)[:NCH]
    return h0, h1


def bridge(h0, h1):
    """返回（低增益末道, 高增益首道, 缺口道, 逐道实测/插值, 缺的计数, 缺的占比）。

    支路边界取累计 0.001% 分位，免得被零星的越界事例带偏。锚点只取紧挨缺口
    两侧的 3 道：本底谱在这一带有 511 keV 线与曲率，锚点放远了反而会偏。
    """
    tot = (h0 + h1).astype(float)
    c0 = np.cumsum(h0)
    lo = int(np.searchsorted(c0, c0[-1] * 0.99999))
    c1 = np.cumsum(h1)
    hi = int(np.searchsorted(c1, c1[-1] * 0.00001))
    a, b = max(min(hi, lo) - 2, 4), min(max(hi, lo) + 2, NCH - 5)
    x = np.concatenate([np.arange(a - 3, a), np.arange(b + 1, b + 4)])
    p = np.polyfit(x, np.log(np.maximum(tot[x], 1)), 1, w=np.sqrt(tot[x]))
    ch = np.arange(a, b + 1)
    pred = np.exp(np.polyval(p, ch))
    r = tot[ch] / pred
    bad = r < 0.95
    return (lo, hi, ch, r, float((pred - tot[ch])[bad].sum()),
            float((pred - tot[ch])[bad].sum() / pred[bad].sum()) if bad.any() else 0.0)


def main(prefix, only=None):
    fh = open(prefix + ".txt", "w")

    def say(s=""):
        print(s, flush=True)
        fh.write(s + "\n")

    say("两档增益交接处的缺口（本底，整小时三路合计，EVT_TYPE=0 且 ANTI_COIN=0）")
    say("判据：跨过缺口、紧挨两侧各 3 道做对数线性插值，比实测。")
    say()
    for ie, (day, hh) in enumerate(EPOCHS):
        if only is not None and ie != only:
            continue
        yy = day[2:4] + day[5:7] + day[8:10]
        f = latest("%s/%s/grm_evt/svom_grm_evt_%s_%s_v*.fits" % (D, day, yy, hh))
        if f is None:
            say("%s %s:00  缺文件" % (day, hh))
            continue
        h0, h1 = hist_hour(f)
        lo, hi, ch, r, miss, frac = bridge(h0, h1)
        bad = ch[r < 0.95]
        say("== %s %s:00  %s  事例 %d ==" % (day, hh, f.split("/")[-1], (h0 + h1).sum()))
        say("   低增益支末道 ch%d，高增益支首道 ch%d；缺口 ch%s，缺 %.0f 个计数（该段的 %.1f%%）"
            % (lo, hi, "%d–%d" % (bad.min(), bad.max()) if len(bad) else "无",
               miss, 100 * frac))
        say("   逐道 实测/插值：" + "  ".join("ch%d %.2f" % (c, v) for c, v in zip(ch, r)))
        say("   逐道 低增益/高增益 计数：" + "  ".join(
            "ch%d %d/%d" % (c, h0[c], h1[c]) for c in ch))
        np.save("%s_h_%d.npy" % (prefix, ie), np.vstack([h0, h1]))
        say()
    fh.close()
    print("→", prefix + ".txt")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
