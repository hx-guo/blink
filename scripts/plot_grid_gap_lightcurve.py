"""读出空洞的光变曲线：scripts/cluster/grid_gap_lightcurve.py 的产物画成一张图。

(a) 整次过境逐秒计数率，GRID-03B 与同星的 GRID-04；
(b) 逐秒空洞占比（四路合计 > 3 ms 无事例的时间比例）对泊松预期 e^{-rτ}(1+rτ)——
    超出预期的部分才是读出空洞，低计数率时 3 ms 的间隔本来就常见；
(c) 最高计数率那一秒里、计数最多的 1 ms 格前后 ±40 ms，1 ms 分格，两颗星；
(d) 低计数率对照，同样分格。

用法: python3 scripts/plot_grid_gap_lightcurve.py <gap_lc_*.json> -o <PNG>
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES

plt.rcParams.update({"font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif",
                     "axes.unicode_minus": False, "font.size": 13})
TAU = 0.003
C03, C04 = "#c53030", "#2b6cb0"


def poisson_gap(r):
    x = r * TAU
    return np.exp(-x) * (1 + x)


def arr(d, k):
    a = np.array(d[k], float)
    a[a < 0] = np.nan
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json))
    t0 = d["gti"][0]
    s = np.array(d["sec"]); r = arr(d, "rate"); g = arr(d, "gap_frac")
    g4 = d["grid04"][0] if d["grid04"] else None

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.18)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(s - t0, r, color=C03, lw=1.2, label="GRID-03B")
    if g4:
        ax.plot(np.array(g4["sec"]) - t0, arr(g4, "rate"), color=C04, lw=1.2, label="GRID-04（同一颗卫星）")
    ax.axvline(d["hi"]["sec"] - t0, color="0.4", ls=":", lw=1.2)
    ax.set_yscale("log"); ax.set_ylabel("四路合计计数率 (c/s)")
    ax.set_xlabel("过境内时间 (s)，起点 = GTI 开始")
    ax.set_title("(a) 整次过境的逐秒计数率　%s" % d["file"], loc="left")
    ax.legend(loc="upper left", fontsize=11)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(s - t0, g, color=C03, lw=1.2, label="GRID-03B 实测")
    ax.plot(s - t0, poisson_gap(r), color=C03, lw=1.2, ls="--", alpha=0.6, label="GRID-03B 泊松预期")
    if g4:
        s4 = np.array(g4["sec"]) - t0; r4 = arr(g4, "rate")
        ax.plot(s4, arr(g4, "gap_frac"), color=C04, lw=1.2, label="GRID-04 实测")
        ax.plot(s4, poisson_gap(r4), color=C04, lw=1.2, ls="--", alpha=0.6, label="GRID-04 泊松预期")
    ax.axvline(d["hi"]["sec"] - t0, color="0.4", ls=":", lw=1.2)
    ax.set_ylim(-0.02, 1.02); ax.set_ylabel("空洞占比\n（> 3 ms 无事例的时间）")
    ax.set_xlabel("过境内时间 (s)")
    ax.set_title("(b) 空洞占比：实测对泊松预期——虚线以上的部分才是读出空洞", loc="left")
    ax.legend(loc="upper right", fontsize=10, ncol=2)

    x = np.arange(-1000, 1000) + 0.5
    for col, (key, title) in enumerate((("hi", "(c) 空洞超出泊松预期最多的一秒，以最长空洞为中心"), ("lo", "(d) 低计数率对照，1 ms 分格"))):
        ax = fig.add_subplot(gs[2, col])
        blk = d[key]
        if blk is None:
            continue
        tot = np.sum(np.array(blk["ms"]), axis=0)
        ax.step(x, tot, where="mid", color=C03, lw=1.4, label="GRID-03B 四路合计")
        per = np.array(blk["ms"])
        for k in range(4):
            ax.step(x, per[k], where="mid", color=C03, lw=0.6, alpha=0.35)
        if key == "hi" and g4 and g4.get("ms_hi"):
            ax.step(x, np.sum(np.array(g4["ms_hi"]), axis=0), where="mid", color=C04, lw=1.4,
                    label="GRID-04 四路合计（同一时刻）")
        txt = "计数率 %.0f c/s，空洞占比 %.2f（泊松预期 %.2f）" % (blk["rate"], blk["gap_frac"], poisson_gap(blk["rate"]))
        if key == "hi":
            txt += "\n扣掉空洞后的帧内计数率 %.0f c/s" % blk["frame_rate"]
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10.5, color="0.25")
        ax.set_xlabel("相对中心的时间 (ms)"); ax.set_ylabel("计数 / ms")
        ax.set_title(title, loc="left"); ax.set_xlim(-1000, 1000)
        ax.set_ylim(0, max(tot.max(), 1) * 1.45)
        ax.legend(loc="upper right", fontsize=10)
    fig.suptitle("GRID-03B 读出空洞：高计数率下的实测形态", fontsize=16, y=0.995)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
