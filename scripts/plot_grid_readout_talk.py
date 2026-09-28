"""天格讲图 4：为什么四颗星里只有 GRID-03B 探测到 TGF——读出结构不同。

(a) 同速率条件下"跨探头相邻事例时间间隔"的逐 tick 直方：GRID-03B 四路独立，(0, τ) 连续
    铺满；GRID-02 四路共帧，(0, τ) 几乎为空、dt = 0 处一个尖峰、τ 处一道硬边沿。
    数据由 scripts/cluster/grid_readout_hist.py 在归档上产出（CSV 拉回本地）。
(b) 一个真实的亮短暴（GRID-03B 2022-10-04T00:09:56.5，闪电认证）的 21 个事例，以及把
    同一串事例按未决项 17(3) 的简化帧模型读出后留下的计数与时戳。
(c) 记录里的数字（OPEN-QUESTIONS.md 未决项 17、20，evidence/grid0x/*.md）。

用法:
    python3 scripts/plot_grid_readout_talk.py --data scratch_gbm/grid_talk_v16 \
        --burst scratch_gbm/lc_v13/GRID-03B_20221004T000956543.csv \
        -o scratch_gbm/grid_talk_v16/grid_talk_4_readout.png
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif",
    "axes.unicode_minus": False, "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "lines.linewidth": 2,
})

TICK_US = 2.0 ** -22 * 1e6            # 0.238419 µs，四星相同
C03B, C02 = "#c53030", "#d97706"       # 与 plot_grid_talk.py 的 SAT_COLORS 一致
TAU03B = 20                             # tick，03B 逐探头死时间（未决项 17(2)）
FRAME_US = 28.6102                      # 未决项 17(3) 简化帧模型的帧长（GRID-04 的 120 tick）
PLOT_TICKS = 140


def load_hist(path):
    h = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    return h[:, 1], h[:, 2]


def load_summary(path):
    return {r["key"]: r["value"] for r in csv.DictReader(open(path))}


def edge_tick(cross, lo=5):
    """跨探头直方从零跳起的第一个 tick（跳过 dt = 0 与 tick 1–lo 的零星计数）：
    取第一个计数超过其后 10–30 tick 坪中位一半的位置。"""
    for k in range(lo, cross.size - 40):
        plateau = np.median(cross[k + 10:k + 30])
        if cross[k] > 0.5 * plateau and plateau > 0:
            return k
    return None


def frame_model(t_us, det, frame_us):
    """未决项 17(3) 的简化帧模型：任一路击中开一帧，帧内每路最多 1 个事例、
    全帧共用触发那一击的时戳，帧结束才开下一帧。返回 [(帧时戳, [探头...])]。"""
    o = np.argsort(t_us, kind="stable")
    t_us, det = t_us[o], det[o]
    frames, i = [], 0
    while i < t_us.size:
        t0 = t_us[i]
        got = []
        j = i
        while j < t_us.size and t_us[j] < t0 + frame_us:
            if det[j] not in got:
                got.append(int(det[j]))
            j += 1
        frames.append((t0, got))
        i = j
    return frames


def panel_hist(ax, data):
    c3, _ = load_hist(os.path.join(data, "g03b_hist.csv"))
    c2, _ = load_hist(os.path.join(data, "g02_hist.csv"))
    s3 = load_summary(os.path.join(data, "g03b_summary.csv"))
    s2 = load_summary(os.path.join(data, "g02_summary.csv"))
    tau02 = edge_tick(c2)

    out = {}
    for name, c, s, col, tau in (("GRID-03B", c3, s3, C03B, TAU03B), ("GRID-02", c2, s2, C02, tau02)):
        pairs = float(s["pairs_all"]); cross = float(s["pairs_cross"])
        lam = float(s["events"]) / float(s["live_s"])
        y = c[:PLOT_TICKS] / pairs * 1e6
        x = np.arange(PLOT_TICKS) * TICK_US
        # 零计数的 tick 不能上对数轴：画在底线上，用短竖线标出
        floor = 0.05
        yy = np.where(y > 0, y, np.nan)
        ax.step(x, yy, where="mid", color=col, lw=1.8, label=name)
        z = np.where(y[1:] == 0)[0] + 1
        ax.plot(x[z], np.full(z.size, floor * 1.25), "|", color=col, ms=7, mew=1.2, alpha=0.8)
        # 四路独立的预期：下一个事例来自其他三路的概率密度 = (3/4)·λ·e^{-λt}
        tt = np.arange(1, PLOT_TICKS) * TICK_US * 1e-6
        ax.plot(tt * 1e6, 0.75 * lam * TICK_US * 1e-6 * np.exp(-lam * tt) * 1e6, "--",
                color=col, lw=1.3, alpha=0.8)
        n_open = int(c[1:tau].sum())
        exp_open = pairs * 0.75 * (1 - np.exp(-lam * tau * TICK_US * 1e-6))
        out[name] = dict(n0=int(c[0]), cross=cross, pairs=pairs, lam=lam, tau=tau,
                         n_open=n_open, exp_open=exp_open, s=s)

    ax.set_yscale("log")
    ax.set_ylim(floor, 3e4)
    ax.set_xlim(-0.6, PLOT_TICKS * TICK_US)
    ax.axvline(TAU03B * TICK_US, color=C03B, lw=1.2, ls=":")
    ax.axvline(tau02 * TICK_US, color=C02, lw=1.2, ls=":")
    ax.set_xlabel("跨探头相邻事例的时间间隔 (µs)　　[1 tick = $2^{-22}$ s = 0.2384 µs，四星相同]")
    ax.set_ylabel("每 tick 的对数 / 每百万相邻对")

    o3, o2 = out["GRID-03B"], out["GRID-02"]
    ax.text(TAU03B * TICK_US + 0.25, 1.2e4, "τ = %d tick = %.2f µs\n03B 逐探头死时间" % (TAU03B, TAU03B * TICK_US),
            color=C03B, fontsize=12.5, va="top")
    ax.text(tau02 * TICK_US - 0.25, 1.2e4, "τ = %d tick = %.2f µs\nGRID-02 读出帧长（本样本边沿）" % (tau02, tau02 * TICK_US),
            color=C02, fontsize=12.5, va="top", ha="right")

    ax.annotate("dt = 0（同一时戳）\n03B：%d 对 = 跨探头对的 %.2f%%\nGRID-02：%d 对 = %.2f%%"
                % (o3["n0"], 100 * o3["n0"] / o3["cross"], o2["n0"], 100 * o2["n0"] / o2["cross"]),
                xy=(0, o2["n0"] / o2["pairs"] * 1e6), xytext=(0.9, 3.0e3), fontsize=12.5,
                arrowprops=dict(arrowstyle="->", color="0.3"), va="center")
    ax.text(8.0, 55, "03B 在 (0, τ) = 1–%d tick 内：%d 对（跨探头对的 %.2f%%）\n"
            "连续铺满，坪高与四路独立预期（虚线）一致；\ntick 1 处另有 %d 对的尖峰（来源未查明）"
            % (TAU03B - 1, o3["n_open"], 100 * o3["n_open"] / o3["cross"], int(load_hist(os.path.join(data, "g03b_hist.csv"))[0][1])),
            color=C03B, fontsize=12.5, va="top")
    ax.text(8.0, 1.2, "GRID-02 在 (0, τ) = 1–%d tick 内：只有 %d 对（%.1e），其中 %d 对在边沿前一格 tick %d；\n"
            "四路独立的预期约 %.0f 对 —— 低 %.0f 倍（底部短竖线 = 该 tick 计数为 0）"
            % (tau02 - 1, o2["n_open"], o2["n_open"] / o2["cross"], int(c2[tau02 - 1]), tau02 - 1,
               o2["exp_open"], o2["exp_open"] / max(o2["n_open"], 1)),
            color=C02, fontsize=12.5, va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), frameon=False)
    sa, sb = o3["s"], o2["s"]
    ax.set_title("(a) 同速率本底下的跨探头间隔：03B 连续，GRID-02 在 (0, τ) 是空的", loc="left")
    samp = ("样本：GRID-03B %s，%d 个 10 s 片 / %.0f s / %.2e 事例，四路合计率中位 %.0f c/s；"
            "GRID-02 %s，%d 片 / %.0f s / %.2e 事例，中位 %.0f c/s。\n"
            "条件：EVT_TYPE = 1，只留四路合计率 500–1000 c/s 的 10 s 片；片上标定脉冲判据剔除 0 片。"
            "虚线 = 四路独立预期 (3/4)·λ·exp(−λt)（λ 取本样本实测率）。"
            % (sa["days"], int(sa["slices_rate"]), float(sa["live_s"]), float(sa["events"]), float(sa["rate_median"]),
               sb["days"], int(sb["slices_rate"]), float(sb["live_s"]), float(sb["events"]), float(sb["rate_median"])))
    return out, samp


def panel_burst(ax, burst_csv):
    rows = list(csv.DictReader(open(burst_csv)))
    t = np.array([float(r["dt_ms"]) for r in rows]) * 1e3
    d = np.array([int(r["det"]) for r in rows])
    m = (t > -1) & (t < 100)
    t, d = t[m], d[m]
    frames = frame_model(t, d, FRAME_US)
    n_out = sum(len(g) for _, g in frames)

    # 上半：03B 实测（四路各一行）；下半：简化帧模型读出后
    for k in range(4):
        sel = d == k
        ax.plot(t[sel], np.full(sel.sum(), 7 - k), "o", color=C03B, ms=8, mec="white", mew=0.6)
    for t0, got in frames:
        ax.axvspan(t0, t0 + FRAME_US, ymin=0.02, ymax=0.47, color=C02, alpha=0.10)
        ax.plot([t0, t0], [-0.4, 3.4], color=C02, lw=0.8, alpha=0.6)
        for k in got:
            ax.plot(t0, 3 - k, "s", color=C02, ms=8, mec="white", mew=0.6)
    ax.set_yticks([7, 6, 5, 4, 3, 2, 1, 0])
    ax.set_yticklabels(["探头 0", "探头 1", "探头 2", "探头 3"] * 2, fontsize=12)
    ax.axhline(3.5, color="0.6", lw=0.8)
    ax.set_ylim(-0.6, 8.9)
    ax.set_xlim(-3, 97)
    ax.set_xlabel("相对第一个事例的时间 (µs)")
    ax.text(96, 7.95, "GRID-03B 实记：%d 个事例 / %.1f µs，%d 个不同时戳"
            % (t.size, t.max() - t.min(), np.unique(np.round(t / TICK_US)).size),
            color=C03B, ha="right", fontsize=13)
    ax.text(96, 3.0, "同一串事例按共帧读出：\n%d 帧 × 每路至多 1 个 = %d 个事例，\n只剩 %d 个时戳（帧 1 的 %d 个、帧 2 的 %d 个同戳）"
            % (len(frames), n_out, len(frames), len(frames[0][1]), len(frames[1][1])),
            color=C02, ha="right", va="top", fontsize=12.5)
    ax.set_title("(b) 一个真实的亮短暴：GRID-03B 2022-10-04 00:09:56.5 UTC（闪电认证）", loc="left")
    return t.size, n_out, len(frames)


def panel_records(ax):
    ax.axis("off")
    ax.set_title("(c) 记录里的实测与注入结果", loc="left")
    lines = [
        ("读出结构（未决项 17(3)，四星各一次过境）", None),
        ("  (0, τ) 占比：04 = 4.3×$10^{-4}$、02 = 4.8×$10^{-6}$；独立预期", None),
        ("  的 1/130 与 1/3900。03B 与独立模型相符", None),
        ("帧成本律（未决项 20(5)，04/07/02 实测同一组整数）", None),
        ("  一帧 m 个事例占 120 + 37(m−1) tick；", None),
        ("  输出上限 4/231 $tick^{-1}$ = 72.6 kc/s", None),
        ("有效并进窗 w（未决项 20(2)）：04 2.91 µs、07 3.37 µs", None),
        ("  —— 帧内只有触发后几 µs 的击中共用时戳", None),
        ("窗长 T 内四路合计的计数上限 4(⌊T/τ⌋+1)", None),
        ("  T = 30 µs：03B 28 个，共帧 8 个（= min_number）", None),
        ("  T = 100 µs：03B 84 个，共帧 16 个", None),
        ("把 03B 的暴注入共帧星（未决项 17(5)/20(6)）", None),
        ("  探测率 ÷ 03B 对照 = 0.41–0.44（24 源 mean(D/A)）", None),
        ("  GRID-04 上计数 8–9 的源 0.04、≥ 15 的源 0.99", None),
        ("  （以上两行为单一 τ、整帧收满的旧读出模型）", None),
    ]
    y = 0.98
    for s, _ in lines:
        bold = not s.startswith("  ")
        ax.text(0.0, y, s, transform=ax.transAxes, fontsize=12.3 if not bold else 12.8,
                fontweight="bold" if bold else "normal", va="top",
                color="0.1" if bold else "0.25")
        y -= 0.066


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--burst", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("--slides", action="store_true",
                    help="幻灯片版：只留 (a)(b)，去掉右侧的数字栏和内部记录的脚注")
    a = ap.parse_args()
    if a.slides:
        fig = plt.figure(figsize=(17.5, 12.0))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.0], hspace=0.42)
        ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1])
        out, samp = panel_hist(ax_a, a.data)
        panel_burst(ax_b, a.burst)
        ax_a.text(-0.06, -0.135, samp, transform=ax_a.transAxes, fontsize=11.5, color="0.35", va="top")
        fig.text(0.01, 0.012, "(b) 下半部分按共帧读出的简化模型重放（帧长 28.6 µs，帧内每路至多 1 个、共用触发时刻），只示意机制。",
                 fontsize=13, color="0.35", va="bottom")
        fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.07)
        fig.savefig(a.o, dpi=160)
        print("wrote", a.o)
        return

    fig = plt.figure(figsize=(17.5, 13.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1.0], hspace=0.52, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[1, 2])
    out, samp = panel_hist(ax_a, a.data)
    n_in, n_out, nf = panel_burst(ax_b, a.burst)
    panel_records(ax_c)
    fig.suptitle("只有 GRID-03B 探测到 TGF：四颗星的时戳栅格相同，读出结构不同\n"
                 "03B 四路各自独立（死时间 4.77 µs）；02/04/07 四路共用一个读出帧（120 tick = 28.6 µs），帧内事例共用一个时戳",
                 fontsize=18, y=0.995)
    ax_a.text(-0.06, -0.135, samp, transform=ax_a.transAxes, fontsize=11.5, color="0.35", va="top")
    fig.text(0.01, 0.012,
             "(b) 的共帧读出用未决项 17(3) 的简化帧模型（帧长 28.61 µs、帧内每路至多 1 个、全帧共用触发时戳），只示意机制；"
             "实测的并进窗只有约 3 µs，帧内其余击中约 2/3 丢弃、约 1/5 排进下一帧（readout-and-b-corner.md §2），\n"
             "逐个暴的去留要看注入检验，不看这一张示意。记录：该暴在共帧读出下仍留 8 个计数、fa = 9.7×$10^{-7}$，"
             "但同戳占比 0.500 > 0.35，在 v10 的同戳门下被当成粒子否决（why-grid03b.md §3.1）。",
             fontsize=11.5, color="0.35", va="bottom")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.905, bottom=0.085)
    fig.savefig(a.o, dpi=160)
    print("wrote", a.o)
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "s"})
    print("burst", n_in, "->", n_out, "in", nf, "frames")


if __name__ == "__main__":
    main()
