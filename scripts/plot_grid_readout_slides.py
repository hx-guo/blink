"""天格读出结构的讲稿用图：字号按投影放大，图里只留坐标、图例和一两个关键数字，说明放讲稿。

四张图：
  hist    不同探头相邻两个事例的时间间隔（GRID-03B 对 GRID-02，同样计数率）
  burst   同一个 TGF：03B 实际记录，和按 MCU 共帧读出重放后的样子
  buffer  同一颗卫星上的 03B 与 04 同一次过境：整次过境 / 辐射带里 2 秒 / 放大 50 毫秒
  pulser  03B 的星上标定脉冲：几毫秒内的逐事例图，和四路同刻信号的间隔分布

数据：
  --hist-data  scripts/cluster/grid_readout_hist.py 的产物目录（g03b_hist.csv 等）
  --burst      scripts/cluster/grid_lightcurve.py 导出的逐事例 CSV（dt_ms, pi, det）
  --buffer     scripts/cluster/grid_buffer_slide_data.py 的 npz
  --pulser     scripts/cluster/grid_pulser_slide_data.py 的 npz

帧模型与跨探头直方的归一沿用 plot_grid_readout_talk.py。

用法:
    python3 scripts/plot_grid_readout_slides.py --hist-data <dir> --burst <csv> --buffer <npz> --pulser <npz> -o <outdir>
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES

from plot_grid_readout_talk import load_hist, load_summary, edge_tick, frame_model, TICK_US, TAU03B, FRAME_US

plt.rcParams.update({
    "font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif",
    "axes.unicode_minus": False, "font.size": 22, "axes.labelsize": 23,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 21,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK = "#2A2F38"
C03B, C02, C04, GREY = "#D62728", "#E8833A", "#2F6497", "0.55"
PULSER_PERIOD_TICKS_03B = 4194.7672       # 与 blink_grid 搜索里的 pulser_period_ticks 相同
PLOT_TICKS = 140


def fig_hist(data, out):
    fig, ax = plt.subplots(figsize=(16, 6.4))
    c2, _ = load_hist(os.path.join(data, "g02_hist.csv"))
    tau02 = edge_tick(c2)
    for name, f, col in (("GRID-03B（FPGA）", "g03b", C03B), ("GRID-02（MCU）", "g02", C02)):
        c, _ = load_hist(os.path.join(data, f + "_hist.csv"))
        s = load_summary(os.path.join(data, f + "_summary.csv"))
        pairs = float(s["pairs_all"]); lam = float(s["events"]) / float(s["live_s"])
        y = c[:PLOT_TICKS] / pairs * 1e6
        x = np.arange(PLOT_TICKS) * TICK_US
        ax.step(x, np.where(y > 0, y, np.nan), where="mid", color=col, lw=2.4, label=name)
        tt = np.arange(1, PLOT_TICKS) * TICK_US * 1e-6
        ax.plot(tt * 1e6, 0.75 * lam * TICK_US * 1e-6 * np.exp(-lam * tt) * 1e6, "--", color=col, lw=1.5, alpha=0.8)
    ax.plot([], [], "--", color="0.4", lw=1.5, label="四路互相独立时的预期")
    for x, col, txt, ha in ((TAU03B * TICK_US, C03B, " 4.77 微秒", "left"), (tau02 * TICK_US, C02, "28.6 微秒 ", "right")):
        ax.axvline(x, color=col, lw=1.5, ls=":")
        ax.text(x, 1.5e4, txt, color=col, ha=ha, va="top", fontsize=22)
    ax.set_yscale("log"); ax.set_ylim(0.05, 3e4); ax.set_xlim(-0.6, PLOT_TICKS * TICK_US)
    ax.set_xlabel("不同探头的相邻两个事例，间隔多少微秒")
    ax.set_ylabel("相对数量")
    ax.legend(loc="lower center", bbox_to_anchor=(0.55, 0.12), frameon=False, ncol=3, fontsize=20)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_burst(burst_csv, out):
    rows = list(csv.DictReader(open(burst_csv)))
    t = np.array([float(r["dt_ms"]) for r in rows]) * 1e3
    d = np.array([int(r["det"]) for r in rows])
    m = (t > -1) & (t < 100)
    t, d = t[m], d[m]
    frames = frame_model(t, d, FRAME_US)
    n_out = sum(len(g) for _, g in frames)
    fig, ax = plt.subplots(figsize=(16, 6.6))
    for k in range(4):
        sel = d == k
        ax.plot(t[sel], np.full(sel.sum(), 7 - k), "o", color=C03B, ms=11, mec="white", mew=0.8)
    for t0, got in frames:
        ax.axvspan(t0, t0 + FRAME_US, ymin=0.03, ymax=0.46, color=C02, alpha=0.12)
        for k in got:
            ax.plot(t0, 3 - k, "s", color=C02, ms=11, mec="white", mew=0.8)
    ax.axhline(3.5, color="0.7", lw=1)
    ax.set_yticks(range(8)); ax.set_yticklabels(["探头 3", "探头 2", "探头 1", "探头 0"] * 2, fontsize=18)
    ax.set_ylim(-0.7, 8.4); ax.set_xlim(-3, 97)
    n_stamp = np.unique(np.round(t / TICK_US)).size
    ax.text(96, 7.9, "03B 实际记录：%d 个计数，%d 个不同时刻" % (t.size, n_stamp), color=C03B, ha="right", fontsize=22)
    ax.text(96, 3.1, "按 MCU 读出：%d 个计数，%d 个时刻" % (n_out, len(frames)), color=C02, ha="right", va="top", fontsize=22)
    ax.set_xlabel("相对第一个事例的时间（微秒）")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_buffer(npz, out):
    z = np.load(npz)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11.5), gridspec_kw=dict(hspace=0.62))
    ax = axes[0]
    g = z["g03b_gti"]; c = float(z["centre"][0])
    for tag, col, lab in (("g04", C04, "04"), ("g03b", C03B, "03B")):
        s, n = z[tag + "_sec"], z[tag + "_cnt"]
        ax.step(s - g[0], np.where(n > 0, n, np.nan), where="post", color=col, lw=1.6, label=lab)
    ax.axvspan(c - g[0] - 1, c - g[0] + 1, color="0.8", alpha=0.6)
    ax.set_yscale("log"); ax.set_ylabel("每秒计数"); ax.set_xlabel("过境内的时间（秒）")
    ax.set_title("整次过境", loc="left", fontsize=22)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=19)
    ax = axes[1]
    edges = np.arange(-1.0, 1.0 + 1e-9, 1e-3)
    for tag, col in (("g04", C04), ("g03b", C03B)):
        n, _ = np.histogram(z[tag + "_t"], edges)
        ax.step(edges[:-1] * 1e3, n, where="post", color=col, lw=1.0)
    ax.set_xlabel("相对时间（毫秒）"); ax.set_ylabel("每毫秒计数")
    ax.set_title("辐射带里的 2 秒", loc="left", fontsize=22)
    # 放大：2 秒窗里第一段有数据的块，从块起点前 5 ms 起取 50 ms
    t3 = z["g03b_t"]; t3 = t3[(t3 > -1) & (t3 < 1)]
    start = t3.min() * 1e3 - 5
    ax.axvspan(start, start + 50, color="0.8", alpha=0.6)
    ax = axes[2]
    e2 = np.arange(start, start + 50 + 1e-9, 1.0) / 1e3
    n, _ = np.histogram(z["g03b_t"], e2)
    ax.step(e2[:-1] * 1e3 - start, n, where="post", color=C03B, lw=1.4)
    ax.set_xlabel("时间（毫秒）"); ax.set_ylabel("每毫秒计数")
    ax.set_title("放大 50 毫秒", loc="left", fontsize=22)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


def fig_pulser(npz, out, span_ms=7):
    """03B 脉冲开着的时段：若干毫秒逐事例，同一时刻 3 路以上的用竖线连起来。"""
    z = np.load(npz)
    t, d = z["t"], z["det"]
    ticks = np.round(t / (TICK_US * 1e-6)).astype(np.int64)
    u, inv, cnt = np.unique(ticks, return_inverse=True, return_counts=True)
    multi = cnt[inv] >= 3                      # 同一时刻 3 路以上
    clus = u[cnt >= 3]
    fig, a1 = plt.subplots(figsize=(15, 5.6))
    w0 = clus[np.searchsorted(clus, 0)] * TICK_US * 1e-3 - 0.3      # 从 0 之后第一个同刻信号前 0.3 ms 起画
    win = (t * 1e3 > w0) & (t * 1e3 < w0 + span_ms)
    cw = clus * TICK_US * 1e-3 - w0
    for x in cw[(cw > 0) & (cw < span_ms)]:
        a1.plot([x, x], [-0.35, 3.35], color=C03B, lw=1.2, ls=":", alpha=0.7, zorder=1)
    for k in range(4):
        s = win & (d == k) & ~multi
        a1.plot(t[s] * 1e3 - w0, np.full(s.sum(), 3 - k), "o", color=GREY, ms=9, zorder=2)
        s = win & (d == k) & multi
        a1.plot(t[s] * 1e3 - w0, np.full(s.sum(), 3 - k), "o", color=C03B, ms=15, mec="white", mew=1, zorder=3)
    x1, x2 = cw[(cw > 0)][:2]
    a1.annotate("", xy=(x2, 3.75), xytext=(x1, 3.75), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
    a1.text((x1 + x2) / 2, 3.9, "1.000110 毫秒", ha="center", va="bottom", fontsize=22, color=INK)
    a1.set_yticks(range(4)); a1.set_yticklabels(["探头 3", "探头 2", "探头 1", "探头 0"], fontsize=22)
    a1.set_ylim(-0.6, 4.5); a1.set_xlim(0, span_ms); a1.tick_params(axis="x", labelsize=20)
    a1.set_xlabel("时间（毫秒）", fontsize=22)
    a1.plot([], [], "o", color=C03B, ms=15, label="同一时刻 3 路以上"); a1.plot([], [], "o", color=GREY, ms=9, label="其他事例")
    a1.text(span_ms, 3.75, "只画 30 keV 以上的事例", ha="right", va="center", fontsize=20, color=GREY)
    a1.legend(frameon=False, loc="upper right", ncol=2, fontsize=21, bbox_to_anchor=(1.0, 1.08))
    a1.spines["left"].set_visible(False); a1.tick_params(axis="y", length=0)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    gaps = np.diff(clus) / PULSER_PERIOD_TICKS_03B
    on = np.abs(gaps - np.round(gaps)) * PULSER_PERIOD_TICKS_03B <= 1.5
    print("wrote", out, "clusters", clus.size, "on-comb fraction %.4f" % on.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-data"); ap.add_argument("--burst"); ap.add_argument("--buffer"); ap.add_argument("--pulser"); ap.add_argument("--tgf-lc", help="逐事例导出目录（含 ebounds），画讲稿用的 TGF 例子")
    ap.add_argument("--packet-raster", help="03B 某段 ±0.7 s 逐事例 npz，画单路丢包例子")
    ap.add_argument("--windowing", help="弱 TGF 的 ±0.7 s 逐事例 npz，画三种框窗口的对比")
    ap.add_argument("--pair", help="03B/04 同时刻逐事例 npz 目录（<tag>_03b.npz、<tag>_04.npz）")
    ap.add_argument("--coverage", help="每天有效观测时长表 sat,date,searched_s")
    ap.add_argument("--lightning", help="±1 分钟闪电表，画闪电对应判据图")
    ap.add_argument("--tgf-detail", nargs=2, metavar=("NPZ", "WWLLN_CSV"), help="grid_event_window.py 的 npz 与 ±1 分钟闪电表，照论文 detail 图画")
    ap.add_argument("-o", "--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.hist_data:
        fig_hist(a.hist_data, os.path.join(a.outdir, "slide_readout_hist.png"))
        fig_hist2(a.hist_data, os.path.join(a.outdir, "slide_readout_hist2.png"))
    if a.burst: fig_burst(a.burst, os.path.join(a.outdir, "slide_readout_burst.png"))
    if a.buffer:
        fig_buffer(a.buffer, os.path.join(a.outdir, "slide_buffer.png"))
        fig_buffer2(a.buffer, os.path.join(a.outdir, "slide_buffer2.png"))
    if a.pulser: fig_pulser(a.pulser, os.path.join(a.outdir, "slide_pulser.png"))
    if a.packet_raster: fig_packet_raster(a.packet_raster, os.path.join(a.outdir, "slide_packet_raster.png"))
    if a.windowing:
        fig_windowing2(a.windowing, os.path.join(a.outdir, "slide_windowing.png"))
        fig_ssa_steps(a.windowing, os.path.join(a.outdir, "slide_ssa_steps.png"))
    if a.pair: fig_same_sat(a.pair, os.path.join(a.outdir, "slide_same_sat.png"))
    if a.coverage: fig_coverage(a.coverage, os.path.join(a.outdir, "slide_coverage.png"))
    if a.lightning: fig_lightning_match(a.lightning, os.path.join(a.outdir, "slide_lightning_match.png"))
    if a.tgf_detail: fig_tgf_detail(a.tgf_detail[0], a.tgf_detail[1], os.path.join(a.outdir, "slide_tgf_detail.png"))
    if a.tgf_lc: fig_tgf_example(a.tgf_lc, os.path.join(a.outdir, "slide_tgf_example.png"))


def fig_tgf_example(lc_dir, out):
    """讲稿用的一个 TGF：左栏 ±8 ms 全貌，右栏同一个暴放大到 ±100 µs；上光变、下逐事例能量。

    读数与 T90/暴中心沿用 plot_grid_talk._load_example（准入与搜索一致，≥30 keV）。
    """
    from plot_grid_talk import _load_example, EXAMPLES
    ex = EXAMPLES[0]
    t, energy, rate, inside, t90, excess, bkg_energy = _load_example(lc_dir, ex)
    tu = t * 1e3                                  # 微秒
    med_b = float(np.median(bkg_energy))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), gridspec_kw=dict(width_ratios=[1, 1.15], hspace=0.42, wspace=0.28),
                             sharex="col")
    # 左栏：±8 ms，200 µs 一格
    ax = axes[0, 0]
    e = np.arange(-8, 8.001, 0.2)
    ax.hist(t, bins=e, color=C03B)
    ax.axhline(rate * 0.2e-3, color="0.4", ls="--", lw=1.4)
    ax.axvspan(-0.1, 0.1, color="0.85", zorder=0)
    ax.set_ylabel("每 200 微秒计数")
    ax.set_title("前后 8 毫秒", loc="left", fontsize=24)
    ax = axes[1, 0]
    far = ~inside
    ax.scatter(t[far], energy[far], s=30, color=GREY)
    ax.scatter(t[inside], energy[inside], s=60, color=C03B, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(med_b, color="0.4", ls="--", lw=1.4)
    ax.axvspan(-0.1, 0.1, color="0.85", zorder=0)
    ax.set_yscale("log"); ax.set_ylim(25, 3000)
    ax.set_xlim(-8, 8); ax.set_xlabel("时间（毫秒）"); ax.set_ylabel("沉积能量（keV）")
    # 右栏：±100 µs，5 µs 一格
    ax = axes[0, 1]
    e = np.arange(-100, 100.001, 5)
    ax.hist(tu, bins=e, color=C03B)
    ax.axhline(rate * 5e-6, color="0.4", ls="--", lw=1.4)
    ax.set_ylabel("每 5 微秒计数")
    ax.set_title("放大：前后 100 微秒", loc="left", fontsize=24)
    ax.text(0.98, 0.95, "80 微秒内 %d 个" % int(((tu > -45) & (tu < 45)).sum()), transform=ax.transAxes,
            ha="right", va="top", color=C03B, fontsize=22)
    ax = axes[1, 1]
    w = np.abs(tu) <= 100
    ax.scatter(tu[w & far], energy[w & far], s=40, color=GREY)
    ax.scatter(tu[w & inside], energy[w & inside], s=110, color=C03B, edgecolor="white", lw=0.8, zorder=3)
    ax.axhline(med_b, color="0.4", ls="--", lw=1.4)
    ax.text(-97, med_b * 0.82, "本底中位 %.0f keV" % med_b, va="top", fontsize=19, color="0.35")
    ax.set_yscale("log"); ax.set_ylim(25, 3000)
    ax.set_xlim(-100, 100); ax.set_xlabel("时间（微秒）")
    for a in axes.flat:
        a.tick_params(labelbottom=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "T90 %.1f us, inside %d, excess %.1f" % (t90 * 1e3, inside.sum(), excess))


def fig_tgf_detail(npz, wwlln_csv, out, lon0=152.91, lat0=-11.96, dur_us=78.7):
    """照 hxmt-catalog 论文 detail 图的结构画一个天格 TGF：
    左：星下点、800 km 半径、±1 分钟内 WWLLN 闪电（±5 ms 内的用星标）；
    右：三层光变，±0.5 ms（叠逐事例能量）/ ±50 ms / ±500 ms，计数率单位 c/s，浅色底标搜索窗。
    时间零点是搜索报的候选窗起点。
    """
    import datetime as dt
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    from cartopy.geodesic import Geodesic
    z = np.load(npz)
    t, pi = z["t"], z["pi"]
    emin, emax = z["emin"], z["emax"]
    energy = np.sqrt(emin[pi - 1] * emax[pi - 1])
    bkg = (np.abs(t) > 0.01) & (np.abs(t) < 0.6)
    rate = bkg.sum() / (2 * (0.6 - 0.01))
    rows = list(csv.DictReader(open(wwlln_csv)))
    tc = dt.datetime.fromisoformat("2022-10-04T00:09:56.543791")
    lt = np.array([(dt.datetime.fromisoformat(r["time"]) - tc).total_seconds() for r in rows])
    la = np.array([float(r["lat"]) for r in rows]); lo = np.array([float(r["lon"]) for r in rows])

    fig = plt.figure(figsize=(19, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.75], hspace=0.75, wspace=0.2)
    ax = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax.set_extent([lon0 - 14, lon0 + 14, lat0 - 13, lat0 + 13], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.92"); ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="0.45")
    circ = Geodesic().circle(lon0, lat0, 800e3, 180)
    ax.plot(circ[:, 0], circ[:, 1], color="0.55", lw=1.5, transform=ccrs.PlateCarree(), label="800 公里半径")
    near = np.abs(lt) <= 0.005
    ax.scatter(lo[~near], la[~near], s=22, color="#2CA02C", alpha=0.7, transform=ccrs.PlateCarree(), label="±1 分钟内的闪电")
    ax.scatter(lo[near], la[near], s=420, marker="*", color="#2CA02C", edgecolor="k", lw=0.8, zorder=5,
               transform=ccrs.PlateCarree(), label="±5 毫秒内的闪电")
    ax.plot(lon0, lat0, "x", color=C04, ms=16, mew=3, transform=ccrs.PlateCarree(), label="星下点")
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="0.8"); gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 16}
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=17, handletextpad=0.3)

    # (左端, 右端, 格宽, 单位, 换算, 标题)；最上层只取暴附近，TGF 才占得满横轴
    panels = [(-100e-6, 200e-6, 10e-6, "微秒", 1e6, "放大：10 微秒一格"),
              (-50e-3, 50e-3, 1e-3, "毫秒", 1e3, "前后 50 毫秒：1 毫秒一格"),
              (-0.5, 0.5, 10e-3, "毫秒", 1e3, "前后 0.5 秒：10 毫秒一格")]
    from matplotlib.ticker import FuncFormatter
    kfmt = FuncFormatter(lambda v, _: "%g 千" % (v / 1e3) if v else "0")
    for i, (x0, x1, bw, unit, sc, title) in enumerate(panels):
        a = fig.add_subplot(gs[i, 1])
        e = np.arange(x0, x1 + bw / 2, bw)
        n, _ = np.histogram(t, e)
        a.stairs(n / bw, e * sc, color=C03B, lw=2.0, fill=False)
        a.axhline(rate, color="0.5", ls="--", lw=1.3)
        a.axvspan(0, dur_us * 1e-6 * sc, color="#E8833A", alpha=0.18, lw=0)
        a.set_xlim(x0 * sc, x1 * sc)
        a.yaxis.set_major_formatter(kfmt)
        a.set_ylabel("计数/秒", fontsize=18)
        a.tick_params(labelsize=17)
        a.set_xlabel("相对时间（%s）" % unit, fontsize=18)
        a.set_title(title, loc="left", fontsize=19)
        if i == 0:
            a2 = a.twinx()
            w = (t >= x0) & (t <= x1)
            a2.scatter(t[w] * sc, energy[w], s=90, facecolor="none", edgecolor="#1B3454", lw=1.6, zorder=4)
            a2.set_yscale("log"); a2.set_ylim(25, 3000); a2.set_ylabel("能量（keV）", fontsize=18)
            a2.tick_params(labelsize=17); a2.spines["right"].set_visible(True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "bkg rate %.0f c/s, lightning ±5ms %d" % (rate, near.sum()))


def fig_lightning_match(wwlln_csv, out, lon0=152.91, lat0=-11.96, t0="2022-10-04T00:09:56.543791"):
    """闪电对应判据的图：前后 1 分钟全球 WWLLN 闪电，横轴与 TGF 的时间差（对称对数），
    纵轴离星下点的距离（对数）；±5 ms × 800 km 的框里才算对上。"""
    import datetime as dt
    rows = list(csv.DictReader(open(wwlln_csv)))
    tc = dt.datetime.fromisoformat(t0)
    dts = np.array([(dt.datetime.fromisoformat(r["time"]) - tc).total_seconds() for r in rows])
    la = np.radians([float(r["lat"]) for r in rows]); lo = np.radians([float(r["lon"]) for r in rows])
    c = np.sin(la) * np.sin(np.radians(lat0)) + np.cos(la) * np.cos(np.radians(lat0)) * np.cos(lo - np.radians(lon0))
    dist = 6371 * np.arccos(np.clip(c, -1, 1))
    ok = (np.abs(dts) <= 5e-3) & (dist <= 800)
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.add_patch(plt.Rectangle((-5, 30), 10, 770, facecolor="#2CA02C", alpha=0.15, edgecolor="#2CA02C", lw=2))
    ax.scatter(dts[~ok] * 1e3, dist[~ok], s=22, color=GREY, alpha=0.6, lw=0)
    ax.scatter(dts[ok] * 1e3, dist[ok], s=520, marker="*", color="#2CA02C", edgecolor="k", lw=1, zorder=5)
    ax.set_xscale("symlog", linthresh=5); ax.set_yscale("log")
    ax.set_xlim(-7e4, 7e4); ax.set_ylim(30, 25000)
    ax.set_xticks([-6e4, -1e3, -5, 0, 5, 1e3, 6e4]); ax.set_xticklabels(["−60 秒", "−1 秒", "", "0", "", "1 秒", "60 秒"])
    ax.set_yticks([100, 800, 3000, 10000]); ax.set_yticklabels(["100", "800", "3000", "10000"])
    ax.set_xlabel("闪电与 TGF 的时间差"); ax.set_ylabel("闪电离星下点的距离（公里）")
    ax.text(6, 55, "±5 毫秒\n800 公里内", color="#2CA02C", fontsize=20, va="bottom")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "n", len(rows), "matched", int(ok.sum()), "within 800 km", int((dist <= 800).sum()))


def fig_coverage(daily_csv, out):
    """四颗星的观测时间线：每天一条竖线，颜色是当天的有效观测小时数（搜索的 searched_seconds）。"""
    import datetime as dt
    from matplotlib.colors import Normalize
    rows = [r for r in csv.DictReader(open(daily_csv)) if float(r["searched_s"]) > 0]
    sats = ["GRID-02", "GRID-03B", "GRID-04", "GRID-07"]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    norm = Normalize(0, 12); cmap = plt.get_cmap("viridis")
    for i, s in enumerate(sats):
        rs = [r for r in rows if r["sat"] == s]
        d = [dt.date.fromisoformat(r["date"]) for r in rs]
        h = np.array([float(r["searched_s"]) / 3600 for r in rs])
        y = len(sats) - 1 - i
        ax.vlines(d, y - 0.34, y + 0.34, colors=cmap(norm(h)), lw=1.4)
    ax.set_yticks(range(len(sats))); ax.set_yticklabels(sats[::-1], fontsize=21)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "GRID-03B": lab.set_color(C03B); lab.set_fontweight("bold")
    ax.set_ylim(-0.6, len(sats) - 0.4)
    ax.set_xlim(dt.date(2020, 10, 1), dt.date(2024, 10, 1))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03); cb.set_label("每天有效观测（小时）", fontsize=19)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


# 03B 与 04 的时间戳固定差约 460 µs：用 03B 同戳 ≥3 路的粒子簇叠加 04 事例，2022 年（36.5 万簇）与
# 2024 年（14 万簇）同时击中峰都在 +440…+480 µs、宽 ≤ 40 µs（scripts/cluster/grid_pair_lag.py）。
# 与 WWLLN 闪电（加传播时间）比，是 03B 的时间戳整体早约 0.45 ms，04 与闪电一致（OPEN-QUESTIONS §26）。
G04_OFFSET_US = 460.0


def fig_same_sat(pair_dir, out, tags=("20221004T000956", "20220502T205135")):
    """同一颗卫星上的 03B 与 04 对同一个 TGF 的记录：上 03B、下 04，同一时间轴，50 µs 一格。
    时间零点是 03B 搜索窗起点（grid_event_window.py 的 t0）；04 的时间减去 G04_OFFSET_US 对齐。"""
    import datetime as dt
    fig, axes = plt.subplots(2, len(tags), figsize=(16, 7.4), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.18))
    e = np.arange(-500, 2000.1, 50)
    for j, tag in enumerate(tags):
        for i, (sfx, col, name) in enumerate((("03b", C03B, "03B（FPGA）"), ("04", C04, "04（MCU）"))):
            t = np.load(os.path.join(pair_dir, "%s_%s.npz" % (tag, sfx)))["t"] * 1e6
            if sfx == "04": t = t - G04_OFFSET_US
            rate = ((np.abs(t) > 1e4) & (np.abs(t) < 6e5)).sum() / 1.18
            n, _ = np.histogram(t, e)
            a = axes[i, j]
            a.stairs(n, e / 1e3, color=col, lw=2.2, fill=True, alpha=0.85)
            a.axhline(rate * 50e-6, color="0.4", ls="--", lw=1.2)
            a.set_ylim(0, 17)
            a.text(0.98, 0.9, "%s　0–200 微秒 %d 个" % (name, ((t >= -20) & (t < 200)).sum()), transform=a.transAxes,
                   ha="right", va="top", color=col, fontsize=20)
            if j == 0: a.set_ylabel("每 50 微秒计数", fontsize=18)
            a.tick_params(labelsize=17)
        d = dt.datetime.strptime(tag, "%Y%m%dT%H%M%S")
        axes[0, j].set_title(d.strftime("%Y-%m-%d %H:%M:%S UTC"), loc="left", fontsize=19)
        axes[1, j].set_xlabel("相对 03B 暴开始的时间（毫秒）", fontsize=18)
    fig.text(0.5, -0.02, "两台按 0.46 毫秒的固定时差对齐（同时打到两台的带电粒子量出）；和闪电比，是 03B 的时钟早了", ha="center", fontsize=17, color="0.4")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


def fig_windowing(npz, out, burst=(0.0, 68.4), phase=28.5):
    """同一个弱 TGF 的几个光子，三种框窗口的办法：固定 100 µs 格（格界切在暴中间）、
    固定 1 ms 格、窗口两头落在光子上。每行右边写每年误报次数（泊松 P(X≥n)，本底取 ±0.6 s）。"""
    from scipy.stats import poisson
    z = np.load(npz); t = z["t"] * 1e6
    rate = ((np.abs(t) > 1e4) & (np.abs(t) < 6e5)).sum() / 1.18
    fa = lambda n, T: poisson.sf(n - 1, rate * T * 1e-6) * 3.156e7 / (T * 1e-6)
    x0, x1 = -150, 330
    w = (t > x0) & (t < x1)
    inb = (t >= burst[0] - 0.05) & (t <= burst[1] + 0.05)
    fig, axes = plt.subplots(4, 1, figsize=(16, 8.6), sharex=True, gridspec_kw=dict(hspace=0.25, height_ratios=[1, 1, 1, 1]))
    def fmt(v):
        e = int(np.floor(np.log10(v))); m = v / 10 ** e
        return "%.0f" % v if v >= 1 else r"$%.0f\times10^{%d}$" % (m, e)
    rows = [("光子", None), ("100 微秒一格", None), ("1 毫秒一格", None), ("窗口框住这几个光子", None)]
    for i, a in enumerate(axes):
        a.set_ylim(0, 1); a.set_yticks([]); a.spines["left"].set_visible(False)
        a.vlines(t[w & ~inb], 0.25, 0.75, color=GREY, lw=2.5)
        a.vlines(t[w & inb], 0.25, 0.75, color=C03B, lw=3)
        a.text(-0.01, 0.5, rows[i][0], transform=a.transAxes, ha="right", va="center", fontsize=21)
        if i == 0:   # 同一时刻到达的光子在图上叠成一根线，右边写明
            u, c = np.unique(np.round(t[inb], 1), return_counts=True)
            a.text(1.01, 0.5, "红：%d 个光子\n（%d 处各有两个同时到达）" % (inb.sum(), (c > 1).sum()),
                   transform=a.transAxes, va="center", fontsize=19, color=C03B)
    # 100 µs
    a = axes[1]
    edges = np.arange(-1000 + phase, 1000, 100)
    n, _ = np.histogram(t, edges)
    for e0, c in zip(edges[:-1], n):
        if x0 - 100 < e0 < x1:
            a.axvspan(e0, e0 + 100, ymin=0.08, ymax=0.92, facecolor="#2F6497" if c else "none", alpha=0.12, edgecolor="0.5", lw=1)
            if c: a.text(e0 + 50, 0.9, "%d 个" % c, ha="center", va="top", fontsize=18, color="#2F6497")
    k = n.max()
    a.text(1.01, 0.5, "最多 %d 个\n每年误报 %s 次" % (k, fmt(fa(k, 100))), transform=a.transAxes, va="center", fontsize=20, color=GREY)
    # 1 ms
    a = axes[2]
    a.axvspan(x0, x1, ymin=0.08, ymax=0.92, facecolor="#2F6497", alpha=0.10, edgecolor="0.5", lw=1)
    m = ((t >= -500) & (t < 500)).sum()
    a.text(1.01, 0.5, "%d 个\n每年误报 %s 次" % (m, fmt(fa(m, 1000))), transform=a.transAxes, va="center", fontsize=20, color=GREY)
    a.text(x1 - 5, 0.88, "这一格比整张图还宽", ha="right", va="top", fontsize=16, color="0.4")
    # SSA
    a = axes[3]
    a.axvspan(burst[0] - 1.5, burst[1] + 1.5, ymin=0.08, ymax=0.92, facecolor=C03B, alpha=0.15, edgecolor=C03B, lw=1.5)
    s = inb.sum()
    a.text(1.01, 0.5, "%d 个 / %.0f 微秒\n每年误报 %s 次" % (s, burst[1] - burst[0], fmt(fa(s, burst[1] - burst[0]))),
           transform=a.transAxes, va="center", fontsize=20, color=C03B, fontweight="bold")
    axes[-1].set_xlim(x0, x1); axes[-1].set_xlabel("时间（微秒）", fontsize=20); axes[-1].tick_params(labelsize=17)
    for a in axes[:-1]: a.tick_params(bottom=False)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "rate %.0f" % rate, "100us max", k, "1ms", m, "ssa", s, fa(s, burst[1]-burst[0]))


def fig_windowing2(npz, out, burst=(0.0, 68.4), phase=14.0):
    """讲稿版：同一个弱 TGF 的光子，三种框窗口的办法。光子画成圆点，同一时刻的叠放；
    格子深浅交替、粗边框，格内写计数；右侧写每年误报次数和“看不出来 / 显著”。"""
    from scipy.stats import poisson
    from matplotlib.patches import FancyBboxPatch, Rectangle
    z = np.load(npz); t = z["t"] * 1e6
    rate = ((np.abs(t) > 1e4) & (np.abs(t) < 6e5)).sum() / 1.18
    fa = lambda n, T: poisson.sf(n - 1, rate * T * 1e-6) * 3.156e7 / (T * 1e-6)
    x0, x1 = -120, 300
    inb = (t >= burst[0] - 0.05) & (t <= burst[1] + 0.05)
    vis = (t > x0) & (t < x1)
    BLUE, LBLUE, DBLUE = "#2F6497", "#E3ECF6", "#C9DAEC"

    def fmt(v):
        if v >= 1: return "%.0f" % v
        e = int(np.floor(np.log10(v))); m = v / 10 ** e
        return r"%.0f\times10^{%d}" % (m, e)

    def photons(ax, y):
        u, c = np.unique(np.round(t[vis], 1), return_counts=True)
        for x, k in zip(u, c):
            red = (x >= burst[0] - 0.05) and (x <= burst[1] + 0.05)
            for j in range(k):
                ax.plot(x, y + 0.13 * j - 0.065 * (k - 1), "o", ms=13, color=C03B if red else "0.6", mec="white", mew=1.2, zorder=6)

    fig = plt.figure(figsize=(17, 7.6))
    ax = fig.add_axes([0.2, 0.12, 0.53, 0.84])
    ax.set_xlim(x0, x1); ax.set_ylim(-0.3, 3.0)
    ys = [2.45, 1.45, 0.45]
    labels = ["100 微秒一格", "1 毫秒一格", "窗口两头落在光子上"]
    # 行 1：100 µs 格
    edges = np.arange(-1000 + phase, 1000, 100)
    n, _ = np.histogram(t, edges)
    k = 0
    for i, (e0, c) in enumerate(zip(edges[:-1], n)):
        if e0 + 100 < x0 or e0 > x1: continue
        ax.add_patch(Rectangle((e0, ys[0] - 0.36), 100, 0.72, facecolor=DBLUE if i % 2 else LBLUE, edgecolor=BLUE, lw=2, zorder=1))
        if c: ax.text(e0 + 50, ys[0] + 0.3, "%d 个" % c, ha="center", va="top", fontsize=19, color=BLUE, fontweight="bold", zorder=7)
        k = max(k, c)
    # 行 2：1 ms 格
    m = ((t >= -500) & (t < 500)).sum()
    ax.add_patch(Rectangle((x0, ys[1] - 0.36), x1 - x0, 0.72, facecolor=LBLUE, edgecolor=BLUE, lw=2, zorder=1, clip_on=False))
    ax.annotate("", xy=(x0 + 4, ys[1] - 0.2), xytext=(x0 + 45, ys[1] - 0.2), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2), annotation_clip=False)
    ax.annotate("", xy=(x1 - 4, ys[1] - 0.2), xytext=(x1 - 45, ys[1] - 0.2), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2), annotation_clip=False)
    ax.text((x0 + x1) / 2, ys[1] + 0.3, "一格 1 毫秒，向两边还各延伸约 400 微秒　·　格里共 %d 个" % m, ha="center", va="top", fontsize=17, color=BLUE)
    # 行 3：SSA
    s = inb.sum(); T = burst[1] - burst[0]
    ax.add_patch(Rectangle((burst[0] - 3, ys[2] - 0.36), T + 6, 0.72, facecolor="#FBE3E3", edgecolor=C03B, lw=2.5, zorder=1))
    for y in ys: photons(ax, y)
    ax.axis("off")
    ax.plot([x0, x1], [-0.12, -0.12], color="0.3", lw=1.2)
    for xt in range(-100, 301, 100):
        ax.plot([xt, xt], [-0.12, -0.17], color="0.3", lw=1.2); ax.text(xt, -0.2, str(xt), ha="center", va="top", fontsize=17)
    ax.text((x0 + x1) / 2, -0.38, "时间（微秒）", ha="center", va="top", fontsize=18)
    # 左侧行名、右侧结论
    res = [(k, 100, "最多一格 %d 个" % k), (m, 1000, "%d 个 / 1 毫秒" % m), (s, T, "%d 个 / %.0f 微秒" % (s, T))]
    for y, lab, (cnt, TT, desc) in zip(ys, labels, res):
        v = fa(cnt, TT); ok = v <= 1e-5
        yf = ax.transData.transform((0, y))[1] / fig.bbox.height
        fig.text(0.19, yf, lab, ha="right", va="center", fontsize=21, fontweight="bold", color=INK)
        fig.text(0.75, yf + 0.03, desc, ha="left", va="center", fontsize=19, color=INK)
        fig.text(0.75, yf - 0.03, r"每年误报 $%s$ 次" % fmt(v), ha="left", va="center", fontsize=19, color=INK)
        fig.text(0.962, yf, "显著" if ok else "看不出来", ha="center", va="center", fontsize=19, fontweight="bold",
                 color="white", bbox=dict(boxstyle="round,pad=0.45", facecolor=C03B if ok else "0.62", edgecolor="none"))
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25); plt.close(fig); print("wrote", out, k, m, s)


def fig_ssa_steps(npz, out, min_n=8):
    """讲稿版快照步进：上层以第一个光子为起点逐个延长窗口（不到 min_n 个不检验，其余标每年误报），
    最后一行示意换下一个光子当起点；下层是前后 0.6 s 的真实光变，标出本底窗与挖空区。"""
    from scipy.stats import poisson
    from matplotlib.patches import Rectangle
    z = np.load(npz); t = np.sort(z["t"] * 1e6)
    rate = ((np.abs(t) > 1e4) & (np.abs(t) < 6e5)).sum() / 1.18
    fa = lambda n, T: poisson.sf(n - 1, rate * T * 1e-6) * 3.156e7 / (T * 1e-6)
    BLUE, LBLUE = "#2F6497", "#E3ECF6"

    def fmt(v):
        e = int(np.floor(np.log10(v))); m = v / 10 ** e
        return r"$%.0f\times10^{%d}$" % (m, e) if v < 1 else "%.0f" % v

    fig = plt.figure(figsize=(21, 7.4))
    ax = fig.add_axes([0.04, 0.08, 0.52, 0.84])
    x0, x1 = -25, 225
    w = t[(t > x0) & (t < x1)]
    u, c = np.unique(np.round(w, 1), return_counts=True)
    ends = u[u > 0.5]
    rows = len(ends) + 1
    ytop = rows + 0.3
    # 光子（顶行）
    for x, k in zip(u, c):
        for j in range(k):
            ax.plot(x, ytop + 0.28 * j - 0.14 * (k - 1), "o", ms=15, color=C03B if x < 100 else "0.6", mec="white", mew=1.2, zorder=6)
    best = None
    for i, e in enumerate(ends):
        y = rows - i - 0.4
        n = int(c[u <= e + 0.05].sum()); T = e
        tested = n >= min_n
        v = fa(n, T) if tested else None
        if tested and (best is None or v < best[0]): best = (v, i)
    for i, e in enumerate(ends):
        y = rows - i - 0.4
        n = int(c[u <= e + 0.05].sum())
        tested = n >= min_n
        v = fa(n, e) if tested else None
        isbest = best is not None and i == best[1]
        col = C03B if isbest else (BLUE if tested else "0.78")
        ax.add_patch(Rectangle((0, y - 0.28), e, 0.56, facecolor=col, alpha=0.85 if tested else 1, edgecolor="none", zorder=3))
        ax.plot([e, e], [y - 0.28, ytop - 0.35], color="0.8", lw=0.8, ls=":", zorder=1)
        lab = "%d 个" % n + ("　不检验" if not tested else "　每年误报 %s 次" % fmt(v))
        ax.text(e + 4, y, lab, va="center", fontsize=22, color=col if tested else "0.5", fontweight="bold" if isbest else "normal")
    # 换下一个起点
    y = -0.4
    s2 = u[1]
    ax.add_patch(Rectangle((s2, y - 0.28), u[2] - s2, 0.56, facecolor="0.78", edgecolor="none"))
    ax.text(u[2] + 4, y, "换下一个光子当起点，再来一遍……", va="center", fontsize=22, color="0.5")
    ax.annotate("", xy=(0, ytop - 0.45), xytext=(-18, ytop - 0.45), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.8))
    ax.text(-20, ytop - 0.45, "起点", ha="right", va="center", fontsize=21, color="0.3")
    ax.set_xlim(x0, x1 + 130); ax.set_ylim(-1, ytop + 0.6); ax.axis("off")
    ax.plot([0, x1], [-0.95, -0.95], color="0.3", lw=1.2)
    for xt in range(0, 201, 50):
        ax.plot([xt, xt], [-0.95, -1.0], color="0.3", lw=1.2); ax.text(xt, -1.05, str(xt), ha="center", va="top", fontsize=19)
    ax.text(x1 + 8, -0.95, "微秒", va="center", fontsize=19)
    # 下层：本底窗
    b = fig.add_axes([0.62, 0.2, 0.37, 0.55])
    e = np.arange(-0.6, 0.6001, 0.01)
    n, _ = np.histogram(t * 1e-6, e)
    b.axvspan(-0.5, -0.005, color=LBLUE, zorder=0); b.axvspan(0.005, 0.5, color=LBLUE, zorder=0)
    b.stairs(n, e, color="0.35", lw=1.2)
    b.axvline(0, color=C03B, lw=3)
    b.set_xlim(-0.6, 0.6); b.set_ylim(0, n.max() * 1.25)
    b.set_yticks([]); b.spines["left"].set_visible(False)
    b.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); b.set_xticklabels(["−0.5", "", "0", "", "0.5 秒"], fontsize=21)
    b.text(-0.25, n.max() * 1.18, "前 0.5 秒", ha="center", va="top", fontsize=21, color=BLUE)
    b.text(0.3, n.max() * 1.18, "后 0.5 秒", ha="center", va="top", fontsize=21, color=BLUE)
    b.annotate("候选窗\n两边各挖掉 5 毫秒", xy=(0.004, n.max() * 0.62), xytext=(0.07, n.max() * 0.62),
               fontsize=19, color=C03B, va="center", arrowprops=dict(arrowstyle="-|>", color=C03B, lw=1.5),
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
    b.set_title("本底怎么取：前后各 0.5 秒（10 毫秒一格）", loc="left", fontsize=23)
    ax.set_title("窗口怎么走", loc="left", fontsize=23)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2); plt.close(fig)
    print("wrote", out, "rate %.0f" % rate)


def fig_hist2(data, out):
    """两栏：左为同一路探头相邻两个事例的间隔（03B 在 4.77 µs 处的硬边 = 每路死时间），
    右为不同探头相邻两个事例的间隔（03B 从 0 平铺 = 四路互相独立；02 在 28.6 µs 内为空 = 共帧）。"""
    c2, _ = load_hist(os.path.join(data, "g02_hist.csv"))
    tau02 = edge_tick(c2)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for j, (col_idx, title) in enumerate(((2, "同一路探头"), (1, "不同探头"))):
        ax = axes[j]
        for name, f, col in (("GRID-03B（FPGA）", "g03b", C03B), ("GRID-02（MCU）", "g02", C02)):
            h = np.loadtxt(os.path.join(data, f + "_hist.csv"), delimiter=",", skiprows=1, dtype=np.int64)
            s = load_summary(os.path.join(data, f + "_summary.csv"))
            pairs = float(s["pairs_all"])
            y = h[:PLOT_TICKS, col_idx] / pairs * 1e6
            x = np.arange(PLOT_TICKS) * TICK_US
            ax.step(x, np.where(y > 0, y, np.nan), where="mid", color=col, lw=2.4, label=name)
        marks = [(TAU03B * TICK_US, C03B, " 4.77 微秒", "left")] if j == 0 else []
        for xv, col, txt, ha in marks + [(tau02 * TICK_US, C02, "28.6 微秒 ", "right")]:
            ax.axvline(xv, color=col, lw=1.5, ls=":")
            ax.text(xv, 1.5e4, txt, color=col, ha=ha, va="top", fontsize=21)
        ax.set_yscale("log"); ax.set_ylim(0.05, 3e4); ax.set_xlim(-0.6, PLOT_TICKS * TICK_US)
        ax.set_title(title, loc="left", fontsize=24)
        ax.set_xlabel("相邻两个事例的间隔（微秒）")
    axes[0].set_ylabel("相对数量")
    axes[1].legend(loc="center", bbox_to_anchor=(0.5, 0.35), frameon=False, fontsize=20)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


def fig_buffer2(npz, out, vertical=True):
    """讲稿版缓冲区图，横排三级放大，数字标在图上（均从数据数出）：
    ① 整次过境（1 秒一格，03B 与 04）；② 辐射带里 2 秒（1 毫秒一格）：16 批一组、组间停约 340 毫秒；
    ③ 放大 50 毫秒（0.25 毫秒一格）：每批的事例数。"""
    from matplotlib.patches import ConnectionPatch
    z = np.load(npz)
    t = np.sort(z["g03b_t"]); d = np.diff(t)
    fcut = np.where(d > 0.5e-3)[0]; fs = np.r_[0, fcut + 1]; fe = np.r_[fcut, len(t) - 1]
    bcut = np.where(d > 0.1)[0]; bs = np.r_[0, bcut + 1]; be = np.r_[bcut, len(t) - 1]
    if vertical:
        fig, axes = plt.subplots(3, 1, figsize=(12, 13.5), gridspec_kw=dict(hspace=0.55))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6.4), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15], wspace=0.28))
    # ① 整次过境
    ax = axes[0]; g = z["g03b_gti"]; c = float(z["centre"][0])
    for tag, col, lab in (("g04", C04, "04（MCU）"), ("g03b", C03B, "03B（FPGA）")):
        s, n = z[tag + "_sec"], z[tag + "_cnt"]
        ax.step(s - g[0], np.where(n > 0, n, np.nan), where="post", color=col, lw=1.6, label=lab)
    xc = c - g[0]
    ax.axvline(xc, color="0.3", lw=1.2, ls="--")
    ax.annotate("放大这里 2 秒", xy=(xc, 3e4), xytext=(xc - 700, 2e5), fontsize=18, color="0.3",
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))
    ax.set_yscale("log"); ax.set_ylim(15, 4e5); ax.set_xlabel("过境内的时间（秒）"); ax.set_ylabel("每秒计数")
    ax.set_title("① 整次过境", loc="left", fontsize=23); ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=17)
    # ② 2 秒
    ax = axes[1]
    e = np.arange(-1.0, 1.0 + 1e-9, 1e-3); n, _ = np.histogram(t, e)
    ax.fill_between(e[:-1] * 1e3, n, step="post", color=C03B, lw=0)
    full = [i for i in range(len(bs)) if be[i] - bs[i] + 1 == 20160 and t[bs[i]] > -1 and t[be[i]] < 1]
    k = full[0]; b0, b1 = t[bs[k]] * 1e3, t[be[k]] * 1e3
    ax.plot([b0, b1], [385, 385], color=INK, lw=2); ax.plot([b0, b0], [370, 400], color=INK, lw=2); ax.plot([b1, b1], [370, 400], color=INK, lw=2)
    ax.text((b0 + b1) / 2, 400, "16 批 = 20160 个\n约 %.0f 毫秒" % (b1 - b0), ha="center", va="bottom", fontsize=17, color=INK)
    nxt = t[be[k] + 1] * 1e3 if be[k] + 1 < len(t) else b1 + 340
    ax.plot([b1, nxt], [150, 150], color=C04, lw=2); ax.plot([b1, b1], [135, 165], color=C04, lw=2); ax.plot([nxt, nxt], [135, 165], color=C04, lw=2)
    ax.text((b1 + nxt) / 2, 165, "停约 %.0f 毫秒" % (nxt - b1), ha="center", va="bottom", fontsize=17, color=C04)
    z0 = b0 - 2; ax.axvspan(z0, z0 + 50, color="0.75", alpha=0.6, zorder=0)
    ax.set_xlim(-1000, 1000); ax.set_ylim(0, 520); ax.set_xlabel("时间（毫秒）"); ax.set_ylabel("每毫秒计数")
    ax.set_title("② 辐射带里的 2 秒", loc="left", fontsize=23)
    # ③ 50 毫秒
    ax3 = axes[2]
    e3 = np.arange(z0, z0 + 50 + 1e-9, 0.25) / 1e3; n3, _ = np.histogram(t, e3)
    ax3.fill_between((e3[:-1] * 1e3 - z0), n3 * 4, step="post", color=C03B, lw=0)
    for a_, b_ in zip(fs, fe):
        x0, x1 = t[a_] * 1e3 - z0, t[b_] * 1e3 - z0
        if x0 >= 0 and x1 <= 50:
            ax3.text((x0 + x1) / 2, 420, "%d" % (b_ - a_ + 1), ha="center", va="bottom", fontsize=16, color=INK)
    ax3.set_xlim(0, 50); ax3.set_ylim(0, 520); ax3.set_xlabel("时间（毫秒）"); ax3.set_ylabel("每毫秒计数")
    ax3.set_title("③ 放大灰色的 50 毫秒：每一小块是一批，上面是它的事例数", loc="left", fontsize=21)

    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "block", k, "%.0f ms" % (b1 - b0), "gap %.0f" % (nxt - b1))


def fig_packet_raster(npz, out, win=(16.75, 18.35), det_gap=3):
    """03B 单路丢包的一个例子：四路各一行逐事例竖线，一路两次沉默，沉默之间正好记下整数包。
    npz 必须是全部事例（grid_event_window_all.py），打包按全部事例计数。"""
    z = np.load(npz); t, d = z["t"], z["det"]
    tk = np.sort(t[d == det_gap]); rate = tk.size / (tk[-1] - tk[0])
    g = np.diff(tk); idx = np.where(g * rate > 20)[0]
    idx = idx[(tk[idx] > win[0]) & (tk[idx + 1] < win[1])]
    fig, ax = plt.subplots(figsize=(15, 5.2))
    for k in range(4):
        w = (d == k) & (t > win[0]) & (t < win[1])
        ax.vlines(t[w], 3 - k - 0.32, 3 - k + 0.32, color=C03B if k == det_gap else "0.45", lw=1.0)
    y = 3 - det_gap
    for i in idx:
        ax.axvspan(tk[i], tk[i + 1], color=C03B, alpha=0.10, lw=0)
        ax.text((tk[i] + tk[i + 1]) / 2, 3.55, "空 %.2f 秒" % (tk[i + 1] - tk[i]), ha="center", fontsize=19, color=C03B)
    for i, j in zip(idx[:-1], idx[1:]):
        n = j - i; x0, x1 = tk[i + 1], tk[j]
        ax.annotate("", xy=(x1, y - 0.55), xytext=(x0, y - 0.55),
                    arrowprops=dict(arrowstyle="|-|", color=C03B, lw=2, mutation_scale=6))
        ax.text((x0 + x1) / 2, y - 0.72, "这一路记下 %d 个 = %d × 20" % (n, n // 20),
                ha="center", va="top", fontsize=21, color=C03B, fontweight="bold")
    ax.set_yticks(range(4)); ax.set_yticklabels(["探头 3", "探头 2", "探头 1", "探头 0"], fontsize=19)
    ax.set_ylim(y - 1.25, 3.85); ax.set_xlim(*win)
    ax.set_xlabel("时间（秒）", fontsize=19); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "rate %.0f/s, 20/rate %.3f s" % (rate, 20 / rate), [(round(tk[i], 3), round(tk[i + 1] - tk[i], 3)) for i in idx])


if __name__ == "__main__":
    main()
