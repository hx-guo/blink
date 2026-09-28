"""天格 TGF 讲稿用的干净版结果图：不带标题和注释字，字号按投影放大，说明放在幻灯片上。

分类口径与 plot_grid_talk.py 相同（从那里导入）：T90 < 2 ms 且 |偶极磁纬| < 33° 为 A 角，
T90 ≥ 2 ms 且 |偶极磁纬| ≥ 33° 为 B 角，其余为中间带。另外把候选窗本底率（搜索报的 mean / 窗长）> 5000 计数/秒
的单列一类"高本底"，不并入三类：搜索已不设本底上限，这一类要单独露出来看。

用法:
    python3 scripts/plot_grid_slides.py <features.csv> <t90.csv> <tgfs_*.json ...> -o <目录>
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plot_grid_talk import load, SAT_COLORS, T90_CUT_US, MLAT_CUT_DEG, RED, BLUE

plt.rcParams.update({
    "font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif",
    "axes.unicode_minus": False, "font.size": 20, "axes.labelsize": 22,
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAVY = "#1B3454"
HIGH_RATE = 5000.0


def split(d):
    # 用搜索自己报的候选窗本底率（mean / 窗长）划分，和搜索输出的口径一致；
    # t90 表里重算的 rate_bkg 取的是不同的本底窗，边界附近会差一两个。
    hi = d["rate_search"] > HIGH_RATE
    return dict(hi=hi, a=d["corner_a"] & ~hi, b=d["corner_b"] & ~hi, m=d["middle"] & ~hi)


def fig_populations(d, out):
    g = split(d)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.add_patch(Rectangle((20, 0), T90_CUT_US - 20, MLAT_CUT_DEG, facecolor=RED, alpha=0.07, zorder=0))
    ax.add_patch(Rectangle((T90_CUT_US, MLAT_CUT_DEG), 40000, 90 - MLAT_CUT_DEG,
                           facecolor=BLUE, alpha=0.07, zorder=0))
    asc = d["assoc"]
    for sat, c in SAT_COLORS.items():
        sel = (d["sat"] == sat) & ~asc & ~g["hi"]
        ax.scatter(d["t90"][sel], np.abs(d["mlat"][sel]), s=90, c=c, lw=0.6, edgecolor="k",
                   alpha=0.9, zorder=4)
    if g["hi"].any():
        ax.scatter(d["t90"][g["hi"]], np.abs(d["mlat"][g["hi"]]), s=40, marker="x",
                   c="0.55", lw=1.2, zorder=3)
    ax.scatter(d["t90"][asc], np.abs(d["mlat"][asc]), s=380, marker="*", c="#f6e05e",
               lw=0.9, edgecolor="k", zorder=6)
    ax.axvline(T90_CUT_US, color="0.5", ls="--", lw=1.4)
    ax.axhline(MLAT_CUT_DEG, color="0.5", ls="--", lw=1.4)
    ax.set_xscale("log"); ax.set_xlim(20, 40000); ax.set_ylim(0, 82)
    ax.set_xlabel("持续时间 T90（微秒）"); ax.set_ylabel("|磁纬|（度）")
    ax.text(26, 30.5, "A：短、低磁纬\n%d 个" % int(g["a"].sum()), fontsize=22, color=RED,
            va="top", ha="left", linespacing=1.4, fontweight="bold")
    ax.text(38000, 30.5, "B：约 3 毫秒、高磁纬\n%d 个（在虚线上方）" % int(g["b"].sum()), fontsize=22,
            color=BLUE, va="top", ha="right", linespacing=1.4, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", ms=12, mfc=c, mec="k", mew=0.6,
                      label="%s（%d）" % (sat, int(((d["sat"] == sat) & ~g["hi"]).sum())))
               for sat, c in SAT_COLORS.items() if ((d["sat"] == sat) & ~g["hi"]).any()]
    handles.append(Line2D([], [], marker="*", ls="", ms=20, mfc="#f6e05e", mec="k", mew=0.7,
                          label="有闪电对应（%d）" % int(asc.sum())))
    if g["hi"].any():
        handles.append(Line2D([], [], marker="x", ls="", ms=10, mec="0.55", mew=1.4,
                              label="本底 > 5000/秒（%d）" % int(g["hi"].sum())))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig); print("wrote", out)


def fig_map(d, out):
    g = split(d)
    asc = d["assoc"]
    fig = plt.figure(figsize=(16, 7.4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -85, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.93")
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.5")
    if g["hi"].any():
        ax.scatter(d["lon"][g["hi"]], d["lat"][g["hi"]], s=30, marker="x", c="0.6", lw=1.0,
                   transform=ccrs.PlateCarree(), zorder=3)
    kinds = [(g["b"], "s", 80, BLUE), (g["m"], "^", 80, "0.45"),
             (g["a"] & ~asc, "o", 95, RED), (g["a"] & asc, "*", 420, RED)]
    for sel, marker, size, c in kinds:
        ax.scatter(d["lon"][sel], d["lat"][sel], s=size, marker=marker, c=c, lw=0.6,
                   edgecolor="k", alpha=0.9, transform=ccrs.PlateCarree(),
                   zorder=6 if marker == "*" else 4)
    handles = [Line2D([], [], marker="o", ls="", ms=12, mfc=RED, mec="k", mew=0.6,
                      label="A：短、低磁纬（%d）" % int(g["a"].sum())),
               Line2D([], [], marker="*", ls="", ms=22, mfc=RED, mec="k", mew=0.6,
                      label="其中有闪电对应（%d）" % int((g["a"] & asc).sum())),
               Line2D([], [], marker="s", ls="", ms=11, mfc=BLUE, mec="k", mew=0.6,
                      label="B：约 3 毫秒、高磁纬（%d）" % int(g["b"].sum())),
               Line2D([], [], marker="^", ls="", ms=12, mfc="0.45", mec="k", mew=0.6,
                      label="中间（%d）" % int(g["m"].sum()))]
    if g["hi"].any():
        handles.append(Line2D([], [], marker="x", ls="", ms=10, mec="0.6", mew=1.4,
                              label="本底 > 5000/秒（%d）" % int(g["hi"].sum())))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(handles) if len(handles) <= 4 else 3, frameon=False, fontsize=19,
              handletextpad=0.3, columnspacing=1.4)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features"); ap.add_argument("t90"); ap.add_argument("tgfs", nargs="+")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = load(args.features, args.t90, args.tgfs)
    import csv
    d["rate_search"] = np.array([float(r["mean"]) / (float(r["dur_us"]) * 1e-6)
                                 for r in csv.DictReader(open(args.features))])
    g = split(d)
    for k in ("a", "b", "m", "hi"):
        sats = {s: int(((d["sat"] == s) & g[k]).sum()) for s in SAT_COLORS}
        print(k, int(g[k].sum()), sats, "assoc", int((g[k] & d["assoc"]).sum()))
    fig_populations(d, os.path.join(args.outdir, "slide_populations.png"))
    fig_map(d, os.path.join(args.outdir, "slide_map.png"))


if __name__ == "__main__":
    main()
