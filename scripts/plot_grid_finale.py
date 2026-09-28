"""讲稿最后一页的综合图：一张图把整个报告串起来。

左：全部显著超出的星下点。灰色密度 = 本底 > 5000/秒的一类（勾出辐射带和南大西洋异常区），
虚线 = 偶极磁纬 ±33°，红 = A（短、低磁纬，星标有闪电对应），蓝 = B（约 3 毫秒、高磁纬），灰三角 = 中间。
下：地图上圈出的两个例子的光变（① 03B 的 TGF、② 04 的 B 群），以及持续时间 × 磁纬分群。

用法:
    python3 scripts/plot_grid_finale.py <set 目录> <TGF 逐事例 npz> <闪电 csv> <B 例子逐事例 csv> -o <png>
"""
import argparse, csv, datetime as dt, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cjk_font import FAMILIES as CJK_FAMILIES
from plot_grid_talk import load, dipole_lat, T90_CUT_US, MLAT_CUT_DEG
from plot_grid_slides import split

plt.rcParams.update({
    "font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif", "axes.unicode_minus": False,
    "font.size": 17, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
})
RED, BLUE, INK, HAZE, GREEN = "#C53030", "#2B6CB0", "#2A2F38", "#8A94A3", "#2F855A"

# 两个例子：搜索报的候选窗起点
TGF_T0 = "2022-10-04T00:09:56.543791"
B_KEY = ("GRID-04", "2024-03-30T02:45:49.572")

ap = argparse.ArgumentParser()
ap.add_argument("setdir"); ap.add_argument("tgf_npz"); ap.add_argument("wwlln"); ap.add_argument("b_csv")
ap.add_argument("-o", required=True)
a = ap.parse_args()

feat = os.path.join(a.setdir, "features_sig.csv")
d = load(feat, os.path.join(a.setdir, "t90.csv"), sorted(glob.glob(os.path.join(a.setdir, "tgfs_grid*.json"))))
d["rate_search"] = np.array([float(r["mean"]) / (float(r["dur_us"]) * 1e-6) for r in csv.DictReader(open(feat))])
g = split(d)
asc = d["assoc"]

fig = plt.figure(figsize=(22, 14.2))
gs = fig.add_gridspec(2, 3, height_ratios=[2.75, 1], hspace=0.16, wspace=0.24)

# ---------------------------------------------------------------- 地图
ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -85, 80], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#EEF0F3", zorder=0)
ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="#A7AFBB", zorder=1)
lo, la = np.meshgrid(np.linspace(-180, 180, 721), np.linspace(-85, 80, 331))
ml = dipole_lat(la, lo)
ax.contour(lo, la, ml, levels=[-MLAT_CUT_DEG, MLAT_CUT_DEG], colors=INK, linewidths=1.2, linestyles="--",
           transform=ccrs.PlateCarree(), zorder=2)
ax.contour(lo, la, ml, levels=[0], colors=INK, linewidths=0.8, linestyles=":", transform=ccrs.PlateCarree(), zorder=2)
hi = g["hi"]
ax.scatter(d["lon"][hi], d["lat"][hi], s=16, c=HAZE, alpha=0.45, lw=0, transform=ccrs.PlateCarree(), zorder=2)
ax.scatter(d["lon"][g["m"]], d["lat"][g["m"]], s=70, marker="^", c="0.5", lw=0.6, edgecolor="k", transform=ccrs.PlateCarree(), zorder=4)
ax.scatter(d["lon"][g["b"]], d["lat"][g["b"]], s=70, marker="s", c=BLUE, lw=0.6, edgecolor="k", transform=ccrs.PlateCarree(), zorder=4)
ax.scatter(d["lon"][g["a"] & ~asc], d["lat"][g["a"] & ~asc], s=110, c=RED, lw=0.7, edgecolor="k", transform=ccrs.PlateCarree(), zorder=5)
ax.scatter(d["lon"][g["a"] & asc], d["lat"][g["a"] & asc], s=520, marker="*", c=RED, lw=0.8, edgecolor="k", transform=ccrs.PlateCarree(), zorder=6)
ax.text(-178, 41, "磁纬 +33°", fontsize=15, color=INK, transform=ccrs.PlateCarree(), zorder=7)
ax.text(-178, -30, "磁纬 −33°", fontsize=15, color=INK, transform=ccrs.PlateCarree(), zorder=7)
ax.text(95, -61, "灰点：本底很高的超出 %d 个，八成在澳大利亚以南的外辐射带" % int(hi.sum()), fontsize=16, color="#5A6270",
        ha="right", va="center", transform=ccrs.PlateCarree(), zorder=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
handles = [Line2D([], [], marker="o", ls="", ms=12, mfc=RED, mec="k", mew=0.6, label="A 短、低磁纬 %d" % int(g["a"].sum())),
           Line2D([], [], marker="*", ls="", ms=22, mfc=RED, mec="k", mew=0.6, label="其中有闪电对应 %d" % int((g["a"] & asc).sum())),
           Line2D([], [], marker="s", ls="", ms=11, mfc=BLUE, mec="k", mew=0.6, label="B 约 3 毫秒、高磁纬 %d" % int(g["b"].sum())),
           Line2D([], [], marker="^", ls="", ms=12, mfc="0.5", mec="k", mew=0.6, label="中间 %d" % int(g["m"].sum()))]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=18,
          handletextpad=0.3, columnspacing=1.6)


def idx(sat, start):
    return int(np.where((d["sat"] == sat) & (d["start"] == start[:23]))[0][0])


# ---------------------------------------------------------------- 例一：03B 的 TGF
z = np.load(a.tgf_npz)
t = z["t"] * 1e3                                   # 毫秒，零点 = 候选窗起点
ax1 = fig.add_subplot(gs[1, 0])
bins = np.arange(-0.3, 0.5001, 0.01)
ax1.hist(t, bins=bins, color=RED, histtype="stepfilled", alpha=0.9)
bkg = (np.abs(t) > 10) & (np.abs(t) < 600)
ax1.axhline(bkg.sum() / (2 * 590) * 0.01, color=INK, lw=1, ls=":")
tc = dt.datetime.fromisoformat(TGF_T0)
lt = np.array([(dt.datetime.fromisoformat(r["time"]) - tc).total_seconds() * 1e3 for r in csv.DictReader(open(a.wwlln))])
ax1.set_xlim(-0.3, 0.5); ax1.set_xlabel("时间（毫秒）"); ax1.set_ylabel("每 10 微秒计数")
ax1.set_title("① 03B　2022-10-04　19 个光子，72 微秒，有闪电", loc="left", fontsize=17, color=RED)
i = idx("GRID-03B", TGF_T0)
ax.scatter(d["lon"][i], d["lat"][i], s=1500, facecolor="none", edgecolor=RED, lw=2.2, transform=ccrs.PlateCarree(), zorder=8)
ax.text(d["lon"][i] + 5, d["lat"][i] + 6, "①", fontsize=26, color=RED, fontweight="bold", transform=ccrs.PlateCarree(), zorder=9)

# ---------------------------------------------------------------- 例二：04 的 B 群
rows = list(csv.DictReader(open(a.b_csv)))
tb = np.array([float(r["dt_ms"]) for r in rows])
ax2 = fig.add_subplot(gs[1, 1])
bins = np.arange(-15, 15.001, 0.5)
ax2.hist(tb, bins=bins, color=BLUE, histtype="stepfilled", alpha=0.9)
bk = (np.abs(tb) > 8)
ax2.axhline(bk.sum() / (np.ptp(bins) - 16) * 0.5, color=INK, lw=1, ls=":")
ax2.set_xlim(-15, 15); ax2.set_xlabel("时间（毫秒）"); ax2.set_ylabel("每 0.5 毫秒计数")
j = idx(*B_KEY)
ax2.set_title("② 04　2024-03-30　约 %.0f 毫秒，磁纬 %.0f°" % (d["t90"][j] / 1e3, abs(d["mlat"][j])), loc="left", fontsize=17, color=BLUE)
ax.scatter(d["lon"][j], d["lat"][j], s=1500, facecolor="none", edgecolor=BLUE, lw=2.2, transform=ccrs.PlateCarree(), zorder=8)
ax.text(d["lon"][j] - 12, d["lat"][j] - 11, "②", fontsize=26, color=BLUE, fontweight="bold", transform=ccrs.PlateCarree(), zorder=9)

# ---------------------------------------------------------------- 分群
ax3 = fig.add_subplot(gs[1, 2])
ax3.add_patch(Rectangle((20, 0), T90_CUT_US - 20, MLAT_CUT_DEG, facecolor=RED, alpha=0.08, zorder=0))
ax3.add_patch(Rectangle((T90_CUT_US, MLAT_CUT_DEG), 40000, 90, facecolor=BLUE, alpha=0.08, zorder=0))
ax3.scatter(d["t90"][hi], np.abs(d["mlat"][hi]), s=10, c=HAZE, alpha=0.35, lw=0, zorder=1)
ax3.scatter(d["t90"][g["m"]], np.abs(d["mlat"][g["m"]]), s=40, marker="^", c="0.5", lw=0.5, edgecolor="k", zorder=3)
ax3.scatter(d["t90"][g["b"]], np.abs(d["mlat"][g["b"]]), s=40, marker="s", c=BLUE, lw=0.5, edgecolor="k", zorder=3)
ax3.scatter(d["t90"][g["a"] & ~asc], np.abs(d["mlat"][g["a"] & ~asc]), s=55, c=RED, lw=0.5, edgecolor="k", zorder=4)
ax3.scatter(d["t90"][g["a"] & asc], np.abs(d["mlat"][g["a"] & asc]), s=240, marker="*", c=RED, lw=0.6, edgecolor="k", zorder=5)
ax3.axvline(T90_CUT_US, color="0.5", ls="--", lw=1); ax3.axhline(MLAT_CUT_DEG, color="0.5", ls="--", lw=1)
ax3.set_xscale("log"); ax3.set_xlim(20, 40000); ax3.set_ylim(0, 80)
ax3.set_xlabel("持续时间 T90（微秒）"); ax3.set_ylabel("|磁纬|（度）")
ax3.text(30, 3, "A", fontsize=24, color=RED, fontweight="bold")
ax3.text(30000, 72, "B", fontsize=24, color=BLUE, fontweight="bold", ha="right", va="top")

fig.savefig(a.o, dpi=110, bbox_inches="tight")
print("wrote", a.o, {k: int(v.sum()) for k, v in g.items()}, "assoc A", int((g["a"] & asc).sum()), "lightning within 2 ms", lt[np.abs(lt) < 2])
