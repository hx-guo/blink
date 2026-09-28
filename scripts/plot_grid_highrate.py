"""天格显著候选里本底 > 5000 计数/秒 的那一类：地理分布，颜色为候选窗本底率。

输入是从搜索输出直接导出的全部 fa ≤ 1e-5 候选（sat,start,fa,count,mean,dur_us,rate,lon,lat），
rate = mean / 窗长。讲稿用，不带标题。

用法:
    python3 scripts/plot_grid_highrate.py <all_sig.csv> -o <png>
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams.update({"font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif",
                     "axes.unicode_minus": False, "font.size": 20})
HIGH_RATE = 5000.0

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("-o", required=True)
a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
lon = np.array([float(r["lon"]) for r in rows]); lat = np.array([float(r["lat"]) for r in rows])
rate = np.array([float(r["rate"]) for r in rows])
hi = rate > HIGH_RATE
fig = plt.figure(figsize=(16, 7.6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -85, 80], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="0.93"); ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.5")
ax.scatter(lon[~hi], lat[~hi], s=40, c="0.55", lw=0, transform=ccrs.PlateCarree(), zorder=3,
           label="其余 %d 个显著候选" % int((~hi).sum()))
sc = ax.scatter(lon[hi], lat[hi], s=46, c=rate[hi], cmap="plasma", norm=LogNorm(5e3, 1e5), lw=0.3,
                edgecolor="k", transform=ccrs.PlateCarree(), zorder=4,
                label="本底 > 5000/秒：%d 个" % int(hi.sum()))
cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.025, pad=0.01)
cb.set_label("候选窗的本底（计数/秒）")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=20)
fig.savefig(a.o, dpi=150, bbox_inches="tight"); print("wrote", a.o, int(hi.sum()), int((~hi).sum()))
