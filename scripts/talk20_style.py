"""Shared palette / rcParams for the talk-20 popular-science slide figures.

Three traces only, one meaning per colour (matches the slide deck):
  NAVY  原始的     — what the instrument actually recorded
  SKY   修复后的   — after filling the lost events (+ dead-time correction)
  RED   其他仪器   — the independent reference instrument
"""
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

NAVY = "#1B3454"   # 原始的（记到的）
SKY = "#4B86C8"    # 修复后的
RED = "#D62728"    # 其他仪器 / 参照
GREY = "#828282"
AXIS = "#C4C9D0"
TICK = "#4A4F57"
FONTS = "/Users/skyair/Developer/ihep/talk-20/fonts/"


def apply(size=13):
    for f in ("Manrope-Light.ttf", "Manrope-Regular.ttf", "Manrope-Medium.ttf"):
        p = FONTS + f
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    home = os.path.expanduser("~/Library/Fonts/")
    for f in ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Light.otf"):
        if os.path.exists(home + f):
            fm.fontManager.addfont(home + f)
    plt.rcParams.update({
        "font.family": ["Manrope", "Noto Sans CJK SC"],
        "font.size": size,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TICK,
        "xtick.color": TICK,
        "ytick.color": TICK,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def clean(ax, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.tick_params(length=3)


def mark(ax, n, x, y, transform=None, size=13.5):
    """Numbered pointer tying a place in the figure to the slide's 看哪里 list."""
    kw = {} if transform is None else {"transform": transform}
    ax.text(x, y, str(n), ha="center", va="center", fontsize=size,
            color="white", fontweight="bold", zorder=20,
            bbox=dict(boxstyle="circle,pad=0.30", fc=NAVY, ec="white", lw=1.4),
            **kw)
