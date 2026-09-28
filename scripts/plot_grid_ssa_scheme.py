"""讲稿用的快照步进示意图（示意，不按比例）：上为窗口怎么取，下为和本底怎么比。

用法: python3 scripts/plot_grid_ssa_scheme.py -o <png>
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif", "axes.unicode_minus": False})
INK, RED, BLUE = "#1E2430", "#A33B2B", "#2F6497"

ap = argparse.ArgumentParser(); ap.add_argument("-o", required=True)
out = ap.parse_args().o

fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 7.6), gridspec_kw=dict(height_ratios=[1.0, 0.8], hspace=0.35))
rng = np.random.default_rng(3)   # 与原示意图同一组随机光子
bg = np.sort(np.concatenate([rng.uniform(0, 1.3, 7), rng.uniform(1.95, 3.0, 7)]))
burst = 1.45 + np.sort(rng.uniform(0, 0.4, 9))
ev = np.sort(np.concatenate([bg, burst]))
a1.set_xlim(-0.05, 3.05); a1.set_ylim(-0.3, 3.4); a1.axis("off")
a1.vlines(ev, 0, 0.6, color=INK, lw=2.2); a1.plot([-0.05, 3.05], [0, 0], color=INK, lw=1)
i0 = np.searchsorted(ev, 1.45)
for k, y in zip((3, 5, 8), (1.0, 1.65, 2.3)):
    x0, x1 = ev[i0], ev[i0 + k]
    a1.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="|-|", color=RED, lw=2.2, mutation_scale=8))
    a1.text(x1 + 0.04, y - 0.13, f"{k + 1} 个光子", fontsize=20, color=RED)
a1.text(ev[i0], 2.95, "以这个光子为起点，逐个光子把窗口往后延长（最长 1 毫秒）", fontsize=21, color=RED)
a1.plot([ev[i0], ev[i0]], [0.65, 2.3], color=RED, lw=1, ls=":")
a1.set_title("① 候选窗口怎么取（示意，不按比例）", loc="left", fontsize=23, color=INK)

a2.set_xlim(-0.62, 0.62); a2.set_ylim(-0.25, 2.2); a2.axis("off")
a2.add_patch(Rectangle((-0.5, 0), 1.0, 0.8, color=BLUE, alpha=0.18))
a2.add_patch(Rectangle((-0.06, 0), 0.12, 0.8, color="#FAFAF7"))
a2.add_patch(Rectangle((-0.012, 0), 0.024, 0.8, color=RED, alpha=0.85))
a2.plot([-0.6, 0.6], [0, 0], color=INK, lw=1)
a2.text(-0.5, 1.0, "−0.5 秒", fontsize=20, ha="center", color=BLUE)
a2.text(0.5, 1.0, "+0.5 秒", fontsize=20, ha="center", color=BLUE)
a2.text(-0.28, 0.4, "本底率 = 计数 ÷ 有数据的时长", fontsize=19, ha="center", va="center", color=BLUE)
a2.text(0.28, 0.4, "期望计数 λ = 本底率 × 窗长", fontsize=20, ha="center", va="center", color=BLUE)
a2.annotate("中间挖掉 ±5 毫秒，\n免得暴本身被算进本底", xy=(-0.06, 0.8), xytext=(-0.34, 1.45), fontsize=20, color=INK,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
a2.annotate("候选窗（≤ 1 毫秒）：\n实际计数 n", xy=(0.012, 0.8), xytext=(0.12, 1.45), fontsize=20, color=RED,
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
a2.set_title("② 和本底怎么比", loc="left", fontsize=23, color=INK)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
