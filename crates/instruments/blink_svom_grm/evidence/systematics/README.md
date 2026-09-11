# 叠加光子谱的系统项拆解与逐 TGF 注量（2026-09-11）

结论见上两级 `OPEN-QUESTIONS.md` 第 27、28 条。第 25 条把系统项写成一个 ±0.20 的
包络，这里把它拆成三支——入射角（仪器）、大气斜程（物理）、堆积（仪器）。

- `photon_systematics.txt`：协变量联合拟合、二维分格、θ 四档、φ/主导探头对照、
  逐能段残差、挡掉增益交接道的影响、注量。
- `tgf_fluence.csv`：83 个闪电证实 TGF 的逐个注量（erg/cm² 与 ph/cm²），
  附入射角、落点距离、源天顶角、空气质量、死时间占比、核心窗计数与脉冲宽度。
- `svom_photon_systematics.png`：α 沿 θ、α 沿 sec ζ、注量分布三张图。

生成：`scripts/plot_svom_photon_systematics.py`，输入是
`scripts/cluster/svom_tgf_response.py` 导出的 npz 与
`scripts/cluster/svom_wwlln_match.py` 的入射方向表。
交接道表由 `scripts/cluster/svom_seam_per_tgf.py` 给。

图里两个折线**只是分档看趋势，分档本身没有互相控制**（sec ζ 那一档因此不单调：
最小的一档里 θ 偏大），标题上的 β 才是三支互相控制后的联合拟合值。
图是用加注前的脚本出的，文字以本 README 与 `photon_systematics.txt` 为准。

