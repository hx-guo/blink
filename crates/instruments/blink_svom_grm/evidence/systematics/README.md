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


- `theta_mechanism.txt`：**θ 项的机理定位**（第 33 条）。三段：① 低能道逐段切掉
  β_θ 从 0.461 塌到 0.070，高能端切掉同样多的道反而涨到 0.741——θ 项住在沉积能
  300 keV 以下；② 42–80 keV 的超出按 θ 分半，正面 0.843 ± 0.122、背面
  1.452 ± 0.119，**差 3.6σ，所以那个软超出是仪器不是源**；③ 把背面响应乘常数或
  乘能量幂律的扰动扫描——数据偏好背面 42–80 keV 面积比 CALDB 大 2–3 倍
  （ΔCash −12.5），但任何能抹平 β_θ 的平滑改正都被数据拒绝（ΔCash +23 到 +36）。
  生成：`scripts/diag_svom_theta_mechanism.py`。
