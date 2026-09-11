# 83 个闪电证实 TGF 的叠加光子谱（2026-09-10）

结论见上两级 `OPEN-QUESTIONS.md` 第 25 条。这是把 CALDB 的方向相关响应**正向折叠**
（不是解卷积——响应矩阵不可逆）得到的入射光子谱，与第 18 条的沉积能量谱不是一回事。

- `svom_photon_spectrum.png`：左为全部 83 个（α = 0.88 ± 0.03 stat），
  右为最忙一路死时间 f < 0.3 的 55 个（α = 1.00 ± 0.04 stat）。下排是实测/模型。
- `photon_syst.txt`：系统项逐条扫描。主导项是入射角相关的响应
  （θ < 90° 给 0.69、θ ≥ 90° 给 1.09），已排除方向近似的嫌疑——用 WWLLN 真落点与用
  天底近似给出的 α 几乎一样（0.878 vs 0.869），而正面/背面的差异在两种方向下同样存在。
  次要项是堆积（亮暴发看起来更硬）。**报数要写成 ± 0.03 (stat) ± 0.20 (syst)。**

生成：`scripts/cluster/svom_tgf_response.py`（集群，调官方 RSP_Generator 导出逐 TGF 的
响应与沉积谱）→ `scripts/plot_svom_photon_spectrum.py`（拟合与作图）。
入射方向来自 `scripts/cluster/svom_wwlln_match.py`。
