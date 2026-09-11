# 目录判选在 HXMT 与 SVOM 上的对比（2026-09-07）

支持「SVOM 沿用 HXMT 的两层判选」这一决定（见 OPEN-QUESTIONS 第 16 条）。

- `selection_bins.csv`：两台仪器的候选按 fa 分 120 箱，逐箱的候选数、闪电关联数、误关联期望，
  以及各自覆盖内的曝光秒数。只统计 WWLLN 覆盖内、且已按池级规则去掉列车成员的候选。
  生成：`scripts/cluster/selection_bins.py`（在集群上读两份 tgfs.json）。
- `selection_comparison.png`：`scripts/plot_selection_comparison.py` 画的三面板图。

数据（覆盖内、去列车后）：

| | HXMT/HE | SVOM/GRM |
|---|---|---|
| 曝光 | 2681 天 | 140 天 |
| 候选 | 1656264 | 5797 |
| TGF 段幂律指数 | 0.035 | 0.032 |
| fa ≤ 1e-5：候选 / 关联 / 偶然 | 4521 / 1172（25.9%）/ 13.0 | 197 / 82（41.6%）/ 0.86 |
| 1e-5 < fa ≤ 1：候选 / 关联 / 偶然 | 163058 / 684（0.4%）/ 43.8 | 435 / 49（11.3%）/ 0.67 |
| fa > 1：候选 / 关联 / 偶然 | 1298164 / 498（0.04%）/ 309.5 | 4692 / 13（0.3%）/ 2.18 |

两条边界都得到两台仪器各自数据的支持：1e-5 处关联率陡升，fa > 1 处掉回偶然水平。
第二层的误救占比 HXMT 6.4%、SVOM 1.4%。
SVOM 的关联率整体高于 HXMT，与它有效面积小、只看得见亮 TGF（母闪电也更强、WWLLN 更易探测）一致。

- `fa_correction.txt`：**`fa` 的单边 p 差一个**（第 35 条）。`poisson::sf` 返回
  P(X > count)，正确的是 P(X ≥ count)。SVOM 改正因子中位 **17.69**（λ 中位 0.640）；
  fa ≤ 1e-5 上 897 → 782（出 115 进 0）。**改正因子随 fa 单调漂 4 倍，不是刚性平移**，
  但第 15 条引用的两个幂律指数几乎不动（TGF 段 0.0391 → 0.0398、关联 0.0306 → 0.0298）。
  含两条对账（`sf` 列 vs 复算的 P(X>count) 差 1.9e-14；`sf/fa×年` 反推的
  bin_size_best 最大 0.000999987 s，对上 1 ms 上限）。
  生成：`scripts/cluster/svom_fa_correction.py` 与 `svom_powerlaw_fa_fixed.py`，
  输入必须是 `svomrun8/tgfs.json` 的 count/mean/sf——**不能用特征表的 dur_ms 与
  rate_bkg 重建 λ，那是 merge 后的包络和另一个本底窗，会错两个量级以上**。
