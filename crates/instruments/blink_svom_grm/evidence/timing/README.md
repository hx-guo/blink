# `met()` 微秒截断的量化与能阈证据复核（2026-09-10）

结论见上两级 `OPEN-QUESTIONS.md` 第 23 条。

- `met_audit.txt`：901 个显著候选逐个重算窗内计数，与搜索报的 `count` 对账。
  最佳格口径下现行 trunc 写法只有 51.4% 一致，exact 99.8%，exact + 整数纳秒 100%。
  末尾附那 2 个 f64 对不上的候选的整数纳秒复核。
  生成：`scripts/cluster/svom_met_audit.py`。
- `thr_compare.txt`：能阈证据修 met() 前后的逐档对照（输入完全相同：svomrun5 的
  795 个显著候选）。S 上升 1.0–2.0%、B 变化 < 0.5%、S/√B 的最优点没有移动。
- `thr_scan_fixed.csv` / `spectra_fixed.csv`：修好之后重跑的结果，与上一级
  `threshold/thr_scan.csv`、`threshold/spectra.csv`（截断版）逐行对应。
  生成：`scripts/cluster/svom_spectra.py`（已修）。
