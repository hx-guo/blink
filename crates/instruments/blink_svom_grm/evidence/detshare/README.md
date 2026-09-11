# 逐 GRD 份额：判别力与响应对照（2026-09-11）

结论见上两级 `OPEN-QUESTIONS.md` 第 29 条。

- `detector_share_auc.txt`：单 GRD 占比作判别量的 AUC（0.473 ± 0.041，等于没有），
  以及与「逐候选按自己的计数抽三路等概率多项分布」基准的对照——实测中位 0.433
  对基准 0.429，就是等照射。**HXMT 的「分组占比」经验在 SVOM 上不成立。**
  生成：`scripts/diag_svom_detector_share.py`。
- `share_vs_response.txt`：83 个闪电证实的份额与 CALDB 在真实入射方向上的预测比，
  拟合 q = (1−ε)/3 + ε·p。**死时间修正会造人为反相关**（f_d 由该探头自己的事例
  算出），所以三种处理并排；自洽处理给 ε = −0.17 ± 0.13，与 0 一致，ε = 1 被
  排除到 ΔlnL = 45。生成：`scripts/diag_svom_share_vs_response.py`。
- `orb_frame_check.txt`：方向链的独立核验。orb 表同时给 `X_J2000` 与 `X_WGS84`
  （任务方端到端算的两套坐标），拿它当真值核地固→惯性那一步：差中位 0.006°、
  最大 0.024°。对照给出这个检验的灵敏度：经度取反差 76°、把 TT 当 UTC 差 0.27°。
  生成：`scripts/diag_svom_orb_frames.py`。
- `azimuth_scan.txt`：方位扫描。在 φ + Δφ（0…330°，步长 30°）与 θ → 180−θ 上用
  CALDB 重算三路有效面积，13 个变体的 ε 全在 −0.28 … +0.25、对数似然只差 2.6
  ——似然面对 Δφ 完全平，**不是方位错位**。
  生成：`scripts/cluster/svom_resp_scan.py` + `scripts/diag_svom_resp_azimuth.py`。
  实测内存峰值约 8.0 GB，农场 `-mem` 取 12000。
