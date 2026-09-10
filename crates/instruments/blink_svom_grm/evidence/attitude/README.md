# SVOM/GRM 姿态四元数约定（2026-09-10）

结论与完整推导见上两级的 `OPEN-QUESTIONS.md` 第 22 条，代码里的约定写在
`src/io/att.rs` 的文件头注释。

- `quat_kinematics.txt`：表内自洽检验。att 表自带角速度 `wx/wy/wz`，把四元数按八种
  可能的约定装成旋转矩阵、相邻采样点的相对旋转除以 Δt 必须等于它。四天各自独立跑，
  只取机动样本（惯性指向时四元数每秒只变 1e-6 rad，贴着 float32 量化底、没有信号）。
  正确约定的相对残差中位 0.017–0.023，次优假设差 11–29 倍。
  生成：`scripts/cluster/svom_quat_convention.py`。
- `grb250919a.txt`：外部真值检验。GRB 250919A 由 Fermi-LAT 定在
  RA = 298.52°、Dec = −48.83°（误差半径 0.08°，GCN 41884）。三路 GRD 的实测计数份额
  0.198/0.739/0.064 对 CALDB 折叠预测 0.159/0.759/0.082，χ² = 179；其余七种约定
  最好的 χ² = 4447。生成：`scripts/cluster/svom_grb_check.py`。
- `tgf_incidence_wwlln.csv`：83 个闪电证实 TGF 的真实入射方向（载荷系 θ/φ）、
  WWLLN 落点经纬、落点到星下点的距离、与天底方向的夹角。
  生成：`scripts/cluster/svom_wwlln_match.py`。
