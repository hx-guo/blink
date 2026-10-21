// 站点常量。联系方式为空时页脚不显示该项。
export const site = {
  email: "",
  tel: "",
  /** 轨道倾角（度），地图上画 ±倾角 的虚线 */
  inclination: { "Insight-HXMT/HE": 43, "Fermi/GBM": 25.6, "SVOM/GRM": 30 } as Record<string, number>,
};
