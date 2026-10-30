/**
 * 各仪器能看到的区域：轨道倾角给出星下点的纬度范围，SAA 多边形内探测器关机。都是卫星运行用的
 * 固定参数，不从数据推断。
 */
export type Coverage = {
  /** 轨道倾角，度；星下点纬度在 ±inclination 之内 */
  inclination: number;
  /** SAA 多边形顶点 [经度 −180..180, 纬度]，度，按顺序连成直线（等距圆柱图上） */
  saa: [number, number][];
};

export const COVERAGE: Record<string, Coverage> = {
  "Insight-HXMT/HE": {
    inclination: 43,
    // HE 关机的 SAA 区域，与 HXMT TGF 目录论文的图同一组顶点
    saa: [
      [-74.3, -45], [-88.2, -28], [-96, -13], [-92, -9], [-70, -2.5], [-45, 3], [-33, 2.1],
      [-15, -15], [0.8, -18.8], [18.2, -23], [31, -31], [27.3, -39], [22, -45],
    ],
  },
};
