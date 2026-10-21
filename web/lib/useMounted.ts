"use client";

import { useEffect, useState } from "react";

/**
 * 图表只在浏览器里画：静态导出时每页都内联一遍地图 SVG 会让 4000 多个事件页
 * 涨到 1.5 GB；地图数据放在共享的 JS 里，下载一次。
 */
export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}
