"use client";

import { useEffect, useState } from "react";

/**
 * 元素的实际宽度（像素），图按它画，字号就是屏幕上的字号。返回回调 ref：元素换了（占位 div 换成
 * 图）就改观察新的；宽度为 0（已卸下或未排版）的读数不用。
 */
export function useWidth<T extends Element>(initial = 600) {
  const [el, setEl] = useState<T | null>(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const w = Math.round(e.contentRect.width);
      if (w > 0) setWidth(Math.max(200, w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width] as const;
}

/**
 * 按 viewBox 等比缩放的图（地图、首页直方图）：viewBox 单位对屏幕像素之比。字号、刻度长度乘上它，
 * 屏幕上的字始终 13px；不放大（宽屏上为 1）。
 */
export function useViewScale<T extends Element>(viewWidth: number) {
  const [ref, width] = useWidth<T>(viewWidth);
  return [ref, Math.max(1, viewWidth / width)] as const;
}
