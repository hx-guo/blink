"use client";

import { useEffect, useRef, useState } from "react";

/** 元素的实际宽度（像素），图按它画，字号就是屏幕上的字号 */
export function useWidth<T extends Element>(initial = 600) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}
