"use client";

import { useEffect, useState } from "react";
import { BASE } from "./format";

export type Loaded<T> = { data: T | null; error: string | null };

/**
 * 运行时取 `blink export` 写出的 JSON（相对 data/）。网页外壳构建时不读数据，
 * 管线更新数据文件即可，无需重建站点。
 */
export function useJson<T>(path: string | null): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState({ data: null, error: null });
    fetch(`${BASE}/data/${path}`)
      .then((r) => (r.ok ? (r.json() as Promise<T>) : Promise.reject(new Error(`${r.status} ${path}`))))
      .then((data) => alive && setState({ data, error: null }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state;
}

/** 查询参数或路径段里的值：`/tgf/?name=X`，服务器重写时也认 `/tgf/X/` */
export function useParam(key: string, route: string): string | null {
  const [value, setValue] = useState<string | null>(null);
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get(key);
    if (q) return setValue(q);
    const path = window.location.pathname.replace(BASE, "");
    const m = path.match(new RegExp(`^/${route}/([^/]+)/?$`));
    setValue(m ? decodeURIComponent(m[1]) : "");
  }, [key, route]);
  return value;
}
