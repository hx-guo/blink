"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { BASE } from "./format";

export type Loaded<T> = { data: T | null; error: string | null };

/**
 * 运行时取 `blink export` 写出的 JSON（相对 data/）。网页外壳构建时不读数据，
 * 管线更新数据文件即可，无需重建站点。
 */
export function useJson<T>(path: string | null): Loaded<T> {
  // 结果连同它的路径一起存，路径一变旧结果自然作废
  const [state, setState] = useState<Loaded<T> & { path: string | null }>({ path: null, data: null, error: null });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    fetch(`${BASE}/data/${path}`)
      .then((r) => (r.ok ? (r.json() as Promise<T>) : Promise.reject(new Error(`${r.status} ${path}`))))
      .then((data) => alive && setState({ path, data, error: null }))
      .catch((e: Error) => alive && setState({ path, data: null, error: e.message }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state.path === path ? { data: state.data, error: state.error } : { data: null, error: null };
}

function subscribeLocation(onChange: () => void) {
  window.addEventListener("popstate", onChange);
  return () => window.removeEventListener("popstate", onChange);
}

/** 查询参数或路径段里的值：`/tgf/?name=X`，服务器重写时也认 `/tgf/X/` */
export function useParam(key: string, route: string): string | null {
  // 读当前地址；服务端渲染时为 null
  return useSyncExternalStore(
    subscribeLocation,
    () => {
      const q = new URLSearchParams(window.location.search).get(key);
      if (q) return q;
      const path = window.location.pathname.replace(BASE, "");
      const m = path.match(new RegExp(`^/${route}/([^/]+)/?$`));
      return m ? decodeURIComponent(m[1]) : "";
    },
    () => null,
  );
}
