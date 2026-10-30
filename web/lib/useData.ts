"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { BASE } from "./format";

export type Loaded<T> = { data: T | null; error: string | null };

/** 同一个文件正在取时，后来的调用等同一次请求（页头和页面都要 index.json） */
const inflight = new Map<string, Promise<unknown>>();
export function getJson<T>(path: string): Promise<T> {
  let p = inflight.get(path);
  if (!p) {
    p = fetch(`${BASE}/data/${path}`, { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${path}`))))
      .finally(() => inflight.delete(path));
    inflight.set(path, p);
  }
  return p as Promise<T>;
}

/**
 * 运行时取 `blink export` 写出的 JSON（相对 data/）。网页外壳构建时不读数据，
 * 管线更新数据文件即可，无需重建站点。数据每小时更新：`no-cache` 让浏览器每次向服务器确认
 * （没变就是一个 304），不按自己估的新鲜期用旧文件。
 */
export function useJson<T>(path: string | null): Loaded<T> {
  // 结果连同它的路径一起存，路径一变旧结果自然作废
  const [state, setState] = useState<Loaded<T> & { path: string | null }>({ path: null, data: null, error: null });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    getJson<T>(path)
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
