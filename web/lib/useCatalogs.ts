"use client";

import { useEffect, useState } from "react";
import { BASE } from "./format";
import type { Catalog } from "./types";

/** 运行时取各仪器的列式目录（几百 KB 到几 MB，不塞进 HTML） */
export function useCatalogs(slugs: string[]): Catalog[] | null {
  const [catalogs, setCatalogs] = useState<Catalog[] | null>(null);
  const key = slugs.join(",");
  useEffect(() => {
    let alive = true;
    Promise.all(
      key.split(",").filter(Boolean).map((s) => fetch(`${BASE}/data/${s}/catalog.json`, { cache: "no-cache" }).then((r) => r.json() as Promise<Catalog>)),
    ).then((c) => alive && setCatalogs(c));
    return () => {
      alive = false;
    };
  }, [key]);
  return catalogs;
}

/** 列式 → 行对象 */
export function rows(cat: Catalog): Record<string, string | number | null>[] {
  const cols = Object.keys(cat.columns);
  return Array.from({ length: cat.n }, (_, i) => Object.fromEntries(cols.map((c) => [c, cat.columns[c][i]])));
}
