"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { eventHref, sci } from "@/lib/format";

export type Ev = {
  key: string;
  name: string | null;
  cand: string;
  instrument: string;
  provisional: boolean;
  start: string;
  startMs: number;
  duration: number;
  fa: number;
  lon: number;
  lat: number;
  alt: number;
  solar: string;
};

type Key = "startMs" | "duration" | "fa" | "lon" | "lat" | "alt" | "solar";

const COLUMNS: { key: Key | null; title: string; unit?: string; num?: boolean; detector?: boolean }[] = [
  { key: null, title: "Name" },
  { key: null, title: "Detector", detector: true },
  { key: "startMs", title: "Date" },
  { key: null, title: "Time", unit: "UTC" },
  { key: "duration", title: "Duration", unit: "µs", num: true },
  { key: "fa", title: "False-alarm rate", unit: "yr⁻¹", num: true },
  { key: "lat", title: "Position", num: true },
];

/** 3.3×10⁻⁸ → 底数与指数分开排，指数对齐更好读 */
function Sci({ x }: { x: number }) {
  if (!x) return <>–</>;
  const e = Math.floor(Math.log10(x));
  const m = x / 10 ** e;
  return (
    <>
      {m.toFixed(1)}
      <span className="secondary"> × 10</span>
      <sup style={{ display: "inline-block", minWidth: "2em", textAlign: "left", fontSize: "0.72em" }}>{e}</sup>
    </>
  );
}

/** 1 2 3 … p−1 p p+1 … n−2 n−1 n（设计稿的页码样式） */
function pageList(page: number, pages: number): (number | "…")[] {
  const keep = new Set<number>();
  for (const p of [0, 1, 2, page - 1, page, page + 1, pages - 3, pages - 2, pages - 1]) {
    if (p >= 0 && p < pages) keep.add(p);
  }
  const sorted = [...keep].sort((a, b) => a - b);
  const out: (number | "…")[] = [];
  sorted.forEach((p, i) => {
    if (i > 0 && p - sorted[i - 1] > 1) out.push("…");
    out.push(p);
  });
  return out;
}

export default function EventTable({ events, caption, showDetector = true }: { events: Ev[]; caption?: React.ReactNode; showDetector?: boolean }) {
  const columns = COLUMNS.filter((c) => showDetector || !c.detector);
  const [sort, setSort] = useState<{ key: Key; desc: boolean }>({ key: "startMs", desc: true });
  const [size, setSize] = useState(20);
  const [page, setPage] = useState(0);
  useEffect(() => setPage(0), [events, size]);

  const sorted = useMemo(() => {
    const out = [...events];
    out.sort((a, b) => {
      const x = a[sort.key];
      const y = b[sort.key];
      return (x < y ? -1 : x > y ? 1 : 0) * (sort.desc ? -1 : 1);
    });
    return out;
  }, [events, sort]);
  const pages = Math.max(1, Math.ceil(sorted.length / size));
  const shown = sorted.slice(page * size, (page + 1) * size);



  return (
    <div className="flex flex-col gap-4">
      {caption}
      <div className="overflow-x-auto">
        <table className="events tabular">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.title}
                  style={{ cursor: c.key ? "pointer" : "default", textAlign: c.num ? "right" : "left" }}
                  aria-sort={c.key && sort.key === c.key ? (sort.desc ? "descending" : "ascending") : undefined}
                  onClick={() => c.key && setSort((s) => ({ key: c.key!, desc: s.key === c.key ? !s.desc : true }))}
                >
                  {c.title}
                  {c.unit && <span className="unit"> {c.unit}</span>}
                  <span className="sort">{c.key && sort.key === c.key ? (sort.desc ? " ↓" : " ↑") : ""}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((e) => (
              <tr key={e.key}>
                <td>
                  <Link href={eventHref({ name: e.name, cand_id: e.cand })} style={{ fontSize: 16 }}>
                    {e.name ? `TGF ${e.name}` : "pending"}
                  </Link>
                  {e.provisional && <span className="badge ml-2">provisional</span>}
                </td>
                {showDetector && <td>{e.instrument}</td>}
                <td>{e.start.slice(0, 10)}</td>
                <td className="secondary">{e.start.slice(11, 23)}</td>
                <td style={{ textAlign: "right" }}>{Math.round(e.duration).toLocaleString("en")}</td>
                <td style={{ textAlign: "right" }}><Sci x={e.fa} /></td>
                <td style={{ textAlign: "right" }}>
                  {Math.abs(e.lat).toFixed(1)}° {e.lat < 0 ? "S" : "N"}, {Math.abs(e.lon).toFixed(1)}° {e.lon < 0 ? "W" : "E"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between gap-3 flex-wrap text-sm">
        <span className="caption tabular">
          {sorted.length ? `${(page * size + 1).toLocaleString("en")}–${Math.min(sorted.length, (page + 1) * size).toLocaleString("en")}` : "0"} of{" "}
          {sorted.length.toLocaleString("en")}
        </span>
        <nav className="flex items-center gap-1 flex-wrap" aria-label="Pages">
          <button className="page-btn" disabled={page === 0} onClick={() => setPage(page - 1)} aria-label="Previous page">‹</button>
          {pageList(page, pages).map((p, i) =>
            p === "…" ? (
              <span key={`e${i}`} className="muted px-1">…</span>
            ) : (
              <button key={p} className="page-btn" aria-current={p === page ? "page" : undefined} onClick={() => setPage(p)}>
                {p + 1}
              </button>
            ),
          )}
          <button className="page-btn" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)} aria-label="Next page">›</button>
        </nav>
        <span className="flex items-center gap-3">
          <select className="control" value={size} onChange={(e) => setSize(Number(e.target.value))} aria-label="Rows per page">
            {[20, 50, 100].map((s) => <option key={s} value={s}>{s} per page</option>)}
          </select>
        </span>
      </div>
    </div>
  );
}
