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
  { key: "startMs", title: "Start", unit: "UTC" },
  { key: "duration", title: "Duration", unit: "µs", num: true },
  { key: "fa", title: "False-alarm rate", unit: "yr⁻¹", num: true },
  { key: "lat", title: "Latitude", num: true },
  { key: "lon", title: "Longitude", num: true },
  { key: "alt", title: "Altitude", unit: "km", num: true },
  { key: "solar", title: "Solar time", num: true },
];

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
  const [size, setSize] = useState(100);
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

  const download = () => {
    const blob = new Blob([JSON.stringify(sorted.map(({ key: _key, startMs: _ms, ...e }) => e), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "tgf-catalog.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="flex flex-col gap-4">
      {caption}
      <div className="overflow-x-auto card">
        <table className="data events tabular">
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
                  {c.key && sort.key === c.key ? (sort.desc ? " ↓" : " ↑") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((e) => (
              <tr key={e.key}>
                <td>
                  <Link href={eventHref({ name: e.name, cand_id: e.cand })} className="font-semibold">
                    {e.name ? `TGF ${e.name}` : "pending"}
                  </Link>
                  {e.provisional && <span className="badge ml-2">provisional</span>}
                </td>
                {showDetector && <td>{e.instrument}</td>}
                <td>{e.start.slice(0, 10)} <span className="secondary">{e.start.slice(11, 23)}</span></td>
                <td style={{ textAlign: "right" }}>{Math.round(e.duration)}</td>
                <td style={{ textAlign: "right" }}>{sci(e.fa)}</td>
                <td style={{ textAlign: "right" }}>{Math.abs(e.lat).toFixed(2)}°{e.lat < 0 ? "S" : "N"}</td>
                <td style={{ textAlign: "right" }}>{Math.abs(e.lon).toFixed(2)}°{e.lon < 0 ? "W" : "E"}</td>
                <td style={{ textAlign: "right" }}>{e.alt.toFixed(0)}</td>
                <td style={{ textAlign: "right" }}>{e.solar.slice(0, 5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <select className="page-btn" style={{ padding: "0 10px" }} value={size} onChange={(e) => setSize(Number(e.target.value))} aria-label="Rows per page">
          {[20, 50, 100].map((s) => <option key={s} value={s}>{s} per page</option>)}
        </select>
        <nav className="flex items-center gap-2 flex-wrap" aria-label="Pages">
          <button className="page-btn" disabled={page === 0} onClick={() => setPage(page - 1)} aria-label="Previous page">‹</button>
          {pageList(page, pages).map((p, i) =>
            p === "…" ? (
              <span key={`e${i}`} className="muted">···</span>
            ) : (
              <button key={p} className="page-btn" aria-current={p === page ? "page" : undefined} onClick={() => setPage(p)}>
                {p + 1}
              </button>
            ),
          )}
          <button className="page-btn" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)} aria-label="Next page">›</button>
        </nav>
        <button className="page-btn" style={{ padding: "0 14px" }} onClick={download}>Download JSON</button>
      </div>
    </div>
  );
}
