"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EventMap, { type ExposureGrid, type MapEvent } from "@/components/EventMap";
import EventTable, { type Ev } from "@/components/EventTable";
import RangeBrush from "@/components/RangeBrush";
import { ago, Loading } from "@/components/Shell";
import { BASE, displayName, eventHref, lat as fmtLat, lon as fmtLon, sci, years } from "@/lib/format";
import type { Index, LatestEvent } from "@/lib/types";
import { useJson } from "@/lib/useData";
import { useCatalogs } from "@/lib/useCatalogs";

type Axis = "startMs" | "duration" | "fa";
const AXES: { key: Axis; title: string }[] = [
  { key: "startMs", title: "Time" },
  { key: "duration", title: "Duration" },
  { key: "fa", title: "False-alarm rate" },
];
const DAY = 86400000;
const iso = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/** 刻度：时间写年份，时长写 µs/ms，虚警率写 10 的幂 */
function tick(axis: Axis, v: number): string {
  if (axis === "startMs") return String(new Date(v).getUTCFullYear());
  if (axis === "duration") return v >= 1000 ? `${+(v / 1000).toPrecision(2)} ms` : `${+v.toPrecision(2)} µs`;
  return sci(v, 0).replace(/^1×/, "");
}

function rangeText(axis: Axis, [a, b]: [number, number]): string {
  if (axis === "startMs") return a + DAY >= b ? iso(a) : `${iso(a)} – ${iso(b - DAY)}`;
  if (axis === "duration") return `${Math.round(a)} – ${Math.round(b)} µs`;
  return `${sci(a)} – ${sci(b)} yr⁻¹`;
}

/** 各仪器的曝光（运行时取） */
function useExposures(slugs: string[]): ExposureGrid[] {
  const [grids, setGrids] = useState<ExposureGrid[]>([]);
  const key = slugs.join(",");
  useEffect(() => {
    let alive = true;
    Promise.all(
      key.split(",").filter(Boolean).map((s) => fetch(`${BASE}/data/${s}/exposure.json`).then((r) => r.json() as Promise<ExposureGrid>)),
    ).then((g) => alive && setGrids(g));
    return () => {
      alive = false;
    };
  }, [key]);
  return grids;
}

export default function Home() {
  const { data: index, error } = useJson<Index>("index.json");
  const { data: latest } = useJson<LatestEvent[]>("latest.json");
  const catalogs = useCatalogs(index?.instruments.map((i) => i.slug) ?? []);
  const [axis, setAxis] = useState<Axis>("startMs");
  const [detector, setDetector] = useState<string>("all");
  const [sel, setSel] = useState<[number, number] | null>(null);
  useEffect(() => setSel(null), [axis, detector]);
  const shownSlugs = (index?.instruments ?? []).filter((i) => detector === "all" || i.instrument === detector).map((i) => i.slug);
  const exposures = useExposures(shownSlugs);

  const all: Ev[] = useMemo(
    () =>
      (catalogs ?? []).flatMap((cat) => {
        const c = cat.columns;
        return Array.from({ length: cat.n }, (_, i) => ({
          key: c.cand_id[i] as string,
          name: c.name[i] as string | null,
          cand: c.cand_id[i] as string,
          instrument: cat.instrument,
          provisional: c.status[i] !== "final",
          start: c.start[i] as string,
          startMs: Date.parse(c.start[i] as string),
          duration: c.duration_us[i] as number,
          fa: c.false_positive_per_year[i] as number,
          lon: c.longitude[i] as number,
          lat: c.latitude[i] as number,
          alt: c.altitude_km[i] as number,
          solar: (c.solar_time?.[i] as string) ?? "",
        }));
      }),
    [catalogs],
  );
  const events = useMemo(() => all.filter((e) => detector === "all" || e.instrument === detector), [all, detector]);
  const values = useMemo(() => events.map((e) => e[axis]), [events, axis]);
  const picked = useMemo(() => (sel ? events.filter((e) => e[axis] >= sel[0] && e[axis] < sel[1]) : events), [events, sel, axis]);
  const highlight = sel ? new Set(picked.map((e) => e.key)) : null;

  if (!index) return <Loading error={error} what="the catalog index" />;
  const nFinal = index.instruments.reduce((a, i) => a + i.n_final, 0);
  const live = index.instruments.reduce((a, i) => a + i.searched_seconds, 0);
  const first = index.instruments.map((i) => i.first_day).sort()[0];
  const last = index.instruments.map((i) => i.last_day).sort().slice(-1)[0];

  const mapEvents: MapEvent[] = events.map((e) => ({
    key: e.key,
    lon: e.lon,
    lat: e.lat,
    href: eventHref({ name: e.name, cand_id: e.cand }),
    provisional: e.provisional,
    rows: [
      ["", e.name ? displayName(e.name) : "Provisional"],
      ["Start", `${e.start.slice(0, 10)} ${e.start.slice(11, 23)} UTC`],
      ["Duration", `${Math.round(e.duration)} µs`],
      ["False-alarm rate", `${sci(e.fa)} yr⁻¹`],
      ["Position", `${fmtLat(e.lat)}, ${fmtLon(e.lon)}`],
      ...(index.instruments.length > 1 ? [["Detector", e.instrument] as [string, string]] : []),
    ],
  }));
  const many = index.instruments.length > 1;
  return (
    <div className="flex flex-col gap-10">
      <section className="grid gap-10 md:grid-cols-[1fr_330px] items-start pt-2">
        <div className="flex flex-col gap-5">
          <div className="eyebrow">Updated in near real time</div>
          <h1 className="font-semibold" style={{ fontSize: "clamp(30px, 4.4vw, 48px)", lineHeight: 1.08, letterSpacing: "-0.01em" }}>
            Terrestrial gamma-ray flashes, catalogued as the data arrive
          </h1>
          <p className="prose secondary" style={{ maxWidth: 560, fontSize: 18, lineHeight: 1.55 }}>
            Sub-millisecond bursts of gamma rays from thunderstorms, found in the data of space-borne gamma-ray
            detectors about an hour after the data arrive and selected by the data of the instrument alone. Every
            confirmed flash has a permanent name, its light curves and the exposure of the search.{" "}
            <Link href="/about/">How the catalog is made</Link>
          </p>
          <dl className="flex flex-wrap gap-x-10 gap-y-4 pt-2">
            <div>
              <dd className="font-semibold" style={{ fontSize: 48, lineHeight: 1 }}>{nFinal.toLocaleString("en")}</dd>
              <dt className="text-sm secondary mt-1">TGFs</dt>
            </div>
            <div className="self-end">
              <dd className="text-2xl font-semibold">{years(live).toFixed(1)} yr</dd>
              <dt className="text-sm secondary mt-1">of searched data</dt>
            </div>
            <div className="self-end">
              <dd className="text-2xl font-semibold">{first.slice(0, 4)}–{last.slice(0, 4)}</dd>
              <dt className="text-sm secondary mt-1">{index.instruments.map((i) => i.instrument).join(", ")}</dt>
            </div>
          </dl>
        </div>
        <aside className="flex flex-col gap-1 md:pl-8 md:border-l" style={{ borderColor: "var(--border-strong)" }}>
          <div className="flex items-baseline justify-between">
            <h2 className="text-lg font-semibold">Latest TGFs</h2>
            <span className="text-xs muted">data through {last}</span>
          </div>
          <ul className="flex flex-col">
            {(latest ?? []).slice(0, 6).map((e) => (
              <li key={e.cand_id} style={{ borderTop: "1px solid var(--grid)" }}>
                <Link href={eventHref({ name: e.name, cand_id: e.cand_id })} className="latest-row">
                  <span className="dot-mark" style={{ background: "var(--accent)" }} aria-hidden />
                  <span className="flex-1 min-w-0">
                    <span className="block font-semibold text-sm" style={{ color: "var(--text-primary)" }}>
                      {e.name ? displayName(e.name) : "Provisional"}
                    </span>
                    <span className="block text-xs secondary tabular">{e.peak.slice(0, 10)} {e.peak.slice(11, 19)} UTC</span>
                    <span className="block text-xs muted tabular">{fmtLat(e.latitude)}, {fmtLon(e.longitude)}</span>
                  </span>
                  <span className="text-xs muted shrink-0 self-start pt-0.5">{ago(e.peak)}</span>
                </Link>
              </li>
            ))}
          </ul>
        </aside>
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex items-end justify-between gap-4 flex-wrap rule pt-5">
          <div>
            <h2 className="text-2xl font-semibold">Where they were seen</h2>
            <p className="text-sm secondary mt-1">
              Sub-satellite point of each TGF. The shading is the live time of the search, so the latitude range of
              each orbit and the South Atlantic Anomaly, where the detectors are off, show as they are.
            </p>
          </div>
          {many && (
            <select className="control" value={detector} onChange={(e) => setDetector(e.target.value)} aria-label="Detector">
              {["all", ...index.instruments.map((i) => i.instrument)].map((d) => (
                <option key={d} value={d}>{d === "all" ? "All detectors" : d}</option>
              ))}
            </select>
          )}
        </div>
        {catalogs ? (
          <>
            <EventMap events={mapEvents} highlight={highlight} exposures={exposures} />
            <div className="flex items-center gap-5 flex-wrap text-xs secondary">
              <span className="flex items-center gap-2">
                <span className="dot-mark" style={{ background: "var(--accent)" }} /> TGF
              </span>
              <span className="flex items-center gap-2">
                <span style={{ width: 48, height: 8, background: "linear-gradient(90deg, transparent, var(--exposure))", opacity: 0.55, border: "1px solid var(--border)" }} />
                live time of the search, none to most
              </span>
            </div>
            <div className="flex flex-col gap-2 pt-3">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="seg" role="group" aria-label="Select by">
                  {AXES.map((a) => (
                    <button key={a.key} aria-pressed={axis === a.key} onClick={() => setAxis(a.key)}>{a.title}</button>
                  ))}
                </div>
                {axis === "startMs" && (
                  <div className="flex items-center gap-2 text-sm">
                    <input
                      type="date"
                      className="control tabular"
                      aria-label="From"
                      min={first}
                      max={last}
                      value={sel ? iso(sel[0]) : ""}
                      onChange={(e) => {
                        if (!e.target.value) return setSel(null);
                        const a = Date.parse(e.target.value);
                        setSel([a, Math.max(sel ? sel[1] : a + DAY, a + DAY)]);
                      }}
                    />
                    <span className="muted">to</span>
                    <input
                      type="date"
                      className="control tabular"
                      aria-label="To"
                      min={first}
                      max={last}
                      value={sel ? iso(sel[1] - DAY) : ""}
                      onChange={(e) => {
                        if (!e.target.value) return setSel(null);
                        const b = Date.parse(e.target.value) + DAY;
                        setSel([Math.min(sel ? sel[0] : b - DAY, b - DAY), b]);
                      }}
                    />
                  </div>
                )}
              </div>
              <RangeBrush
                key={axis + detector}
                kind={axis === "startMs" ? "time" : "log"}
                values={values}
                selection={sel}
                onChange={setSel}
                tick={(v) => tick(axis, v)}
                bins={axis === "startMs" ? 112 : 80}
              />
              <div className="text-sm secondary">
                {sel ? (
                  <>
                    <strong style={{ color: "var(--text-primary)" }}>{picked.length.toLocaleString("en")}</strong> of{" "}
                    {events.length.toLocaleString("en")} events, {rangeText(axis, sel)}{" "}
                    <button className="link-btn" onClick={() => setSel(null)}>Clear</button>
                  </>
                ) : (
                  <>Drag across the histogram to select a range{axis === "startMs" ? ", or enter the dates" : ""}; click a bar to select it.</>
                )}
              </div>
            </div>
          </>
        ) : (
          <Loading error={null} what="the catalog" />
        )}
      </section>

      <section className="flex flex-col gap-3 rule pt-5">
        <h2 className="text-2xl font-semibold">{sel ? "Selected events" : "All events"}</h2>
        <EventTable events={picked} showDetector={many} />
      </section>
    </div>
  );
}
