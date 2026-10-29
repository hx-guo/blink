"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EventMap, { type ExposureGrid, type MapEvent } from "@/components/EventMap";
import EventTable, { type Ev } from "@/components/EventTable";
import RangeBrush from "@/components/RangeBrush";
import { Loading } from "@/components/Shell";
import { BASE, displayName, eventHref, lat as fmtLat, lon as fmtLon, sci, years } from "@/lib/format";

const fmtLat1 = (x: number) => `${Math.abs(x).toFixed(1)}° ${x < 0 ? "S" : "N"}`;
const fmtLon1 = (x: number) => `${Math.abs(x).toFixed(1)}° ${x < 0 ? "W" : "E"}`;
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
  const instruments = index.instruments.map((i) => i.instrument).join(", ");
  return (
    <div className="flex flex-col gap-16">
      <section className="grid gap-12 md:grid-cols-[minmax(0,1fr)_300px]">
        <div>
          <h1 style={{ fontSize: "clamp(34px, 5vw, 54px)", lineHeight: 1.06, fontWeight: 400, letterSpacing: "-0.015em" }}>
            Terrestrial gamma-ray flashes, catalogued as the data arrive
          </h1>
          <p className="mt-6 secondary" style={{ maxWidth: 540, fontSize: 19, lineHeight: 1.55 }}>
            Sub-millisecond bursts of gamma rays from thunderstorms, found in the data of space-borne gamma-ray
            detectors about an hour after the data arrive and selected by the data of the instrument alone. Every
            confirmed flash has a permanent name, its light curves and the exposure of the search.{" "}
            <Link href="/about/">How the catalog is made.</Link>
          </p>
          <dl className="mt-10 grid grid-cols-3 gap-6 pt-3 rule" style={{ maxWidth: 540 }}>
            <div>
              <dt className="label">TGFs</dt>
              <dd className="tabular mt-1" style={{ fontSize: 34, lineHeight: 1.1 }}>{nFinal.toLocaleString("en")}</dd>
            </div>
            <div>
              <dt className="label">Searched data</dt>
              <dd className="tabular mt-1" style={{ fontSize: 34, lineHeight: 1.1 }}>{years(live).toFixed(1)} <span style={{ fontSize: 18 }}>yr</span></dd>
            </div>
            <div>
              <dt className="label">Since</dt>
              <dd className="tabular mt-1" style={{ fontSize: 34, lineHeight: 1.1 }}>{first.slice(0, 4)}</dd>
            </div>
          </dl>
          <p className="caption mt-3">{instruments}; data through {last}.</p>
        </div>
        <aside>
          <h2 className="label pb-2" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>Latest</h2>
          <ol>
            {(latest ?? []).slice(0, 6).map((e) => (
              <li key={e.cand_id} className="py-2.5" style={{ borderBottom: "0.5px solid var(--border)" }}>
                <Link href={eventHref({ name: e.name, cand_id: e.cand_id })} style={{ textDecoration: "none", fontSize: 17 }}>
                  {e.name ? displayName(e.name) : "Provisional"}
                </Link>
                <div className="caption tabular" style={{ fontSize: 13.5 }}>
                  {e.peak.slice(0, 10)} {e.peak.slice(11, 16)} UTC · {fmtLat1(e.latitude)}, {fmtLon1(e.longitude)}
                </div>
              </li>
            ))}
          </ol>
        </aside>
      </section>

      <figure className="flex flex-col gap-3">
        {many && (
          <div className="flex gap-5 text-sm">
            {["all", ...index.instruments.map((i) => i.instrument)].map((d) => (
              <button key={d} className="opt" aria-pressed={detector === d} onClick={() => setDetector(d)}>
                {d === "all" ? "All detectors" : d}
              </button>
            ))}
          </div>
        )}
        {catalogs ? <EventMap events={mapEvents} highlight={highlight} exposures={exposures} /> : <Loading error={null} what="the catalog" />}
        <figcaption className="caption" style={{ maxWidth: 760 }}>
          <strong>Figure 1.</strong> Sub-satellite points of the {events.length.toLocaleString("en")} TGFs. The tinted
          area is where the search had live time: it spans the latitudes of the orbit and leaves out the South Atlantic
          Anomaly, where the detectors are off. Select a range in Figure 2 to pick out events.
        </figcaption>
      </figure>

      {catalogs && (
        <figure className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between gap-4 flex-wrap text-[15px]">
            <div className="flex gap-5">
              {AXES.map((a) => (
                <button key={a.key} className="opt" aria-pressed={axis === a.key} onClick={() => setAxis(a.key)}>{a.title}</button>
              ))}
            </div>
            {axis === "startMs" && (
              <div className="flex items-baseline gap-2">
                <span className="caption">From</span>
                <input
                  type="date"
                  className="field"
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
                <span className="caption">to</span>
                <input
                  type="date"
                  className="field"
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
          <figcaption className="caption">
            <strong>Figure 2.</strong>{" "}
            {axis === "startMs" ? "TGFs per month." : axis === "duration" ? "TGFs by the width of their time window." : "TGFs by their nominal false-alarm rate."}{" "}
            {sel ? (
              <>
                Selected: <span style={{ color: "var(--accent)" }}>{picked.length.toLocaleString("en")}</span> of{" "}
                {events.length.toLocaleString("en")}, {rangeText(axis, sel)}.{" "}
                <button className="link-btn" onClick={() => setSel(null)}>Clear</button>
              </>
            ) : (
              <>Drag across the bars to select a range{axis === "startMs" ? " or enter the dates" : ""}; click a bar to select it.</>
            )}
          </figcaption>
        </figure>
      )}

      <section className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <h2 style={{ fontSize: 26, fontWeight: 400 }}>{sel ? "Selected events" : "Events"}</h2>
          <span className="caption">Table 1</span>
        </div>
        <EventTable events={picked} showDetector={many} />
      </section>
    </div>
  );
}
