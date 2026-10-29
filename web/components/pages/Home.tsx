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

/** 日期输入：纯文本 YYYY-MM-DD，回车或离开时生效；清空即取消选择 */
function DateField({ label, value, onCommit }: { label: string; value: string; onCommit: (ms: number | null) => void }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = () => {
    if (!text.trim()) return onCommit(null);
    const ms = Date.parse(`${text.trim()}T00:00:00Z`);
    if (/^\d{4}-\d{2}-\d{2}$/.test(text.trim()) && Number.isFinite(ms)) onCommit(ms);
    else setText(value);
  };
  return (
    <input
      className="field tabular"
      aria-label={label}
      placeholder="YYYY-MM-DD"
      size={10}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
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
    <div className="flex flex-col" style={{ gap: 96 }}>
      <section className="margin-grid pt-16">
        <div />
        <div>
          <h1 style={{ fontSize: "clamp(32px, 4.2vw, 46px)", lineHeight: 1.12, fontWeight: 400, letterSpacing: "-0.01em", maxWidth: "15em" }}>
            Terrestrial gamma-ray flashes, catalogued as the data arrive
          </h1>
          <p className="mt-8 secondary prose" style={{ maxWidth: "31em", fontSize: 18, lineHeight: 1.6 }}>
            Bursts of gamma rays from thunderstorms, found in the data of space-borne detectors within about an hour
            of their arrival, each with a permanent name. <Link href="/about/">How the catalog is made</Link>
          </p>
          <dl className="mt-14 flex flex-wrap gap-x-16 gap-y-6">
            <div>
              <dd className="tabular" style={{ fontSize: 40, lineHeight: 1 }}>{nFinal.toLocaleString("en")}</dd>
              <dt className="caption mt-2">TGFs</dt>
            </div>
            <div>
              <dd className="tabular" style={{ fontSize: 40, lineHeight: 1 }}>{years(live).toFixed(1)}</dd>
              <dt className="caption mt-2">years of searched data</dt>
            </div>
            <div>
              <dd className="tabular" style={{ fontSize: 40, lineHeight: 1 }}>1 h</dd>
              <dt className="caption mt-2">between updates</dt>
            </div>
          </dl>
          <p className="caption mt-6">{instruments}, {first} to {last}.</p>
        </div>
      </section>

      <section className="margin-grid section">
        <header className="margin-note">
          <h2 className="section-title">Latest</h2>
          <p className="caption mt-2">The newest named TGFs.</p>
        </header>
        <ol className="grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
          {(latest ?? []).slice(0, 4).map((e) => (
            <li key={e.cand_id}>
              <Link href={eventHref({ name: e.name, cand_id: e.cand_id })} style={{ textDecoration: "none", fontSize: 16 }}>
                {e.name ? displayName(e.name) : "Provisional"}
              </Link>
              <div className="caption tabular mt-1">{e.peak.slice(0, 10)}, {e.peak.slice(11, 16)} UTC</div>
              <div className="caption tabular muted">{fmtLat1(e.latitude)}, {fmtLon1(e.longitude)}</div>
            </li>
          ))}
        </ol>
      </section>

      <figure className="margin-grid section">
        <figcaption className="margin-note caption">
          <h2 className="section-title mb-2">Where</h2>
          <span style={{ color: "var(--text-primary)" }}>Figure 1.</span> Where the {events.length.toLocaleString("en")} TGFs were seen, at the point below the satellite. The tint is
          the area searched: the latitudes of the orbit, without the South Atlantic Anomaly, where the detectors are off.
          {many && (
            <span className="flex flex-col gap-1 mt-4">
              {["all", ...index.instruments.map((i) => i.instrument)].map((d) => (
                <button key={d} className="opt self-start" aria-pressed={detector === d} onClick={() => setDetector(d)}>
                  {d === "all" ? "All detectors" : d}
                </button>
              ))}
            </span>
          )}
        </figcaption>
        <div>{catalogs ? <EventMap events={mapEvents} highlight={highlight} exposures={exposures} /> : <Loading error={null} what="the catalog" />}</div>
      </figure>

      {catalogs && (
        <figure className="margin-grid section">
          <figcaption className="margin-note caption">
            <h2 className="section-title mb-2">When</h2>
            <span style={{ color: "var(--text-primary)" }}>Figure 2.</span>{" "}
            {axis === "startMs" ? "TGFs per month." : axis === "duration" ? "TGFs by the width of their window." : "TGFs by their false-alarm rate."}{" "}
            Drag across it to select a range; the map and the table follow.
            <span className="flex flex-col gap-1 mt-4">
              {AXES.map((a) => (
                <button key={a.key} className="opt self-start" aria-pressed={axis === a.key} onClick={() => setAxis(a.key)}>{a.title}</button>
              ))}
            </span>
          </figcaption>
          <div className="flex flex-col gap-4">
            <RangeBrush
              key={axis + detector}
              kind={axis === "startMs" ? "time" : "log"}
              values={values}
              selection={sel}
              onChange={setSel}
              tick={(v) => tick(axis, v)}
              bins={axis === "startMs" ? 112 : 80}
            />
            <div className="flex items-baseline justify-between gap-6 flex-wrap caption" style={{ minHeight: 28 }}>
              <span>
                {sel ? (
                  <>
                    <span style={{ color: "var(--accent)" }}>{picked.length.toLocaleString("en")}</span> of{" "}
                    {events.length.toLocaleString("en")} selected, {rangeText(axis, sel)}.{" "}
                    <button className="link-btn" onClick={() => setSel(null)}>Clear</button>
                  </>
                ) : (
                  <span className="muted">Nothing selected.</span>
                )}
              </span>
              {axis === "startMs" && (
                <span className="flex items-baseline gap-2">
                  <DateField
                    label="From"
                    value={sel ? iso(sel[0]) : ""}
                    onCommit={(d) => (d == null ? setSel(null) : setSel([d, Math.max(sel ? sel[1] : d + DAY, d + DAY)]))}
                  />
                  <span className="muted">to</span>
                  <DateField
                    label="To"
                    value={sel ? iso(sel[1] - DAY) : ""}
                    onCommit={(d) => (d == null ? setSel(null) : setSel([Math.min(sel ? sel[0] : d, d), d + DAY]))}
                  />
                </span>
              )}
            </div>
          </div>
        </figure>
      )}

      <section className="margin-grid section pb-10">
        <div className="margin-note caption">
          <h2 className="section-title mb-2">Events</h2>
          <span style={{ color: "var(--text-primary)" }}>Table 1.</span> {sel ? "The selected TGFs." : "All TGFs, newest first."} Altitude, local time and light curves are on the
          page of each event.
        </div>
        <EventTable events={picked} showDetector={many} />
      </section>
    </div>
  );
}
