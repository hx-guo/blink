"use client";

import Link from "next/link";
import { useMemo, useState, useEffect } from "react";
import EventMap, { MAP_ASPECT_H, type MapEvent } from "@/components/EventMap";
import EventTable, { type Ev } from "@/components/EventTable";
import Ruler, { type RulerBin } from "@/components/Ruler";
import { ago, Loading } from "@/components/Shell";
import { displayName, eventHref, lat as fmtLat, lon as fmtLon, sci, years } from "@/lib/format";
import type { Index, LatestEvent } from "@/lib/types";
import { useJson } from "@/lib/useData";
import { useCatalogs } from "@/lib/useCatalogs";
import { site } from "@/site.config";

type Axis = "startMs" | "duration" | "fa";
const AXES: { key: Axis; title: string }[] = [
  { key: "startMs", title: "Time" },
  { key: "duration", title: "Duration" },
  { key: "fa", title: "False-alarm rate" },
];
const N_BINS = 120;


type Exposure = { cell_deg: number; map: [number, number, number][] };

/** 曝光近零的格（星载仪器在 SAA 关机）：倾角带内、低于同纬度中位数 10% 的格 */
function saaCells(exposure: Exposure | null, inclination: number | null): [number, number, number][] {
  if (!exposure || inclination == null) return [];
  const d = exposure.cell_deg;
  const seconds = new Map(exposure.map.map(([lat, lon, s]) => [`${lat},${lon}`, s]));
  const out: [number, number, number][] = [];
  for (let lat = -90; lat < 90; lat += d) {
    if (lat < -inclination + d || lat + d > inclination - d) continue;
    const row = Array.from({ length: 360 / d }, (_, j) => seconds.get(`${lat},${j * d}`) ?? 0);
    const median = [...row].sort((a, b) => a - b)[row.length >> 1];
    row.forEach((s, j) => s < 0.1 * median && out.push([lat, j * d, d]));
  }
  return out;
}

function label(axis: Axis, v: number): string {
  if (axis === "startMs") return new Date(v).toISOString().slice(0, 10);
  if (axis === "duration") return `${Math.round(v)}µs`;
  return sci(v);
}

export default function Home() {
  const { data: index, error } = useJson<Index>("index.json");
  const { data: latest } = useJson<LatestEvent[]>("latest.json");
  const catalogs = useCatalogs(index?.instruments.map((i) => i.slug) ?? []);
  const firstSlug = index?.instruments[0]?.slug ?? null;
  const { data: exposure } = useJson<Exposure>(firstSlug ? `${firstSlug}/exposure.json` : null);
  const [axis, setAxis] = useState<Axis>("startMs");
  const [detector, setDetector] = useState<string>("all");
  const [hoverBin, setHoverBin] = useState<number | null>(null);
  const [lockBin, setLockBin] = useState<number | null>(null);
  useEffect(() => setLockBin(null), [axis, detector]);

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

  // 刻度尺：按当前量排序后等分成箱
  const { bins, members } = useMemo(() => {
    const sorted = [...events].sort((a, b) => a[axis] - b[axis]);
    const n = Math.min(N_BINS, sorted.length);
    const members: Ev[][] = Array.from({ length: n }, (_, k) =>
      sorted.slice(Math.floor((k * sorted.length) / n), Math.floor(((k + 1) * sorted.length) / n)),
    );
    const bins: RulerBin[] = members.map((m) => ({ label: label(axis, m[0][axis]), count: m.length }));
    return { bins, members };
  }, [events, axis]);
  const activeBin = hoverBin ?? lockBin;
  const highlight = activeBin != null && members[activeBin] ? new Set(members[activeBin].map((e) => e.key)) : null;
  const tableEvents = lockBin != null && members[lockBin] ? members[lockBin] : events;

  if (!index) return <Loading error={error} what="the catalog index" />;
  const nFinal = index.instruments.reduce((a, i) => a + i.n_final, 0);
  const live = index.instruments.reduce((a, i) => a + i.searched_seconds, 0);
  const first = index.instruments.map((i) => i.first_day).sort()[0];
  const last = index.instruments.map((i) => i.last_day).sort().slice(-1)[0];
  const inst = detector === "all" ? index.instruments[0]?.instrument : detector;
  const inclination = inst ? site.inclination[inst] ?? null : null;

  const mapEvents: MapEvent[] = events.map((e) => ({
    key: e.key,
    lon: e.lon,
    lat: e.lat,
    href: eventHref({ name: e.name, cand_id: e.cand }),
    provisional: e.provisional,
    rows: [
      ["ID", e.name ? displayName(e.name) : "pending (provisional)"],
      ["Detector", e.instrument.replace("Insight-", "")],
      ["Start Time", `${e.start.slice(0, 10)} ${e.start.slice(11, 29).replace("Z", "")}`],
      ["Duration", `${Math.round(e.duration)}µs`],
      ["False-alarm rate", `${sci(e.fa)} yr⁻¹`],
      ["Altitude", `${e.alt.toFixed(2)}km`],
      ["Position", `${Math.abs(e.lon).toFixed(2)}°${e.lon < 0 ? "W" : "E"} ${Math.abs(e.lat).toFixed(2)}°${e.lat < 0 ? "S" : "N"}`],
    ],
  }));

  const many = index.instruments.length > 1;
  return (
    <div className="flex flex-col gap-10">
      <section className="grid gap-8 md:grid-cols-[1fr_340px] items-start pt-4">
        <div className="flex flex-col gap-5">
          <div className="eyebrow">Updated in near real time</div>
          <h1 className="font-semibold tracking-tight" style={{ fontSize: "clamp(28px, 4.2vw, 44px)", lineHeight: 1.1 }}>
            Terrestrial gamma-ray flashes, catalogued as the data arrive
          </h1>
          <p className="secondary leading-relaxed" style={{ maxWidth: 560 }}>
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
        <aside className="card p-4 flex flex-col gap-1">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-semibold">Latest TGFs</h2>
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
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-lg font-semibold">Where they were seen</h2>
            <p className="text-sm secondary">Sub-satellite point of each TGF. Use the scale to pick a range.</p>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs muted">Scale</span>
            <div className="seg" role="group" aria-label="Scale">
              {AXES.map((a) => (
                <button key={a.key} aria-pressed={axis === a.key} onClick={() => setAxis(a.key)}>{a.title}</button>
              ))}
            </div>
            {many && (
              <select className="control" value={detector} onChange={(e) => setDetector(e.target.value)} aria-label="Detector">
                {["all", ...index.instruments.map((i) => i.instrument)].map((d) => (
                  <option key={d} value={d}>{d === "all" ? "All detectors" : d}</option>
                ))}
              </select>
            )}
          </div>
        </div>
        {catalogs ? (
          <div className="card p-3 flex items-center gap-2 overflow-hidden">
            <div className="hidden md:block">
              <Ruler
                title={AXES.find((a) => a.key === axis)!.title}
                bins={bins}
                height={Math.round(MAP_ASPECT_H * 0.95)}
                active={activeBin}
                locked={lockBin}
                onHover={setHoverBin}
                onToggle={(k) => setLockBin((b) => (b === k ? null : k))}
              />
            </div>
            <EventMap events={mapEvents} highlight={highlight} inclination={inclination} saaCells={saaCells(exposure, inclination)} />
          </div>
        ) : (
          <Loading error={null} what="the catalog" />
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold">All events</h2>
        <EventTable
          events={tableEvents}
          showDetector={many}
          caption={
            lockBin != null && members[lockBin] ? (
              <div className="text-sm flex items-center gap-2">
                <span className="badge" style={{ borderColor: "var(--accent)", color: "var(--text-primary)" }}>
                  {members[lockBin].length} events, {AXES.find((a) => a.key === axis)!.title.toLowerCase()}{" "}
                  {label(axis, members[lockBin][0][axis])} – {label(axis, members[lockBin][members[lockBin].length - 1][axis])}
                </span>
                <button className="muted" onClick={() => setLockBin(null)}>Show all</button>
              </div>
            ) : (
              <div className="text-sm secondary">{events.length.toLocaleString("en")} events</div>
            )
          }
        />
      </section>
    </div>
  );
}
