"use client";

import { useMemo, useRef, useState, useEffect } from "react";
import EventMap, { MAP_ASPECT_H, type MapEvent } from "@/components/EventMap";
import EventTable, { type Ev } from "@/components/EventTable";
import Ruler, { type RulerBin } from "@/components/Ruler";
import { Loading } from "@/components/Shell";
import { displayName, eventHref, sci, years } from "@/lib/format";
import type { Index } from "@/lib/types";
import { useJson } from "@/lib/useData";
import { useCatalogs } from "@/lib/useCatalogs";
import { site } from "@/site.config";

type Axis = "startMs" | "duration" | "fa";
const AXES: { key: Axis; title: string }[] = [
  { key: "startMs", title: "Start Time" },
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
  const catalogs = useCatalogs(index?.instruments.map((i) => i.slug) ?? []);
  const firstSlug = index?.instruments[0]?.slug ?? null;
  const { data: exposure } = useJson<Exposure>(firstSlug ? `${firstSlug}/exposure.json` : null);
  const [axis, setAxis] = useState<Axis>("startMs");
  const [detector, setDetector] = useState<string>("all");
  const [menu, setMenu] = useState(false);
  const [hoverBin, setHoverBin] = useState<number | null>(null);
  const [lockBin, setLockBin] = useState<number | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => menuRef.current && !menuRef.current.contains(e.target as Node) && setMenu(false);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);
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

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-5">
        <h1 className="text-center font-bold" style={{ fontSize: "clamp(26px, 4vw, 40px)" }}>
          Catalog for Terrestrial Gamma-Ray Flashes
        </h1>
        <p className="font-medium leading-relaxed" style={{ textAlign: "justify" }}>
          Terrestrial gamma-ray flashes (TGFs) are bursts of gamma rays from thunderstorms that last less than a
          millisecond. This catalog lists the TGFs found by the Event-Based Transient Search (EBTS), which tests every
          time window that starts and ends on a detected event instead of binning the data, made fast by the
          Snapshot-Stepping Algorithm (SSA). The search runs on the data of each instrument as soon as they arrive. An
          event is selected by its significance alone, a nominal false-alarm rate below 10⁻⁵ per year, after trains of
          candidates from precipitating particles and gamma-ray bursts are removed; no lightning data are needed, so
          the catalog is updated in near real time and each confirmed event receives a permanent name TGF yymmddAA. It
          now holds {nFinal.toLocaleString("en")} TGFs found by {index.instruments.map((i) => i.instrument).join(", ")} in{" "}
          {years(live).toFixed(1)} years of searched data from {first} to {last}.
        </p>
      </section>

      <section className="flex flex-col gap-4">
        <div className="flex justify-center gap-4 flex-wrap">
          {AXES.map((a) => (
            <button key={a.key} className="chip" aria-pressed={axis === a.key} onClick={() => setAxis(a.key)}>
              <span className="dot" />
              {a.title}
            </button>
          ))}
          <div className="relative" ref={menuRef}>
            <button className="chip" aria-expanded={menu} onClick={() => setMenu((m) => !m)}>
              {detector === "all" ? "Detector" : detector.replace("Insight-", "")} <span aria-hidden>{menu ? "⌃" : "⌄"}</span>
            </button>
            {menu && (
              <div className="absolute z-20 mt-2 flex flex-col text-sm" style={{ background: "var(--surface)", border: "2px solid var(--border-strong)", borderRadius: 6, minWidth: 170 }}>
                {["all", ...index.instruments.map((i) => i.instrument)].map((d) => (
                  <button
                    key={d}
                    className="text-left px-3 py-2"
                    style={{ borderBottom: "1px dotted var(--border)", fontWeight: d === detector ? 700 : 500, color: d === detector ? "var(--accent)" : "inherit" }}
                    onClick={() => {
                      setDetector(d);
                      setMenu(false);
                    }}
                  >
                    {d === "all" ? "All detectors" : d.replace("Insight-", "")}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {catalogs ? (
          <div className="flex items-center gap-2 overflow-hidden">
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

      <section>
        <EventTable
          events={tableEvents}
          caption={
            lockBin != null && members[lockBin] ? (
              <div className="text-sm flex items-center gap-2">
                <span className="badge" style={{ borderColor: "var(--accent)", color: "var(--accent)" }}>
                  {members[lockBin].length} events, {AXES.find((a) => a.key === axis)!.title.toLowerCase()}{" "}
                  {label(axis, members[lockBin][0][axis])} – {label(axis, members[lockBin][members[lockBin].length - 1][axis])}
                </span>
                <button className="muted" onClick={() => setLockBin(null)}>Show all</button>
              </div>
            ) : (
              <div className="text-sm secondary">
                {events.length.toLocaleString("en")} events · click the scale on the left of the map to pick a range
              </div>
            )
          }
        />
      </section>
    </div>
  );
}
