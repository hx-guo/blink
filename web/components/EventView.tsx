"use client";

import Link from "next/link";
import type { EventRecord } from "@/lib/types";
import { instrumentColor, instrumentHref, lat, lon, sci, utc } from "@/lib/format";
import { ContextLightCurve, EventLightCurve } from "./LightCurve";
import WorldMap from "./WorldMap";

const PENDING: Record<string, string> = {
  not_vetted: "the candidate has not been vetted yet",
  grb_background_gap: "the data around the candidate are incomplete (GRB test undetermined)",
  neighbors_unsearched: "the data within ±10 min have not all been searched (train test incomplete)",
};

function us(t: string, peak: string): number {
  // 两个 UTC 纳秒串之差（µs），避开 Date 的毫秒精度
  const ns = (s: string) => {
    const [d, f = ""] = s.replace("Z", "").split(".");
    return BigInt(Date.parse(`${d}Z`)) * 1_000_000n + BigInt(f.padEnd(9, "0").slice(0, 9));
  };
  return Number(ns(t) - ns(peak)) / 1000;
}

export default function EventView({ event }: { event: EventRecord }) {
  const lc = event.light_curve;
  const rows: [string, React.ReactNode][] = [
    ["Peak", `${utc(event.peak, 6)} UTC`],
    ["Duration", `${Math.round(event.duration_us)} µs (best interval)`],
    ["Window", `${utc(event.start, 6)} – ${utc(event.stop, 6).slice(11)}`],
    ["Counts", event.count],
    ["Nominal false-alarm rate", `${sci(event.false_positive_per_year, 2)} per year`],
    ["Sub-satellite point", `${lat(event.latitude)}, ${lon(event.longitude)}, ${event.altitude_km.toFixed(0)} km`],
    ["Candidates within ±10 min", event.neighbors_10min ?? "–"],
    ["GRB test", event.grb_z == null ? "undetermined" : `Z = ${event.grb_z.toFixed(1)}, largest box share ${event.grb_max_box_share?.toFixed(2)}`],
    ["ACD coincidences", event.n == null ? "–" : `${event.n_acd} of ${event.n} events (background ${event.n_acd_bg} of ${event.n_bg})`],
    ["Electron beam", event.electron_beam == null ? "–" : event.electron_beam ? `yes (P = ${sci(event.beam_p)})` : `no (P = ${sci(event.beam_p)})`],
    ["Largest detector share", event.det_share_max?.toFixed(2) ?? "–"],
    ["Candidate ID", <code key="id" className="text-xs">{event.cand_id}</code>],
  ];
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 text-sm">
        <span style={{ width: 10, height: 10, borderRadius: 999, background: instrumentColor(event.instrument) }} />
        <Link href={instrumentHref(event.slug)}>{event.instrument}</Link>
        <span className="badge">{event.status === "final" ? "confirmed" : "provisional"}</span>
        {event.electron_beam === 1 && <span className="badge">electron beam</span>}
      </div>
      {event.status !== "final" && event.pending && (
        <div className="card p-3 text-sm" style={{ background: "var(--banner)" }}>
          Provisional: {event.pending.split("|").map((p) => PENDING[p] ?? p).join("; ")}. It gets a name once
          the selection is complete.
        </div>
      )}
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
        <div className="card p-4">
          <table className="data wrap tabular">
            <tbody>
              {rows.map(([k, v]) => (
                <tr key={k}><td className="secondary">{k}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card p-2">
          <WorldMap
            height={470}
            points={[{ lon: event.longitude, lat: event.latitude, color: instrumentColor(event.instrument), href: "#", title: event.instrument, lines: [`${lat(event.latitude)}, ${lon(event.longitude)}`] }]}
          />
        </div>
      </div>
      {lc ? (
        <>
          <div className="card p-4">
            <h3 className="font-semibold mb-2">Light curve</h3>
            <EventLightCurve lc={lc} startUs={us(event.start, lc.peak)} stopUs={us(event.stop, lc.peak)} />
          </div>
          <div className="card p-4">
            <h3 className="font-semibold mb-2">Counts per 10 ms around the candidate</h3>
            <ContextLightCurve lc={lc} />
          </div>
        </>
      ) : (
        <div className="card p-4 muted text-sm">No light curve: the candidate has not been vetted.</div>
      )}
    </div>
  );
}
