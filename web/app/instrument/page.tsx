"use client";

import Bars, { type Bin } from "@/components/Bars";
import RateMap, { type RateCell } from "@/components/RateMap";
import { Loading } from "@/components/Shell";
import { instrumentColor, years } from "@/lib/format";
import type { Index } from "@/lib/types";
import { useJson, useParam } from "@/lib/useData";
import { useCatalogs } from "@/lib/useCatalogs";

type Exposure = {
  live_seconds: number;
  demo: boolean;
  cell_deg: number;
  map: [number, number, number][];
  latitude: number[];
  lst: number[];
};

/** 事件按曝光的同一套格子计数，得每格率 */
function rateCells(exposure: Exposure, lons: number[], lats: number[]): RateCell[] {
  const d = exposure.cell_deg;
  const key = (lat: number, lon: number) => `${Math.floor((lat + 90) / d)},${Math.floor((((lon % 360) + 360) % 360) / d)}`;
  const counts = new Map<string, number>();
  lons.forEach((lon, i) => counts.set(key(lats[i], lon), (counts.get(key(lats[i], lon)) ?? 0) + 1));
  return exposure.map.map(([lat0, lon0, s]) => ({
    lat0,
    lon0,
    size: d,
    hours: s / 3600,
    events: counts.get(`${Math.floor((lat0 + 90) / d)},${Math.floor(lon0 / d)}`) ?? 0,
  }));
}

/** 边缘分布的率：每箱事件数 / 曝光小时 */
function rateBins(events: number[], seconds: number[], label: (i: number) => string, tick: (i: number) => string | undefined): Bin[] {
  return seconds.map((s, i) => ({
    label: `${label(i)} · ${events[i]} TGFs in ${(s / 3600).toFixed(0)} h`,
    value: s > 3600 * 24 ? Number(((events[i] / s) * 3600).toFixed(3)) : 0,
    tick: tick(i),
  }));
}

type Manifest = {
  generated_at: string;
  selection: Record<string, number | string | boolean>;
  counts: Record<string, number>;
};

/** 直方图：对数等距分箱 */
function logBins(values: number[], lo: number, hi: number, perDecade: number, fmt: (x: number) => string): Bin[] {
  const n = Math.round(Math.log10(hi / lo) * perDecade);
  const edges = Array.from({ length: n + 1 }, (_, i) => lo * 10 ** (i / perDecade));
  const counts = new Array(n).fill(0);
  for (const v of values) {
    const k = Math.floor(Math.log10(v / lo) * perDecade);
    if (k >= 0 && k < n) counts[k] += 1;
  }
  return counts.map((c, i) => ({
    label: `${fmt(edges[i])} – ${fmt(edges[i + 1])}`,
    value: c,
    tick: Number.isInteger(Math.round(Math.log10(edges[i]) * 1e6) / 1e6) ? fmt(edges[i]) : undefined,
  }));
}

export default function InstrumentPage() {
  const slug = useParam("slug", "instrument");
  const { data: index, error } = useJson<Index>("index.json");
  const { data: manifest } = useJson<Manifest>(slug ? `${slug}/manifest.json` : null);
  const catalogs = useCatalogs(slug ? [slug] : []);
  const { data: exposure } = useJson<Exposure>(slug ? `${slug}/exposure.json` : null);
  const inst = index?.instruments.find((i) => i.slug === slug);
  if (!index) return <Loading error={error} what="the catalog index" />;
  if (!inst) return <p className="secondary">Unknown instrument.</p>;
  const cat = catalogs?.[0];
  const c = manifest?.counts;
  const funnel: [string, number | undefined][] = [
    ["Candidates of the search (fa < 20 per year)", c?.pool],
    ["fa ≤ 10⁻⁵ per year", c?.significant],
    ["Not in a train of candidates", c && c.significant - c.train_removed],
    ["Not a gamma-ray burst", c?.catalog],
    ["Confirmed (final)", c?.final],
  ];
  const lats = (cat?.columns.latitude as number[]) ?? [];
  const latBins: Bin[] = Array.from({ length: 18 }, (_, i) => {
    const lo = -45 + i * 5;
    return { label: `${lo}° to ${lo + 5}°`, value: lats.filter((l) => l >= lo && l < lo + 5).length, tick: lo % 15 === 0 ? `${lo}°` : undefined };
  });
  const yr = years(exposure?.live_seconds ?? inst.searched_seconds);
  // 率只用已确认事件
  const finalIdx = cat ? (cat.columns.status as string[]).map((s, i) => (s === "final" ? i : -1)).filter((i) => i >= 0) : [];
  const fLat = finalIdx.map((i) => cat!.columns.latitude[i] as number);
  const fLon = finalIdx.map((i) => cat!.columns.longitude[i] as number);
  const latEvents = new Array(180).fill(0);
  fLat.forEach((l) => (latEvents[Math.min(179, Math.floor(l + 90))] += 1));
  // 纬度并成 5° 箱再算率
  const lat5 = (arr: number[]) => Array.from({ length: 36 }, (_, k) => arr.slice(k * 5, k * 5 + 5).reduce((a, b) => a + b, 0));
  const lstEvents = new Array(48).fill(0);
  if (cat) {
    finalIdx.forEach((i) => {
      // 地方平太阳时：UTC + 经度/15（视太阳时与它差一个时差方程，±16 min，这里只到半小时一格）
      const p = cat.columns.peak[i] as string;
      const h = (Number(p.slice(11, 13)) + Number(p.slice(14, 16)) / 60 + (cat.columns.longitude[i] as number) / 15 + 48) % 24;
      lstEvents[Math.min(47, Math.floor(h * 2))] += 1;
    });
  }
  const sel = manifest?.selection;

  return (
    <div className="flex flex-col gap-6">
      <title>{`${inst.instrument} · TGF Catalog`}</title>
      <div className="flex items-center gap-2">
        <span style={{ width: 12, height: 12, borderRadius: 999, background: instrumentColor(inst.instrument) }} />
        <h1 className="text-2xl font-semibold">{inst.instrument}</h1>
        {!inst.public && <span className="badge">not validated</span>}
      </div>
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
        <div className="card p-4">
          <h2 className="font-semibold mb-2">Selection</h2>
          <table className="data wrap tabular">
            <tbody>
              {funnel.map(([k, v]) => (
                <tr key={k}><td className="secondary">{k}</td><td style={{ textAlign: "right" }}>{v == null ? "…" : v.toLocaleString("en")}</td></tr>
              ))}
            </tbody>
          </table>
          {sel && (
            <p className="text-xs muted mt-2">
              Train threshold {String(sel.train_threshold)} candidates within ±{String(sel.train_half_window_s)} s; GRB rule{" "}
              {String(sel.grb_rule ?? "Z > 5 outside the candidate, shared by the boxes")}; electron beams flagged at P &lt;{" "}
              {String(sel.electron_beam_p)}.
            </p>
          )}
        </div>
        <div className="card p-4">
          <h2 className="font-semibold mb-2">Data</h2>
          <table className="data wrap tabular">
            <tbody>
              <tr><td className="secondary">Data range</td><td>{inst.first_day} – {inst.last_day}</td></tr>
              <tr><td className="secondary">Searched live time</td><td>{yr.toFixed(2)} yr{exposure?.demo ? " (modeled)" : ""}</td></tr>
              <tr><td className="secondary">Confirmed TGFs per year searched</td><td>{(inst.n_final / yr).toFixed(0)}</td></tr>
              <tr><td className="secondary">Electron beams</td><td>{inst.electron_beams}</td></tr>
              <tr><td className="secondary">Catalog generated</td><td>{inst.generated_at.slice(0, 19).replace("T", " ")} UTC</td></tr>
              <tr><td className="secondary">Pipeline version</td><td><code>{inst.blink_git}</code></td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div className="card p-4">
        <h2 className="font-semibold">Rate per unit exposure</h2>
        <p className="text-xs muted mb-2">
          Confirmed TGFs at their sub-satellite points divided by the live time of the search spent over each{" "}
          {exposure?.cell_deg ?? 5}° cell.
          {exposure?.demo && " Demo: the exposure is modeled for a circular orbit at 43° inclination, uniform in longitude (no SAA gap)."}
        </p>
        {exposure && cat ? <RateMap cells={rateCells(exposure, fLon, fLat)} minHours={24} /> : <Loading error={null} what="the exposure" />}
      </div>
      {exposure && cat && (
        <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }}>
          <div className="card p-4">
            <h2 className="font-semibold mb-2">TGFs per hour of exposure, by latitude</h2>
            <Bars
              bins={rateBins(lat5(latEvents), lat5(exposure.latitude), (k) => `${k * 5 - 90}° to ${k * 5 - 85}°`, (k) => ((k * 5 - 90) % 30 === 0 ? `${k * 5 - 90}°` : undefined)).slice(9, 27)}
              unit=" per hour"
            />
          </div>
          <div className="card p-4">
            <h2 className="font-semibold mb-2">TGFs per hour of exposure, by local solar time</h2>
            <Bars
              bins={rateBins(lstEvents, exposure.lst, (k) => `${(k / 2).toFixed(1)} to ${(k / 2 + 0.5).toFixed(1)} h`, (k) => (k % 12 === 0 ? `${k / 2} h` : undefined))}
              unit=" per hour"
            />
          </div>
        </div>
      )}
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }}>
        <div className="card p-4">
          <h2 className="font-semibold mb-2">Width of the candidate window (µs)</h2>
          {cat ? (
            <Bars bins={logBins((cat.columns.duration_us as number[]).filter((d) => d > 0), 10, 10000, 8, (x) => (x >= 1000 ? `${x / 1000} ms` : `${Math.round(x)}`))} unit=" events" />
          ) : <Loading error={null} what="the catalog" />}
        </div>
        <div className="card p-4">
          <h2 className="font-semibold mb-2">Latitude of the sub-satellite point</h2>
          {cat ? <Bars bins={latBins} unit=" events" /> : <Loading error={null} what="the catalog" />}
        </div>
      </div>
    </div>
  );
}
