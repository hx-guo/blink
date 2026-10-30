"use client";

import Link from "next/link";
import Bars, { type Bin } from "@/components/Bars";
import RateMap, { type RateCell } from "@/components/RateMap";
import { Loading } from "@/components/Shell";
import { sci, utc, years } from "@/lib/format";
import type { Index } from "@/lib/types";
import { useJson, useParam } from "@/lib/useData";
import { useCatalogs } from "@/lib/useCatalogs";
import Sup from "@/components/Sup";

type Exposure = {
  live_seconds: number;
  demo: boolean;
  cell_deg: number;
  map: [number, number, number][];
  /** 1° 纬度带的活时间（秒），自 −90° 起 */
  latitude: number[];
  /** 半小时一格的视太阳时活时间（秒），自 0 h 起 */
  lst: number[];
};

type Manifest = {
  generated_at: string;
  selection: Record<string, number | string | boolean>;
  counts: Record<string, number> & { pending?: Record<string, number> };
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

/** 率与泊松误差：每箱事件数 / 活时间（小时）；活时间不足一天的箱不给 */
function rateBins(events: number[], seconds: number[], label: (i: number) => string, tick: (i: number) => string | undefined): Bin[] {
  return seconds.map((s, i) => {
    const ok = s > 86400;
    const h = s / 3600;
    return {
      label: label(i),
      value: ok ? events[i] / h : 0,
      err: ok ? Math.sqrt(events[i]) / h : undefined,
      tick: tick(i),
      detail: `${events[i]} TGFs in ${h.toFixed(0)} h`,
    };
  });
}

/** 直方图：对数等距分箱，刻度在 10 的整数幂 */
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
    tick: i % perDecade === 0 ? fmt(edges[i]) : undefined,
  }));
}

const fmtUs = (x: number) => (x >= 1000 ? `${+(x / 1000).toPrecision(2)} ms` : `${+x.toPrecision(2)} µs`);
const rate = (v: number) => String(+v.toFixed(3));

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="tabular" style={{ fontSize: 40, lineHeight: 1.1 }}>{value}</div>
      <div className="caption mt-1">{label}</div>
    </div>
  );
}

function Chart({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <figure>
      <figcaption className="label pb-2 mb-3" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>{title}</figcaption>
      {children}
    </figure>
  );
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
  const sel = manifest?.selection;
  const live = exposure?.live_seconds ?? inst.searched_seconds;
  const yr = years(live);

  // 率只用已确认事件
  const fin = cat ? (cat.columns.status as string[]).flatMap((s, i) => (s === "final" ? [i] : [])) : [];
  const col = <T,>(k: string) => fin.map((i) => cat!.columns[k][i] as T);
  const fLat = col<number>("latitude");
  const fLon = col<number>("longitude");
  // 纬度：1° 带并成 5° 箱，只画 ±45°
  const latEvents = new Array(180).fill(0);
  fLat.forEach((l) => (latEvents[Math.min(179, Math.floor(l + 90))] += 1));
  const lat5 = (arr: number[]) => Array.from({ length: 36 }, (_, k) => arr.slice(k * 5, k * 5 + 5).reduce((a, b) => a + b, 0));
  // 视太阳时：目录的 solar_time 列，与曝光同一公式
  const lstEvents = new Array(48).fill(0);
  col<string>("solar_time").forEach((t) => {
    const h = Number(t.slice(0, 2)) + Number(t.slice(3, 5)) / 60;
    lstEvents[Math.min(47, Math.floor(h * 2))] += 1;
  });
  const latText = (d: number) => (d === 0 ? "0°" : `${Math.abs(d)}°${d < 0 ? "S" : "N"}`);

  const funnel: [string, number | undefined, number | undefined][] = c
    ? [
        ["Candidates of the search", undefined, c.pool],
        [`False-alarm rate above ${sci(Number(sel?.max_false_positive_per_year ?? 1e-5), 0).replace(/^1×/, "")} per year`, c.pool - c.significant, c.significant],
        ["In a train of candidates", c.train_removed, c.significant - c.train_removed],
        ["Gamma-ray bursts", c.grb_removed + (c.grb_same_burst_removed ?? 0), c.catalog],
        ["Provisional, waiting for data", c.catalog - c.final, c.final],
      ]
    : [];

  return (
    <div className="flex flex-col" style={{ gap: 88 }}>
      <title>{`${inst.instrument} · TGF Catalog`}</title>
      <section className="margin-grid pt-14">
        <div />
        <div>
          <h1 style={{ fontSize: "clamp(32px, 4.2vw, 46px)", lineHeight: 1.12, fontWeight: 400, letterSpacing: "-0.01em" }}>
            {inst.instrument}
            {!inst.public && <span className="badge ml-4 align-middle">not validated</span>}
          </h1>
          <p className="mt-6 secondary prose" style={{ maxWidth: "34em", fontSize: 18, lineHeight: 1.6 }}>
            Every hour of data from {inst.first_day} to {inst.last_day} searched, {inst.n_final.toLocaleString("en")} TGFs confirmed.
            {exposure?.demo && " The exposure is modeled for development."}
          </p>
          <div className="mt-10 grid gap-8" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", maxWidth: 820 }}>
            <Figure value={inst.n_final.toLocaleString("en")} label="confirmed TGFs" />
            <Figure value={yr.toFixed(2)} label="years of live time" />
            <Figure value={(inst.n_final / yr).toFixed(0)} label="TGFs per year of live time" />
            <Figure value={String(inst.electron_beams)} label="electron beams, flagged" />
          </div>
        </div>
      </section>

      <section className="margin-grid section">
        <header className="margin-note caption">
          <h2 className="section-title mb-3">Rate</h2>
          <p>Confirmed TGFs divided by the live time of the search at the same place or time: the rate does not depend on where the satellite spends its time.</p>
          <p className="mt-3">Blank cells had less than a day of live time: the South Atlantic Anomaly, where the detectors are off, and the edges of the orbit.</p>
        </header>
        <div className="flex flex-col gap-14">
          {exposure && cat ? <RateMap cells={rateCells(exposure, fLon, fLat)} minHours={24} /> : <Loading error={null} what="the exposure" />}
          {exposure && cat && (
            <div className="grid grid-cols-1 gap-12 lg:grid-cols-2">
              <Chart title="By latitude, TGFs per hour">
                <Bars
                  label="TGFs per hour of exposure by latitude"
                  bins={rateBins(lat5(latEvents), lat5(exposure.latitude), (k) => `${latText(k * 5 - 90)} to ${latText(k * 5 - 85)}`, (k) =>
                    (k * 5 - 90) % 15 === 0 ? latText(k * 5 - 90) : undefined,
                  ).slice(9, 27)}
                  format={rate}
                  unit=" per hour"
                />
              </Chart>
              <Chart title="By local solar time, TGFs per hour">
                <Bars
                  label="TGFs per hour of exposure by local solar time"
                  bins={rateBins(lstEvents, exposure.lst, (k) => `${(k / 2).toFixed(1)} to ${(k / 2 + 0.5).toFixed(1)} h`, (k) => (k % 12 === 0 ? `${k / 2} h` : undefined))}
                  format={rate}
                  unit=" per hour"
                />
              </Chart>
            </div>
          )}
        </div>
      </section>

      <section className="margin-grid section">
        <header className="margin-note caption">
          <h2 className="section-title mb-3">Events</h2>
          <p>Duration is the best interval of the search, from the first to the last count of the most significant window. Counts are the events in that interval.</p>
        </header>
        {cat ? (
          <div className="grid grid-cols-1 gap-12 lg:grid-cols-2">
            <Chart title="Duration">
              <Bars label="Number of TGFs by duration" bins={logBins(col<number>("duration_us").filter((d) => d > 0), 10, 1000, 8, fmtUs)} endTick="1 ms" unit=" TGFs" />
            </Chart>
            <Chart title="Counts">
              <Bars label="Number of TGFs by counts" bins={logBins(col<number>("count"), 10, 1000, 8, (x) => String(Math.round(x)))} endTick="1000" unit=" TGFs" />
            </Chart>
          </div>
        ) : (
          <Loading error={null} what="the catalog" />
        )}
      </section>

      <section className="margin-grid section pb-10">
        <header className="margin-note caption">
          <h2 className="section-title mb-3">Selection</h2>
          <p>
            What each step of the selection removed in the current run. The criteria are described on the <Link href="/about/">About</Link> page.
          </p>
        </header>
        <div className="grid grid-cols-1 gap-12 lg:grid-cols-2">
          <div className="table-scroll">
            <table className="data tabular">
              <thead>
                <tr>
                  <th>Step</th>
                  <th style={{ textAlign: "right" }}>Removed</th>
                  <th style={{ textAlign: "right" }}>Left</th>
                </tr>
              </thead>
              <tbody>
                {funnel.map(([k, removed, left]) => (
                  <tr key={k}>
                    <td className="secondary" style={{ whiteSpace: "normal" }}><Sup text={k} /></td>
                    <td style={{ textAlign: "right" }}>{removed == null ? "" : removed.toLocaleString("en")}</td>
                    <td style={{ textAlign: "right" }}>{left == null ? "…" : left.toLocaleString("en")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3 className="label pb-2" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>This run</h3>
            <dl className="caption tabular">
              {[
                ["Data", `${inst.first_day} – ${inst.last_day}`],
                ["Live time", `${yr.toFixed(2)} yr of ${years(inst.searched_seconds).toFixed(2)} yr searched`],
                ["Catalog made", `${utc(inst.generated_at, 0)} UTC`],
                ["Pipeline version", inst.blink_git],
                ...(sel
                  ? [
                      ["Train", `more than ${sel.train_threshold} candidates within ±${Number(sel.train_half_window_s) / 60} min`],
                      ["Gamma-ray burst", String(sel.grb_rule)],
                      ["Electron beam", `P < ${sci(Number(sel.electron_beam_p), 0).replace(/^1×/, "")}`],
                    ]
                  : []),
              ].map(([k, v]) => (
                <div key={k} className="flex gap-4 py-2" style={{ borderBottom: "0.5px solid var(--border)" }}>
                  <dt className="muted" style={{ width: "9em", flex: "none" }}>{k}</dt>
                  <dd style={{ color: "var(--text-primary)" }}><Sup text={v} /></dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </section>
    </div>
  );
}
