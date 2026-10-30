"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import EventMap, { type MapEvent, type Region } from "@/components/EventMap";
import { COVERAGE } from "@/lib/coverage";
import EventTable, { type Ev } from "@/components/EventTable";
import RangeBrush from "@/components/RangeBrush";
import { ago, Loading } from "@/components/Shell";
import { BASE, displayName, eventHref, lat as fmtLat, lon as fmtLon, sci } from "@/lib/format";

import type { Catalog, Index } from "@/lib/types";
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

/** "2023"、"2023-05"、"2023-05-14" → 该年/月/日的 [起, 止)，UTC 毫秒；不是日期则 null */
function dateSpan(t: string): [number, number] | null {
  const m = t.trim().match(/^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$/);
  if (!m) return null;
  const y = +m[1], mo = m[2] ? +m[2] - 1 : null, d = m[3] ? +m[3] : null;
  if (mo == null) return [Date.UTC(y, 0, 1), Date.UTC(y + 1, 0, 1)];
  if (d == null) return [Date.UTC(y, mo, 1), Date.UTC(y, mo + 1, 1)];
  return [Date.UTC(y, mo, d), Date.UTC(y, mo, d + 1)];
}

/** 搜索框：TGF 名字跳到事件页；日期、月、年或 "A to B" 设为时间范围 */
function parseQuery(q: string): { name: string } | { span: [number, number] } | null {
  const t = q.trim();
  const name = t.match(/^(?:tgf\s*)?(\d{6}[a-z]{2})$/i);
  if (name) return { name: name[1].toUpperCase() };
  const parts = t.split(/\s*(?:to|–|—|\.\.)\s*|\s+-\s+/i);
  if (parts.length === 2) {
    const a = dateSpan(parts[0]), b = dateSpan(parts[1]);
    if (a && b && a[0] < b[1]) return { span: [a[0], b[1]] };
    return null;
  }
  const one = dateSpan(t);
  return one ? { span: one } : null;
}

const regionText = ([w, e, s, n]: Region) => `${fmtLat(s).replace(".00", "")} to ${fmtLat(n).replace(".00", "")}, ${fmtLon(w).replace(".00", "")} to ${fmtLon(e).replace(".00", "")}`;

/** 选中事件的全部列，CSV */
function downloadCsv(catalogs: Catalog[], keys: Set<string>) {
  const cols = Object.keys(catalogs[0].columns);
  const esc = (v: unknown) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const lines = [["instrument", ...cols].join(",")];
  for (const cat of catalogs) {
    for (let i = 0; i < cat.n; i++) {
      if (keys.has(cat.columns.cand_id[i] as string)) lines.push([cat.instrument, ...cols.map((c) => cat.columns[c]?.[i])].map(esc).join(","));
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  a.download = "tgf-catalog-selection.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function Home() {
  const router = useRouter();
  const { data: index, error } = useJson<Index>("index.json");
  const catalogs = useCatalogs(index?.instruments.map((i) => i.slug) ?? []);
  const [axis, setAxis] = useState<Axis>("startMs");
  const [detector, setDetector] = useState<string>("all");
  const [sels, setSels] = useState<Partial<Record<Axis, [number, number]>>>({});
  const [region, setRegion] = useState<Region | null>(null);
  const [query, setQuery] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const setSel = (a: Axis, v: [number, number] | null) =>
    setSels((s) => {
      const out = { ...s };
      if (v) out[a] = v;
      else delete out[a];
      return out;
    });

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
  const picked = useMemo(
    () =>
      events.filter(
        (e) =>
          (Object.keys(sels) as Axis[]).every((a) => e[a] >= sels[a]![0] && e[a] < sels[a]![1]) &&
          (!region || (e.lon >= region[0] && e.lon <= region[1] && e.lat >= region[2] && e.lat <= region[3])),
      ),
    [events, sels, region],
  );
  const filtered = Object.keys(sels).length > 0 || region != null;
  const highlight = filtered ? new Set(picked.map((e) => e.key)) : null;

  if (!index) return <Loading error={error} what="the catalog index" />;
  const many = index.instruments.length > 1;
  const nFinal = index.instruments.reduce((a, i) => a + i.n_final, 0);
  const first = index.instruments.map((i) => i.first_day).sort()[0];
  const last = index.instruments.map((i) => i.last_day).sort().slice(-1)[0];
  const newest = [...all].filter((e) => e.name).sort((a, b) => b.startMs - a.startMs)[0];

  const search = () => {
    const q = parseQuery(query);
    if (!q) return setNote("Type a name such as 240106AA, a date such as 2023-05-14, a month, a year, or two dates joined by “to”.");
    if ("name" in q) {
      const hit = all.find((e) => e.name === q.name);
      if (!hit) return setNote(`No TGF is named ${q.name}.`);
      return router.push(eventHref({ name: hit.name, cand_id: hit.cand }));
    }
    setNote(null);
    setSel("startMs", q.span);
    setAxis("startMs");
    document.getElementById("find")?.scrollIntoView({ behavior: "smooth" });
  };

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
      ...(many ? [["Detector", e.instrument] as [string, string]] : []),
    ],
  }));
  const files = index.instruments.flatMap((i) => [
    { href: `${BASE}/data/${i.slug}/catalog.csv`, name: "catalog.csv", what: `${i.instrument}: one row per TGF, all columns` },
    { href: `${BASE}/data/${i.slug}/catalog.json`, name: "catalog.json", what: "the same, as JSON" },
    { href: `${BASE}/data/${i.slug}/exposure.json`, name: "exposure.json", what: "live time of the search in 5° cells, by local time and by month" },
    { href: `${BASE}/data/${i.slug}/manifest.json`, name: "manifest.json", what: "pipeline version, selection constants and the counts of every step" },
  ]);

  return (
    <div className="flex flex-col" style={{ gap: 88 }}>
      <section className="margin-grid pt-14">
        <div />
        <div>
          <h1 style={{ fontSize: "clamp(32px, 4.2vw, 46px)", lineHeight: 1.12, fontWeight: 400, letterSpacing: "-0.01em", maxWidth: "15em" }}>
            Terrestrial gamma-ray flashes, catalogued as the data arrive
          </h1>
          <p className="mt-6 secondary prose" style={{ maxWidth: "34em", fontSize: 18, lineHeight: 1.6 }}>
            {nFinal.toLocaleString("en")} TGFs found by {index.instruments.map((i) => i.instrument).join(", ")} from {first} to{" "}
            {last}, updated every hour. Each has a permanent name, its light curves and the exposure of the search.
          </p>
          <form
            className="mt-10 flex items-baseline gap-4 flex-wrap"
            style={{ maxWidth: "36em" }}
            onSubmit={(e) => {
              e.preventDefault();
              search();
            }}
          >
            <input
              className="search"
              aria-label="Find a TGF by name or date"
              placeholder="Name or date: 240106AA, 2023-05-14, 2023-05 to 2023-08"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setNote(null);
              }}
            />
            <button type="submit" className="link-btn">Find</button>
          </form>
          {note && <p className="caption mt-2">{note}</p>}
          <p className="caption mt-8 flex gap-x-6 gap-y-1 flex-wrap">
            {newest && (
              <span>
                Latest <Link href={eventHref({ name: newest.name, cand_id: newest.cand })}>{displayName(newest.name!)}</Link>, {newest.start.slice(0, 10)}
              </span>
            )}
            <span>Updated {ago(index.generated_at)}</span>
            <a href="#data">Download the catalog</a>
            <a href="#cite">How to cite</a>
          </p>
        </div>
      </section>

      <section id="find" className="margin-grid section">
        <header className="margin-note caption">
          <h2 className="section-title mb-3">Find events</h2>
          <p>Drag on the map to select a region, or across the bars to select a span of time. The table lists what is selected.</p>
          <p className="mt-3">The tint on the map is where the detectors observe: within the inclination of the orbit, 43° for Insight-HXMT, and outside the South Atlantic Anomaly, where they are switched off.</p>
          {many && (
            <span className="flex flex-col gap-1 mt-4">
              {["all", ...index.instruments.map((i) => i.instrument)].map((d) => (
                <button key={d} className="opt self-start" aria-pressed={detector === d} onClick={() => setDetector(d)}>
                  {d === "all" ? "All detectors" : d}
                </button>
              ))}
            </span>
          )}
        </header>
        <div className="flex flex-col gap-8">
          {catalogs ? (
            <EventMap events={mapEvents} highlight={highlight} coverage={(index.instruments ?? []).filter((i) => (detector === "all" || i.instrument === detector) && COVERAGE[i.instrument]).map((i) => COVERAGE[i.instrument])} region={region} onRegion={setRegion} />
          ) : (
            <Loading error={null} what="the catalog" />
          )}
          {catalogs && (
            <div className="flex flex-col gap-3">
              <div className="flex items-baseline justify-between gap-6 flex-wrap caption">
                <span className="flex items-baseline gap-5">
                  <span className="muted">Bars by</span>
                  {AXES.map((a) => (
                    <button key={a.key} className="opt" aria-pressed={axis === a.key} onClick={() => setAxis(a.key)}>{a.title.toLowerCase()}</button>
                  ))}
                </span>
                {axis === "startMs" && (
                  <span className="flex items-baseline gap-2">
                    <DateField
                      label="From"
                      value={sels.startMs ? iso(sels.startMs[0]) : ""}
                      onCommit={(d) => setSel("startMs", d == null ? null : [d, Math.max(sels.startMs ? sels.startMs[1] : d + DAY, d + DAY)])}
                    />
                    <span className="muted">to</span>
                    <DateField
                      label="To"
                      value={sels.startMs ? iso(sels.startMs[1] - DAY) : ""}
                      onCommit={(d) => setSel("startMs", d == null ? null : [Math.min(sels.startMs ? sels.startMs[0] : d, d), d + DAY])}
                    />
                  </span>
                )}
              </div>
              <RangeBrush
                key={axis + detector}
                kind={axis === "startMs" ? "time" : "log"}
                values={values}
                selection={sels[axis] ?? null}
                onChange={(v) => setSel(axis, v)}
                tick={(v) => tick(axis, v)}
                bins={axis === "startMs" ? 112 : 80}
              />
            </div>
          )}
          <div className="flex items-baseline justify-between gap-6 flex-wrap pt-2" style={{ borderTop: "0.5px solid var(--border)" }}>
            <p style={{ fontSize: 16 }}>
              {filtered ? (
                <>
                  <span style={{ color: "var(--accent)" }}>{picked.length.toLocaleString("en")}</span>
                  <span className="secondary"> of {events.length.toLocaleString("en")} TGFs</span>
                </>
              ) : (
                <span className="secondary">All {events.length.toLocaleString("en")} TGFs, newest first</span>
              )}
            </p>
            {catalogs && (
              <button className="link-btn caption" onClick={() => downloadCsv(catalogs, new Set(picked.map((e) => e.key)))}>
                Download {filtered ? "these" : "all"} as CSV
              </button>
            )}
          </div>
          {filtered && (
            <div className="caption flex gap-x-6 gap-y-1 flex-wrap -mt-5">
              {(Object.keys(sels) as Axis[]).map((a) => (
                <span key={a}>
                  {rangeText(a, sels[a]!)} <button className="link-btn" aria-label="Remove" onClick={() => setSel(a, null)}>×</button>
                </span>
              ))}
              {region && (
                <span>
                  {regionText(region)} <button className="link-btn" aria-label="Remove" onClick={() => setRegion(null)}>×</button>
                </span>
              )}
              <button className="link-btn" onClick={() => { setSels({}); setRegion(null); }}>Clear all</button>
            </div>
          )}
          <EventTable events={picked} showDetector={many} />
        </div>
      </section>

      <section id="data" className="margin-grid section pb-10">
        <header className="margin-note caption">
          <h2 className="section-title mb-3">Data</h2>
          <p>The files behind this page, rewritten at every update. Versioned releases are archived with a DOI.</p>
        </header>
        <div className="grid gap-12 lg:grid-cols-2">
          <div>
            <h3 className="label pb-2" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>Files</h3>
            <ul>
              {files.map((f) => (
                <li key={f.href} className="py-2.5" style={{ borderBottom: "0.5px solid var(--border)" }}>
                  <a href={f.href} download>{f.name}</a>
                  <span className="caption"> — {f.what}</span>
                </li>
              ))}
              <li className="py-2.5" style={{ borderBottom: "0.5px solid var(--border)" }}>
                <a href={`${BASE}/data/names.json`} download>names.json</a>
                <span className="caption"> — the registry of names: every name, when it was given, and its detections</span>
              </li>
            </ul>
            <p className="caption mt-3">
              The columns are described on the <Link href="/about/#columns">About</Link> page. Light curves and the vetting of each TGF are on its page and in <span className="tabular">events/&lt;id&gt;.json</span>.
            </p>
          </div>
          <div className="flex flex-col gap-8">
            <div>
              <h3 className="label pb-2" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>Names and status</h3>
              <p className="caption mt-3">
                A TGF is named <span className="tabular" style={{ color: "var(--text-primary)" }}>TGF yymmddAA</span> by the UTC date of its peak; the letters, AA, AB and
                so on, count the TGFs of that day in the order they are named. A name is given when the event is <em>final</em>, when all the data its
                selection needs are in, and it never changes. A <em>provisional</em> event still waits for data, for example a
                neighbouring hour.
              </p>
            </div>
            <div id="cite">
              <h3 className="label pb-2" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>How to cite</h3>
              <p className="caption mt-3">
                Cite the paper that describes the catalog and the version you used; the DOI of each version is given with
                the release. Refer to single events by their names.
              </p>
              <p className="caption mt-2 muted">Paper and DOI: to be added at the first release.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
