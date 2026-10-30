"use client";

import Link from "next/link";
import { useState } from "react";
import { COVERAGE } from "@/lib/coverage";
import { BASE, instrumentHref, lat, lon, sci, utc } from "@/lib/format";
import type { EventRecord } from "@/lib/types";
import { useJson } from "@/lib/useData";
import { BIN_CHOICES, ContextLightCurve, EventLightCurve } from "./LightCurve";
import Locator from "./Locator";
import Sup from "./Sup";

const PENDING: Record<string, string> = {
  not_vetted: "the candidate has not been vetted yet",
  grb_background_gap: "the data around it have a gap, so the gamma-ray burst test is undetermined",
  neighbors_unsearched: "not all data within ±10 min have been searched, so the train test is incomplete",
};

type Selection = Record<string, number | string | boolean>;

function us(t: string, peak: string): number {
  // 两个 UTC 纳秒串之差（µs），避开 Date 的毫秒精度
  const ns = (s: string) => {
    const [d, f = ""] = s.replace("Z", "").split(".");
    return BigInt(Date.parse(`${d}Z`)) * 1_000_000n + BigInt(f.padEnd(9, "0").slice(0, 9));
  };
  return Number(ns(t) - ns(peak)) / 1000;
}


const sup = (v: React.ReactNode) => (typeof v === "string" ? <Sup text={v} /> : v);
const pow = (x: number) => sci(x, 0).replace(/^1×/, "");
/** 概率：不太小时写小数 */
const prob = (p: number) => (p >= 0.01 ? p.toFixed(2) : sci(p, 1));

function Figure({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="tabular" style={{ fontSize: 40, lineHeight: 1.1 }}>{value}</div>
      <div className="caption mt-1">{label}</div>
    </div>
  );
}

function Section({ title, note, children }: { title: string; note?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="margin-grid section">
      <header className="margin-note caption">
        <h2 className="section-title mb-3">{title}</h2>
        {note}
      </header>
      <div>{children}</div>
    </section>
  );
}

/**
 * 一次探测的页面主体：页头由调用方给（名字页或候选页），其后是关键数字、光变曲线、前后 ±2 s、
 * 位置、各项检验与记录。
 */
export default function EventView({ event, head }: { event: EventRecord; head: React.ReactNode }) {
  const lc = event.light_curve;
  const [binUs, setBinUs] = useState(20);
  const { data: manifest } = useJson<{ selection: Selection }>(`${event.slug}/manifest.json`);
  const sel = manifest?.selection;
  const d = event.duration_us;
  const acdShare = (a: number | null, n: number | null) => (a != null && n ? `${((100 * a) / n).toFixed(0)} %` : "–");

  const tests: [string, React.ReactNode, React.ReactNode][] = [
    ["False-alarm rate", `${sci(event.false_positive_per_year, 1)} per year`, sel ? `at most ${pow(Number(sel.max_false_positive_per_year))}` : ""],
    ["Train", event.neighbors_10min == null ? "–" : `${event.neighbors_10min} other candidates within ±10 min`, sel ? `at most ${sel.train_threshold}` : ""],
    [
      "Gamma-ray burst",
      event.grb_z == null ? "undetermined" : `Z = ${event.grb_z.toFixed(1)}, largest box share ${event.grb_max_box_share?.toFixed(2)}`,
      sel ? `a burst if ${String(sel.grb_rule)}` : "",
    ],
    [
      "Electron beam",
      event.beam_p == null ? "–" : `P = ${prob(event.beam_p)}${event.electron_beam ? ", flagged" : ""}`,
      sel ? `flagged if P < ${pow(Number(sel.electron_beam_p))}` : "",
    ],
    [
      "Anti-coincidence",
      event.n == null ? "–" : `${event.n_acd} of ${event.n} counts (${acdShare(event.n_acd, event.n)}); ${event.n_acd_multi} with two or more shields`,
      event.n_bg ? `background ${event.n_acd_bg} of ${event.n_bg} (${acdShare(event.n_acd_bg, event.n_bg)})` : "",
    ],
    ["Detector share", event.det_share_max?.toFixed(2) ?? "–", "largest share of the counts from one detector"],
  ];

  const record: [string, React.ReactNode][] = [
    ["Detector", <Link key="i" href={instrumentHref(event.slug)}>{event.instrument}</Link>],
    ["Peak", `${utc(event.peak, 6)} UTC`],
    ["Best interval", `${d.toFixed(1)} µs, centred on the peak`],
    ["Candidate window", `${utc(event.start, 6)} – ${utc(event.stop, 6).slice(11)} UTC`],
    ["Candidate ID", <span key="c" className="tabular" style={{ overflowWrap: "anywhere" }}>{event.cand_id}</span>],
    ["Data", <a key="d" href={`${BASE}/data/${event.slug}/events/${event.cand_id}.json`} download>{event.cand_id}.json</a>],
  ];

  return (
    <div className="flex flex-col" style={{ gap: 88 }}>
      <section className="margin-grid pt-14">
        <div />
        <div>
          {head}
          {event.status !== "final" && event.pending && (
            <p className="caption mt-4" style={{ maxWidth: "36em" }}>
              <span className="badge mr-2">Provisional</span>
              {(() => {
                const why = event.pending.split("|").map((p) => PENDING[p] ?? p).join("; ");
                return why.charAt(0).toUpperCase() + why.slice(1);
              })()}
              . It is named once the selection is complete.
            </p>
          )}
          <div className="mt-10 grid gap-8" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", maxWidth: 820 }}>
            <Figure value={`${Math.round(d)} µs`} label="duration" />
            <Figure value={event.count} label="counts" />
            <Figure value={<Sup text={sci(event.false_positive_per_year, 1)} />} label="false alarms per year" />
            <Figure value={event.solar_time ? event.solar_time.slice(0, 5) : "–"} label="local solar time" />
          </div>
        </div>
      </section>

      <Section
        title="Light curve"
        note={
          <>
            <p>Counts within 1 ms of the candidate window (shaded); the dashed lines bound the best interval.</p>
            <p className="mt-3">In full ink: counts with an anti-coincidence shield firing. Below: the energy channel of each count, filled when a shield fired.</p>
            {lc && (
              <p className="mt-4 flex gap-4 flex-wrap tabular">
                {BIN_CHOICES.map((b) => (
                  <button key={b} className="opt" aria-pressed={binUs === b} onClick={() => setBinUs(b)}>{b} µs</button>
                ))}
              </p>
            )}
          </>
        }
      >
        {lc ? (
          <>
            <EventLightCurve lc={lc} binUs={binUs} window={[us(event.start, lc.peak), us(event.stop, lc.peak)]} best={[-d / 2, d / 2]} />
            <p className="caption mt-1 text-right">Time from the peak (ms)</p>
          </>
        ) : (
          <p className="caption">No light curve yet: the candidate has not been vetted.</p>
        )}
      </Section>

      {lc && (
        <Section
          title="Two seconds around"
          note={<p>Counts per 10 ms. A gamma-ray burst would show as emission in the shaded interval, 0.1 s before to 0.3 s after the candidate.</p>}
        >
          <ContextLightCurve lc={lc} />
          <p className="caption mt-1 text-right">Time from the candidate (s)</p>
        </Section>
      )}

      <Section
        title="Position"
        note={
          <p className="tabular">
            Sub-satellite point at the peak: {lat(event.latitude)}, {lon(event.longitude)}, {event.altitude_km.toFixed(0)} km up. The tint is where the detectors observe.
          </p>
        }
      >
        <Locator lon={event.longitude} lat={event.latitude} coverage={COVERAGE[event.instrument] ?? null} />
      </Section>

      <Section title="Tests" note={<p>The values behind the selection, against the rule of each test.</p>}>
        <div className="table-scroll">
          <table style={{ minWidth: 520 }} className="data wrap">
            <thead>
              <tr>
                <th>Test</th>
                <th>This event</th>
                <th>Rule</th>
              </tr>
            </thead>
            <tbody>
              {tests.map(([k, v, r]) => (
                <tr key={k}>
                  <td className="secondary" style={{ whiteSpace: "nowrap", overflowWrap: "normal" }}>{k}</td>
                  <td className="tabular">{sup(v)}</td>
                  <td className="secondary" style={{ fontSize: 14.5 }}>{sup(r)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Record">
        <dl className="caption tabular pb-10" style={{ maxWidth: 760 }}>
          {record.map(([k, v]) => (
            <div key={k} className="flex gap-4 py-2" style={{ borderBottom: "0.5px solid var(--border)" }}>
              <dt className="muted" style={{ width: "9.5em", flex: "none" }}>{k}</dt>
              <dd style={{ color: "var(--text-primary)", minWidth: 0 }}>{v}</dd>
            </div>
          ))}
        </dl>
      </Section>
    </div>
  );
}
